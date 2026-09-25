"""FastAPI application: query + ingest endpoints, health, chat UI with document upload.

Runs the single Qdrant writer process (local mode). Ingestion must therefore
go through this API (POST /v1/ingest), not through a separate CLI, when the
server is running.
"""
from __future__ import annotations

import logging
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config import Settings, configure_hf_env, get_settings
from .embeddings import ColPaliEmbedder, TextEmbedder
from .generator import Generation, generate_with_fallback, get_generator
from .ingest import IngestPipeline, IngestReport
from .qdrant_store import QdrantStore
from .retrieve import Retriever
from .ui import UI_HTML

log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
    generate: bool = Field(default=True)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class QueryResponse(BaseModel):
    query: str
    answer: str = ""
    generation_model: str = ""
    generation_backend: str = ""
    generation_note: str = ""
    citations: list[str] = []
    pages: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    trace: dict[str, Any] | None = None  # retrieval fusion trace (inspector UI)


class IngestStatus(BaseModel):
    job_id: str
    status: str
    report: dict[str, Any] | None = None


class AppRuntime:
    """Lazy singletons shared by endpoints."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.colpali = ColPaliEmbedder(settings)
        self.text = TextEmbedder(settings)
        self.store = QdrantStore(settings)
        self.pipeline = IngestPipeline(settings, self.store, self.colpali, self.text)
        self.retriever = Retriever(settings, self.store, self.colpali, self.text)
        self.generator = get_generator(settings)
        self._jobs: dict[str, IngestStatus] = {}
        self._lock = threading.Lock()
        self._timings: dict[str, float] = {}

    def health(self) -> dict[str, Any]:
        try:
            points = self.store.count()
            db = "ok"
        except Exception as exc:  # pragma: no cover
            points = -1
            db = f"error: {exc}"
        return {
            "status": "ok",
            "qdrant": db,
            "points": points,
            "colpali_loaded": self.colpali.is_loaded,
            "text_embed_loaded": self.text.is_loaded,
            "generation_backend": self.generator.name,
            "settings": self.settings.summary(),
        }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    configure_hf_env(settings)

    runtime = AppRuntime(settings)
    app = FastAPI(title="ColPali RAG", version="0.1.0", description="Visual + hybrid document RAG")
    app.state.runtime = runtime
    api = APIRouter(prefix="/v1")

    def require_auth():
        from fastapi import Header

        key = settings.api_key

        async def dep(x_api_key: str | None = Header(default=None)):
            if key and x_api_key != key:
                raise HTTPException(status_code=401, detail="invalid api key")

        return dep

    def _page_dict(page) -> dict[str, Any]:
        return {
            "score": round(page.score, 4),
            "src": page.src,
            "page": page.page,
            "text": (page.text or "")[:1500],
            "image": page.image if page.image else None,
            "leg": {k: round(v, 4) for k, v in page.leg.items()},
            "legs": list(page.legs),
        }

    def _run_query(req: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()
        result = runtime.retriever.retrieve(req.query, req.top_k)
        pages = result.pages
        response = QueryResponse(
            query=req.query,
            pages=[_page_dict(p) for p in pages],
            citations=[p.citation for p in pages[: settings.generation_top_pages]],
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            trace=result.trace,
        )
        if req.generate and pages:
            gen: Generation = generate_with_fallback(runtime.generator, req.query, pages)
            response.answer = gen.answer
            response.generation_model = gen.model
            response.generation_backend = f"{gen.backend}"
            response.generation_note = gen.note
            response.citations = gen.citations or response.citations
        runtime._timings["query"] = time.perf_counter() - t0
        return response

    @api.get("/health", response_model=dict)
    def health():
        return runtime.health()

    @api.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest, _: Any = Depends(require_auth())):
        try:
            return _run_query(req)
        except HTTPException:
            raise
        except Exception as exc:  # keep the chat UI informed instead of a bare 500
            log.exception("query failed")
            raise HTTPException(status_code=502, detail=f"query failed: {exc}") from exc

    @api.post("/ingest", response_model=IngestStatus, status_code=202)
    async def ingest(file: UploadFile, _: Any = Depends(require_auth())):
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="only PDF files are supported")
        job_id = uuid.uuid4().hex[:12]
        status = IngestStatus(job_id=job_id, status="queued")
        runtime._jobs[job_id] = status
        bytes_ = file.file.read()

        def worker() -> None:
            tmp = Path(tempfile.mkstemp(suffix=".pdf", prefix="ingest-")[1])
            try:
                tmp.write_bytes(bytes_)
                status.status = "running"
                report: IngestReport = runtime.pipeline.ingest_pdf(tmp, src_name=file.filename)
                status.report = report.to_dict()
                status.status = "done" if not report.errors else "partial"
            except Exception as exc:  # pragma: no cover
                log.exception("ingest job failed")
                status.status = "error"
                status.report = {"errors": [str(exc)]}
            finally:
                # Windows may briefly hold the PDF handle after fitz closes
                for _attempt in range(5):
                    try:
                        tmp.unlink(missing_ok=True)
                        break
                    except PermissionError:
                        time.sleep(0.4)

        threading.Thread(target=worker, daemon=True).start()
        return status

    @api.get("/ingest/{job_id}", response_model=IngestStatus)
    def ingest_status(job_id: str):
        status = runtime._jobs.get(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="unknown job")
        return status

    @api.delete("/collection", status_code=204)
    def wipe(_: Any = Depends(require_auth())):
        runtime.store.delete_collection()
        return None

    @api.get("/collection", response_model=dict)
    def collection_info():
        try:
            return {"collection": settings.collection, "points": runtime.store.count()}
        except Exception as exc:  # pragma: no cover
            return {"collection": settings.collection, "points": 0, "note": f"{exc}"}

    @api.get("/sources", response_model=list[dict[str, Any]])
    def sources():
        """Indexed documents: [{'src': filename, 'pages': n}, ...]."""
        return runtime.store.list_sources()

    app.include_router(api)

    # ---- UI + static page images --------------------------------------
    settings.pages_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=str(settings.pages_dir)), name="assets")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return UI_HTML
    return app


def main() -> None:
    import uvicorn

    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    configure_hf_env(settings)
    app = create_app(settings)
    runtime = app.state.runtime
    if settings.warmup_on_start:
        log.info("warmup_on_start: loading embedders before serving...")
        try:
            runtime.colpali.embed_query("warmup")
            runtime.text.embed_query("warmup")
            log.info("embedders warm: first query will be instant")
        except Exception as exc:  # pragma: no cover
            log.warning("model warmup failed (server will lazy-load on first use): %s", exc)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
