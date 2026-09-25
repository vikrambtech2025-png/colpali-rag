"""FastAPI application: query + ingest endpoints, health, chat UI with document upload.

Security: all /v1 data endpoints are auth-gated (shared API key via X-API-Key
header) when settings.api_key is set, rate-limited per client IP, and PDF
uploads are capped in size. Health/ready stay public for load balancers.
"""
from __future__ import annotations

import hmac
import logging
import logging.handlers
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config import Settings, configure_hf_env, get_settings
from .embeddings import ColPaliEmbedder, TextEmbedder
from .generator import Generation, generate_with_fallback, get_generator
from .ingest import IngestPipeline, IngestReport
from .jobs import IngestGate, JobStore, QueryGate
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
        self.jobs = JobStore(settings.jobs_db)  # durable: survives restarts
        self._jobs: dict[str, IngestStatus] = {}  # hot cache over JobStore
        self.gate = IngestGate(settings.max_concurrent_ingests)
        self.query_gate = QueryGate(settings.max_concurrent_queries)
        self._lock = threading.Lock()
        self._timings: dict[str, float] = {}
        self._boot_ts = time.time()

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
            "uptime_s": round(time.time() - self._boot_ts, 1),
            "ingests": self.gate.snapshot(),
            "settings": self.settings.summary(),
        }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    configure_hf_env(settings)

    runtime = AppRuntime(settings)
    app = FastAPI(title="ColPali RAG", version="0.3.0", description="Visual + hybrid document RAG")
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
        max_age=600,
    )
    # Fail fast at boot: create the collection (idempotent) so a mistyped cloud
    # URL / API key / collection name surfaces immediately, not on first query.
    try:
        runtime.store.ensure_collection()
    except Exception as exc:  # pragma: no cover - infra dependent
        log.error("Vector store setup failed: %s", exc)
        raise
    if settings.api_key:
        log.info("API auth ENABLED (clients must send X-API-Key)")
    else:
        log.warning("API auth DISABLED - set API_KEY=... in .env before exposing this server")
    api = APIRouter(prefix="/v1")

    # ---- rate limiting: in-memory fixed window per client IP ----
    class _RateLimiter:
        def __init__(self, per_minute: int) -> None:
            self.per_minute = per_minute
            self._hits: dict[str, list[float]] = {}
            self._lock = threading.Lock()

        def allow(self, key: str) -> bool:
            now = time.monotonic()
            cutoff = now - 60.0
            with self._lock:
                hits = [t for t in self._hits.get(key, []) if t > cutoff]
                if len(hits) >= self.per_minute:
                    self._hits[key] = hits
                    return False
                hits.append(now)
                self._hits[key] = hits
                return True

    query_limiter = _RateLimiter(settings.rate_limit_per_minute)
    ingest_limiter = _RateLimiter(settings.rate_limit_ingest_per_minute)

    def rate_limit(limiter: _RateLimiter):
        async def dep(request: Request):
            client = request.client.host if request.client else "unknown"
            if not limiter.allow(client):
                raise HTTPException(status_code=429, detail="rate limit exceeded, slow down")
            return None

        return dep

    def require_auth():
        key = settings.api_key

        async def dep(x_api_key: str | None = Header(default=None)):
            if not key:
                return None  # auth disabled: allow
            if not x_api_key or not hmac.compare_digest(x_api_key, key):
                raise HTTPException(status_code=401, detail="missing or invalid api key")
            return None

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
        with runtime.query_gate:  # cap concurrent GPU retrievals
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

    @api.get("/ready", response_model=dict)
    def ready():
        """Readiness probe: store reachable + app can serve. Models are reported
        as informational (they lazy-load per request if warmup hasn't run)."""
        problems: list[str] = []
        try:
            runtime.store.count()
        except Exception as exc:  # pragma: no cover
            problems.append(f"qdrant unreachable: {exc}")
        return {
            "ready": not problems,
            "problems": problems,
            "models_warm": bool(runtime.colpali.is_loaded and runtime.text.is_loaded),
        }

    @api.post("/query", response_model=QueryResponse)
    def query(
        req: QueryRequest,
        _: Any = Depends(require_auth()),
        __: Any = Depends(rate_limit(query_limiter)),
    ):
        try:
            return _run_query(req)
        except HTTPException:
            raise
        except Exception as exc:  # keep the chat UI informed instead of a bare 500
            log.exception("query failed")
            raise HTTPException(status_code=502, detail=f"query failed: {exc}") from exc

    @api.post("/ingest", response_model=IngestStatus, status_code=202)
    async def ingest(
        file: UploadFile,
        _: Any = Depends(require_auth()),
        __: Any = Depends(rate_limit(ingest_limiter)),
    ):
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="only PDF files are supported")
        max_bytes = settings.max_upload_mb * 1024 * 1024
        declared = int(file.headers.get("content-length") or 0)
        if declared > max_bytes:
            raise HTTPException(status_code=413, detail=f"file exceeds {settings.max_upload_mb} MB cap")
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"file exceeds {settings.max_upload_mb} MB cap")
        job_id = uuid.uuid4().hex[:12]
        status = IngestStatus(job_id=job_id, status="queued")
        runtime._jobs[job_id] = status
        runtime.jobs.create(job_id, file.filename)
        runtime.gate.acquire()
        bytes_ = data

        def worker() -> None:
            tmp = Path(tempfile.mkstemp(suffix=".pdf", prefix="ingest-")[1])
            try:
                tmp.write_bytes(bytes_)
                status.status = "running"
                runtime.jobs.update(job_id, "running")
                report: IngestReport = runtime.pipeline.ingest_pdf(tmp, src_name=file.filename)
                status.report = report.to_dict()
                status.status = "done" if not report.errors else "partial"
            except Exception as exc:  # pragma: no cover
                log.exception("ingest job failed")
                status.status = "error"
                status.report = {"errors": [str(exc)]}
            finally:
                runtime.jobs.update(job_id, status.status, status.report)
                runtime.gate.release()
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
    def ingest_status(job_id: str, _: Any = Depends(require_auth())):
        status = runtime._jobs.get(job_id)
        if status:
            return status
        # survive restarts: fall back to the durable job log
        row = runtime.jobs.get(job_id)
        if not row:
            raise HTTPException(status_code=404, detail="unknown job")
        return IngestStatus(job_id=row["job_id"], status=row["status"], report=row["report"])

    @api.delete("/collection", status_code=204)
    def wipe(_: Any = Depends(require_auth())):
        runtime.store.delete_collection()
        return None

    @api.get("/collection", response_model=dict)
    def collection_info(_: Any = Depends(require_auth())):
        try:
            return {"collection": settings.collection, "points": runtime.store.count()}
        except Exception as exc:  # pragma: no cover
            return {"collection": settings.collection, "points": 0, "note": f"{exc}"}

    @api.get("/sources", response_model=list[dict[str, Any]])
    def sources(_: Any = Depends(require_auth())):
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


def configure_logging(settings: Settings) -> None:
    """Console + rotating file logging (data/logs/app.log, gitignored)."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if settings.log_file:
        try:
            settings.log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                settings.log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            handlers.append(fh)
        except Exception as exc:  # pragma: no cover
            log.warning("file logging unavailable (%s); console only", exc)
    for h in handlers:
        h.setFormatter(fmt)
    logging.basicConfig(
        level=getattr(logging, str(settings.log_level).upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )


def main() -> None:
    import uvicorn

    settings = get_settings()
    configure_logging(settings)
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
