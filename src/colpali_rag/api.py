"""FastAPI application: query + ingest endpoints, health, minimal chat UI.

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
from .generator import Generation, get_generator
from .ingest import IngestPipeline, IngestReport
from .qdrant_store import QdrantStore
from .retrieve import Retriever

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
        }

    @api.get("/health", response_model=dict)
    def health():
        return runtime.health()

    @api.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest, _: Any = Depends(require_auth())):
        t0 = time.perf_counter()
        result = runtime.retriever.retrieve(req.query, req.top_k)
        pages = result.pages
        response = QueryResponse(
            query=req.query,
            pages=[_page_dict(p) for p in pages],
            citations=[p.citation for p in pages[: settings.generation_top_pages]],
        )
        if req.generate and pages:
            gen: Generation = runtime.generator.generate(req.query, pages)
            response.answer = gen.answer
            response.generation_model = gen.model
            response.generation_backend = f"{gen.backend}"
            response.generation_note = gen.note
            response.citations = gen.citations or response.citations
        runtime._timings["query"] = time.perf_counter() - t0
        return response

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

    app.include_router(api)

    # ---- UI + static page images --------------------------------------
    settings.pages_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=str(settings.pages_dir)), name="assets")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return UI_HTML
    return app


UI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ColPali RAG</title>
<style>
  :root { color-scheme: dark; --bg:#0b0f14; --panel:#131a22; --line:#223042; --txt:#dbe6f2; --mut:#7f93a8; --acc:#3ba55d; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, Segoe UI, sans-serif; background:var(--bg); color:var(--txt); }
  header { padding:14px 22px; border-bottom:1px solid var(--line); display:flex; align-items:baseline; gap:12px; }
  header h1 { font-size:17px; margin:0; } header span { color:var(--mut); font-size:12px; }
  main { max-width:920px; margin:18px auto 40px; padding:0 16px; }
  #ask { display:flex; gap:8px; margin-bottom:14px; }
  #ask input { flex:1; background:var(--panel); border:1px solid var(--line); color:var(--txt); padding:11px 13px; border-radius:8px; font-size:14px; }
  #ask button { background:var(--acc); border:0; color:#04110a; font-weight:700; padding:11px 18px; border-radius:8px; cursor:pointer; }
  #ask button:disabled { opacity:.5; cursor:wait; }
  #status { color:var(--mut); font-size:12px; min-height:18px; margin-bottom:12px; }
  #answer { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; white-space:pre-wrap; line-height:1.55; display:none; }
  #answer h3 { margin:0 0 8px; font-size:13px; color:var(--mut); text-transform:uppercase; letter-spacing:.08em; }
  #pages { display:grid; grid-template-columns:repeat(auto-fill, minmax(230px,1fr)); gap:12px; margin-top:18px; }
  .page { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .page img { width:100%; display:block; border-bottom:1px solid var(--line); background:#fff; }
  .page div { padding:9px 11px; font-size:12px; }
  .page b { color:var(--acc); }
  .empty { color:var(--mut); font-size:13px; }
</style>
</head>
<body>
<header><h1>ColPali RAG</h1><span>visual + hybrid document retrieval</span></header>
<main>
  <div id="ask"><input id="q" placeholder="Ask about your documents… e.g. what does the revenue chart show?" />
  <button id="go" onclick="ask()">Ask</button></div>
  <div id="status"></div>
  <div id="answer"></div>
  <div id="pages"></div>
</main>
<script>
async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const go = document.getElementById('go'), st = document.getElementById('status'),
        ans = document.getElementById('answer'), pages = document.getElementById('pages');
  go.disabled = true; st.textContent = 'retrieving…'; pages.innerHTML = ''; ans.style.display = 'none';
  try {
    const r = await fetch('/v1/query', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ query: q, top_k: 6, generate: true }) });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    st.textContent = 'retrieved ' + d.pages.length + ' pages · generation: ' + (d.answer ? 'ok' : ('off — ' + (d.generation_note||'backend not configured')));
    ans.style.display = 'block';
    ans.innerHTML = '<h3>answer</h3>' + (d.answer && d.answer !== '""' ? d.answer.replace(/&/g,'&amp;').replace(/</g,'&lt;') : '<span class="empty">' + (d.generation_note || 'No answer (no pages retrieved).') + '</span>');
    if (!d.pages.length) { pages.innerHTML = '<div class="empty">No pages retrieved — ingest documents first.</div>'; return; }
    d.pages.forEach(p => {
      const c = document.createElement('div'); c.className='page';
      c.innerHTML = (p.image? '<img src="/assets/' + p.image + '" loading="lazy">' : '') +
        '<div><b>' + p.src + ' · p' + p.page + '</b> · ' + p.score.toFixed(3) +
        '<br><span class="empty">' + (p.text ? p.text.slice(0,140) : 'no text layer') + '…</span></div>';
      pages.appendChild(c);
    });
  } catch (e) { st.textContent = 'error: ' + e.message; }
  finally { go.disabled = false; }
}
document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
</script>
</body>
</html>
"""


def main() -> None:
    import uvicorn

    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    configure_hf_env(settings)
    app = create_app(settings)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()