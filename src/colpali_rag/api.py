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

    def _run_query(req: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()
        result = runtime.retriever.retrieve(req.query, req.top_k)
        pages = result.pages
        response = QueryResponse(
            query=req.query,
            pages=[_page_dict(p) for p in pages],
            citations=[p.citation for p in pages[: settings.generation_top_pages]],
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


UI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ColPali RAG</title>
<style>
  :root { color-scheme: dark; --bg:#0b0f14; --panel:#131a22; --line:#223042; --txt:#dbe6f2; --mut:#7f93a8; --acc:#3ba55d; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, Segoe UI, sans-serif; background:var(--bg); color:var(--txt); height:100vh; display:flex; flex-direction:column; }
  header { padding:12px 22px; border-bottom:1px solid var(--line); display:flex; align-items:baseline; gap:12px; }
  header h1 { font-size:17px; margin:0; } header .tag { color:var(--mut); font-size:12px; }
  #stats { margin-left:auto; color:var(--mut); font-size:12px; }
  #upload { padding:12px 22px; border-bottom:1px solid var(--line); display:flex; flex-direction:column; gap:8px; }
  #drop { border:1px dashed var(--line); border-radius:10px; padding:14px; text-align:center; color:var(--mut); font-size:13px; cursor:pointer; transition:border-color .15s; }
  #drop.over { border-color:var(--acc); color:var(--txt); }
  #uploadbar { display:flex; gap:8px; align-items:center; }
  #file { display:none; }
  #pick { background:var(--panel); border:1px solid var(--line); color:var(--txt); padding:8px 14px; border-radius:8px; cursor:pointer; font-size:13px; }
  #upbtn { background:var(--acc); border:0; color:#04110a; font-weight:700; padding:8px 16px; border-radius:8px; cursor:pointer; }
  #upbtn:disabled { opacity:.5; cursor:wait; }
  #upstatus { color:var(--mut); font-size:12px; flex:1; }
  #sources { display:flex; flex-wrap:wrap; gap:6px; font-size:12px; }
  .chip { background:var(--panel); border:1px solid var(--line); color:var(--mut); padding:3px 9px; border-radius:999px; }
  #chat { flex:1; overflow-y:auto; padding:18px 22px; display:flex; flex-direction:column; gap:14px; }
  .msg { max-width:820px; display:flex; flex-direction:column; }
  .msg.user { align-self:flex-end; align-items:flex-end; }
  .msg.bot { align-self:flex-start; align-items:flex-start; }
  .bubble { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; white-space:pre-wrap; line-height:1.55; font-size:14px; }
  .msg.user .bubble { background:#1c2b23; border-color:#2c4a3a; }
  .meta { margin-top:8px; font-size:11px; color:var(--mut); }
  .mut { color:var(--mut); font-size:12px; margin-top:8px; }
  .empty { color:var(--mut); font-size:13px; }
  .thumbs { display:flex; gap:10px; overflow-x:auto; margin-top:10px; padding-bottom:4px; }
  .thumbs .thumb { flex:0 0 auto; }
  .thumbs img { height:110px; border-radius:6px; border:1px solid var(--line); background:#fff; cursor:pointer; display:block; }
  .thumbs .cap { font-size:10px; color:var(--mut); margin-top:3px; max-width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #composer { border-top:1px solid var(--line); padding:14px 22px; display:flex; gap:8px; align-items:flex-end; }
  #q { flex:1; background:var(--panel); border:1px solid var(--line); color:var(--txt); padding:11px 13px; border-radius:8px; font-size:14px; resize:none; font-family:inherit; }
  #go { background:var(--acc); border:0; color:#04110a; font-weight:700; padding:11px 18px; border-radius:8px; cursor:pointer; }
  #go:disabled { opacity:.5; cursor:wait; }
</style>
</head>
<body>
<header>
  <h1>ColPali RAG</h1><span class="tag">visual + hybrid document retrieval</span>
  <span id="stats">— docs</span>
</header>
<section id="upload">
  <div id="drop">Drop PDFs here or click to choose — they are indexed and you can ask about them right away</div>
  <div id="uploadbar">
    <input type="file" id="file" accept=".pdf" multiple>
    <button id="pick" onclick="document.getElementById('file').click()">choose files</button>
    <button id="upbtn" onclick="uploadFiles()">upload</button>
    <span id="upstatus"></span>
  </div>
  <div id="sources"></div>
</section>
<main id="chat"></main>
<section id="composer">
  <textarea id="q" rows="2" placeholder="Ask about your documents… e.g. what does the revenue chart show?"></textarea>
  <button id="go" onclick="ask()">Ask</button>
</section>
<script>
const $ = id => document.getElementById(id);
const esc = s => (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;');
const sleep = ms => new Promise(r => setTimeout(r, ms));

function addMsg(role, html) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = html;
  $('chat').appendChild(d);
  $('chat').scrollTop = $('chat').scrollHeight;
  return d;
}

async function refreshInfo() {
  try {
    const s = await (await fetch('/v1/sources')).json();
    const pts = await (await fetch('/v1/collection')).json();
    const docs = s.length, points = pts.points || 0;
    $('stats').textContent = docs + ' doc' + (docs === 1 ? '' : 's') + ' · ' + points + ' pages';
    $('sources').innerHTML = s.length ? s.map(x =>
      '<span class="chip">' + esc(x.src) + ' · ' + x.pages + 'p</span>').join('')
      : '<span class="chip">no documents indexed yet</span>';
    if (!docs && !$('chat').children.length) {
      addMsg('bot', '<div class="bubble"><span class="empty">Welcome. Upload a PDF above (or drop it anywhere on the box) and then ask me anything about it.</span></div>');
    }
  } catch (e) { $('stats').textContent = 'server starting…'; }
}

async function ask() {
  const q = $('q').value.trim();
  if (!q) return;
  const go = $('go');
  addMsg('user', '<div class="bubble">' + esc(q) + '</div>');
  $('q').value = '';
  go.disabled = true;
  const waitEl = addMsg('bot', '<div class="bubble"><span class="empty">retrieving…</span></div>');
  try {
    const r = await fetch('/v1/query', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ query: q, top_k: 6, generate: true }) });
    if (!r.ok) {
      const errBody = await r.json().catch(() => ({}));
      throw new Error('HTTP ' + r.status + (errBody.detail ? ' — ' + errBody.detail : '') + (r.status === 401 ? ' (server requires an API key the web UI cannot supply)' : ''));
    }
    const d = await r.json();
    const srcLabel = d.generation_backend ? esc(d.generation_backend + (d.generation_model ? ' · ' + d.generation_model : '')) : '';
    const sources = (d.citations && d.citations.length) ? '<div class="mut">sources: ' + esc(d.citations.join(' · ')) + '</div>' : '';
    const note = d.generation_note ? '<div class="mut">' + esc(d.generation_note) + '</div>' : '';
    const meta = d.answer ? '<div class="meta">answered by ' + (srcLabel || 'retrieval only') + '</div>' : '';
    const thumbs = (d.pages && d.pages.length) ? '<div class="thumbs">' + d.pages.map(p =>
        '<div class="thumb">' +
        (p.image ? '<img src="/assets/' + esc(p.image) + '" loading="lazy" title="' + esc((p.text || '').slice(0, 180)) + '" onclick="window.open(this.src)">' : '') +
        '<div class="cap">' + esc(p.src) + ' · p' + p.page + ' · ' + p.score.toFixed(3) + '</div></div>').join('')
      : '';
    const ans = (d.answer && d.answer !== '""') ? esc(d.answer)
      : '<span class="empty">' + esc(d.generation_note || 'No answer (no pages retrieved — upload a PDF first).') + '</span>';
    waitEl.innerHTML = '<div class="bubble">' + ans + meta + sources + note + '</div>' + thumbs;
  } catch (e) {
    waitEl.innerHTML = '<div class="bubble"><span class="empty">error: ' + esc(e.message) + '</span></div>';
  } finally {
    go.disabled = false;
    $('q').focus();
  }
}

async function uploadFiles(picked) {
  const files = Array.from(picked || $('file').files || []);
  if (!files.length) return;
  const btn = $('upbtn'), st = $('upstatus');
  btn.disabled = true;
  for (const f of files) {
    if (!f.name.toLowerCase().endsWith('.pdf')) { st.textContent = f.name + ' skipped (only PDF files are supported)'; continue; }
    st.textContent = 'uploading ' + f.name + '…';
    try {
      const fd = new FormData(); fd.append('file', f);
      const r = await fetch('/v1/ingest', { method:'POST', body: fd });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) { st.textContent = f.name + ' failed: ' + (body.detail || ('HTTP ' + r.status)); continue; }
      await pollJob(body.job_id, f.name);
    } catch (e) { st.textContent = f.name + ' failed: ' + e.message; }
  }
  $('file').value = '';
  btn.disabled = false;
  refreshInfo();
}

async function pollJob(id, name) {
  for (;;) {
    await sleep(1200);
    let j;
    try { j = await (await fetch('/v1/ingest/' + id)).json(); }
    catch (e) { continue; }
    if (j.status === 'done' || j.status === 'partial') {
      const rep = j.report || {};
      $('upstatus').textContent = name + ' indexed · ' + rep.pages + ' pages in ' + rep.duration_s + 's' +
        ((j.status === 'partial' && rep.errors && rep.errors.length) ? ' (' + esc(rep.errors.join('; ')) + ')' : '') +
        ' — you can ask about it now.';
      return;
    }
    if (j.status === 'error') {
      const rep = j.report || {};
      $('upstatus').textContent = name + ' failed: ' + ((rep.errors || ['unknown error']).join('; '));
      return;
    }
  }
}

const drop = $('drop');
drop.addEventListener('click', () => $('file').click());
['dragenter','dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => {
  const files = Array.from(e.dataTransfer.files || []);
  if (files.length) uploadFiles(files);
});
$('file').addEventListener('change', () => uploadFiles());
$('q').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); } });
refreshInfo();
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