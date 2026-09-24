# ColPali RAG

Production-grade RAG over **visually rich documents** — retrieve PDF pages by their
*appearance* using ColPali (VLM late-interaction MaxSim) fused with BGE-M3 dense +
sparse text retrieval, then answer grounded in the retrieved pages via a free-tier
local kilo LLM (OmniRoute gateway).

Runs **100% locally, bare metal, no Docker, no paid tokens**:
ColPali, BGE-M3 and OCR are local models on your GPU; only the final answer text
goes to your OmniRoute kilo model.

| | |
|---|---|
| Visual retrieval | `vidore/colqwen2-v1.0-hf` — per-patch 128-d vectors, Qdrant `late_interaction` MaxSim |
| Text retrieval | `BAAI/bge-large-en-v1.5` dense (1024-d) + `Qdrant/bm25` sparse via fastembed |
| Fusion | Reciprocal Rank Fusion (Qdrant server-side, python RRF fallback) |
| Vector store | Qdrant local mode (single writer process = the API server) |
| Generation | Pluggable: `TextGenerator` (grounded in page text, free kilo model), `VisionGenerator` slot |
| Tracked | per-ingest + per-eval runs in MLflow (`data/mlruns`) |

## Quickstart

```powershell
# 1) one-time setup: install deps + generate demo corpus (figures + text PDFs)
.\powershell\setup.ps1

# 2) start the app (owns the Qdrant index; first start downloads models into data/hf-cache)
.\powershell\start-app.ps1      # then open http://localhost:8000 (chat UI)

# 3) ingest PDFs through the API (single-writer rule)
.\powershell\ingest.ps1 -Path .\data\corpus

# or all-in-one
.\powershell\demo.ps1
```

Ask in the UI, e.g. *"What does the revenue chart show for each product line?"* —
the answer is grounded in the retrieved page with `[file.pdf (page N)]` citations.

## API

| Endpoint | Description |
|---|---|
| `POST /v1/query` | `{query, top_k, generate}` → answer + citations + retrieved pages (with page images) |
| `POST /v1/ingest` | multipart PDF upload → async job, poll `GET /v1/ingest/{job_id}` |
| `GET /v1/health` | qdrant + models + generator status |
| `GET /v1/collection` | point count |
| `DELETE /v1/collection` | wipe index (auth required if `API_KEY` set) |
| `GET /docs` | OpenAPI/Swagger |

## Config

`.env` (copy from `.env.example`) — the only thing you must set is the OmniRoute
gateway for generation:

```
OMNIRoute_BASE_URL=http://localhost:8080/v1
OMNIRoute_API_KEY=local
OMNIRoute_MODEL=kilo
```

Retrieval/ingest tunables live in `config.yaml`. The app still serves retrieval +
citations if the gateway is unreachable (the answer field reports the reason).

## Evaluation

```powershell
uv run python -m colpali_rag.eval            # hybrid mode
uv run python -m colpali_rag.eval --mode colpali   # visual leg alone
```

Golden queries live in `evals/golden_queries.json`; metrics: hit@5, MRR, NDCG@5.
Runs are logged to MLflow (`data/mlruns`).

## Architecture

```
PDF -> render pages (PyMuPDF) -> ColPali per-patch vectors -> Qdrant "colpali" (MaxSim)
     -> text layer (embedded text / OCR) -> BGE-M3 dense + sparse -> Qdrant
query -> same three legs -> RRF fusion -> top pages -> TextGenerator -> kilo LLM -> answer
```

Notes:
- **Single writer**: Qdrant runs in local mode inside the API process. Ingest PDFs
  through `POST /v1/ingest` (or `ingest.ps1`), never a parallel CLI, to avoid
  file locks. An offline CLI exists (`python -m colpali_rag.ingest`) for
  batch-initializing an index before the server starts.
- **GPU**: fits 6 GB VRAM fp16; switch `colpali_model` for larger cards
  (e.g. `vidore/colpali-v1.3`). `device=auto` picks CUDA when available.
- **OCR**: `OCR_ENABLED=true` OCRs pages with no embedded text (rapidocr, local).
- **Vision answering**: set `GENERATION_MODE=vision` and implement
  `VisionGenerator.generate` to feed page images to a vision LLM.

## Tests

```powershell
uv run pytest
```

(API tests stub the embedders; they do not download models.)