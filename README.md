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
| Fusion | Reciprocal Rank Fusion (traced three-leg RRF keyed by source+page; the per-leg rankings are returned with every query for the UI inspector) |
| Vector store | Qdrant — local embedded, or your **hosted Qdrant Cloud** cluster (server-side MaxSim, concurrent reads/writes) |
| Generation | Pluggable: `TextGenerator` (grounded in page text, free kilo model), `VisionGenerator` slot |
| Tracked | per-ingest + per-eval runs in MLflow (`data/mlruns`) |

## Quickstart

```powershell
# 1) one-time setup: install deps (incl. MLflow tracking) + generate demo corpus
.\powershell\setup.ps1

# 2) start the app (owns the Qdrant index; first start downloads models into data/hf-cache)
.\powershell\start-app.ps1      # then open http://localhost:8000 (chat UI)

# 3) ingest PDFs through the API (single-writer rule)
.\powershell\ingest.ps1 -Path .\data\corpus

# or all-in-one
.\powershell\demo.ps1
```

For the **college showcase** version (scripted demo, curated chart-heavy docs,
story materials, failover notes): see [`demo/README.md`](demo/README.md). The
one-command setup is `scripts\prepare-demo.ps1` (stop app → generate demo PDFs →
ingest → start with pre-warmed models). Poster/slide material: [`demo/STORY.md`](demo/STORY.md).

Ask in the UI, e.g. *"What does the revenue chart show for each product line?"* —
the answer is grounded in the retrieved page with `[file.pdf (page N)]` citations.

## API

| Endpoint | Description |
|---|---|
| `POST /v1/query` | `{query, top_k, generate}` → answer + citations + retrieved pages (with page images), `latency_ms`, and a `trace` of the three retrieval legs + RRF fusion |
| `POST /v1/ingest` | multipart PDF upload → async job, poll `GET /v1/ingest/{job_id}` |
| `GET /v1/sources` | indexed documents: `[{"src": filename, "pages": n}, ...]` |
| `GET /v1/health` | qdrant + models + generator status |
| `GET /v1/collection` | point count |
| `DELETE /v1/collection` | wipe index (auth required if `API_KEY` set) |
| `GET /docs` | OpenAPI/Swagger |

The web UI at `/` is a chat surface: hero header with live doc/page stats and
quick-start question chips, drag & drop or pick PDFs to upload (asynchronous jobs
poll `GET /v1/ingest/{job_id}`), then ask questions. Answers show the generation
backend/model, a `sources:` line with citations, clickable page thumbnails of the
retrieved pages (colored badges = which retrieval legs found each page), and a
collapsible **retrieval inspector** visualizing encode/leg/fusion latencies and
the three per-leg rankings + fused order for that exact query.

## Config

`.env` (copy from `.env.example`) — the only thing you must set is the OmniRoute
gateway for generation:

```
OMNIRoute_BASE_URL=http://localhost:8080/v1
OMNIRoute_API_KEY=local
OMNIRoute_MODEL=kilo
```

`OMNIRoute_MODEL` must be a model id the gateway actually serves — check with
`GET <OMNIRoute_BASE_URL>/models`. Verified working on the OmniRoute `cfp`
pool: `cfp/zai-org/glm-5.2` and `cfp/deepseek-ai/deepseek-v4-pro-0813`.
Any OpenAI-compatible endpoint works (Ollama `http://localhost:11434/v1`,
LM Studio `http://localhost:1234/v1`), not just OmniRoute.

The gateway's free pools can expire or return empty replies. The app retries
(`generation_retries`, `generation_retry_delay`) and falls back to local
extractive answers when the gateway is down. To see which models are actually
alive right now:

```powershell
uv run python scripts/check_gateway.py                 # families + sample probe
uv run python scripts/check_gateway.py --model cfp/zai-org/glm-5.2
uv run python scripts/check_gateway.py --probe kilo-auto
```

Retrieval/ingest tunables live in `config.yaml`. The app still serves retrieval +
citations if the gateway is unreachable (the answer field reports the reason).
`runtime.warmup_on_start: true` pre-loads the embedders at boot so the first
query is instant — set it for demo boxes (already on in `config.yaml`).

### Qdrant: local vs hosted

By default Qdrant runs **local (embedded)** — the index lives under `data/qdrant`
and the API server is the single writer (ingest only through the API). For a
real deployment, point the app at your **hosted Qdrant cluster** instead:

```
# .env  (values are secrets - never commit them)
QDRANT_URL=https://<cluster-id>.<region>.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-api-key>
```

With these set, the app creates the `docs` collection on your cluster with
**server-side late-interaction MaxSim** for the ColPali leg plus dense and sparse
vector fields, and normal concurrent reads/writes apply (the single-writer
restriction is local-mode only). The collection is created automatically at
startup, so a bad URL / API key fails the boot immediately with a clear error.
`DELETE /v1/collection` wipes the cloud collection too (auth-gated if `API_KEY` set).

Sparse vectors are stored as a proper Qdrant named sparse vector (queried with
`using="sparse"`), not in the payload — re-ingest existing documents after
upgrading to populate the sparse leg.

## Answer generation modes (`GENERATION_MODE` in `.env`)

- `text` (default) — the configured LLM (OmniRoute gateway) answers from page text.
  If the gateway is unreachable, the API **automatically falls back to local
  extractive answers** so you always get an answer with citations.
- `extractive` — fully local, no-LLM answers: the most query-relevant sentences
  are pulled straight from the retrieved page text. Zero dependencies, works offline.
- `vision` — stub slot for a future vision-LLM answerer (falls back to extractive).

## Evaluation

```powershell
uv run python -m colpali_rag.eval                      # hybrid mode
uv run python -m colpali_rag.eval --mode colpali       # visual leg alone
uv run python -m colpali_rag.eval --mode dense         # text-dense leg alone
uv run python -m colpali_rag.eval --out evals/results/hybrid.json   # save results JSON
```

Golden queries live in `evals/golden_queries.json` (10 hand-checked queries over
the demo corpus). Each query is retrieved **once** at `--top-k` (default 10) and
metrics are computed from that single ranked list: a recall curve **hit@3 / hit@5 /
hit@10**, **MRR@10**, and **NDCG@5** (binary relevance against the gold `src`+`page`).
Runs are logged to MLflow (`data/mlruns`).

Baseline (committed in `evals/baseline.json`, re-run with the commands above):

| mode   | hit@3 | hit@5 | hit@10 | mrr@10 | ndcg@5 |
|--------|-------|-------|--------|--------|--------|
| hybrid | 1.000 | 1.000 | 1.000  | 1.000  | 1.000  |
| colpali| 1.000 | 1.000 | 1.000  | 1.000  | 1.000  |
| dense  | 1.000 | 1.000 | 1.000  | 0.900  | 0.926  |

Findings: hybrid matches the perfect visual-only retrieval — RRF fusion lifts the
weaker dense-text leg (2/10 golds at rank 1 on chart/figure queries) to rank 0.
The numbers were re-verified on 2026-09-25 after the fusion path became a traced
three-leg RRF (and on the larger corpus: 5+ PDFs, 26 indexed pages) — unchanged.
The demo pack's 8 scripted questions also retrieve their gold pages at rank 0,
including 3 questions answerable only from chart pixels. Scores are
ceiling-limited by the small dev set; grow `golden_queries.json` toward 30–100
queries before relying on the absolute numbers.

## Architecture

```
PDF -> render pages (PyMuPDF) -> ColPali per-patch vectors -> Qdrant "colpali" (MaxSim)
     -> text layer (embedded text / OCR) -> BGE-M3 dense + sparse -> Qdrant
query -> same three legs -> RRF fusion -> top pages -> TextGenerator -> kilo LLM -> answer
```

Notes:
- **Writer model**: local mode runs Qdrant embedded inside the API process (single
  writer — ingest through `POST /v1/ingest` or `ingest.ps1`, never a parallel CLI,
  to avoid file locks). Hosted mode (`QDRANT_URL`/`QDRANT_API_KEY` in `.env`) removes
  the single-writer constraint: any number of app/CLI processes can read/write
  concurrently. An offline CLI exists (`python -m colpali_rag.ingest`) for both modes.
- **Late interaction**: server-side MaxSim is used everywhere the engine supports
  multivector queries; embedded cores fall back to a numpy MaxSim sweep.
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