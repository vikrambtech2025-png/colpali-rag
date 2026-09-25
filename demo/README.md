# College Showcase - Demo Pack

Everything needed to run a reliable, impressive live demo of the ColPali RAG app:
curated PDFs, a scripted question set, failover notes, and talking points.

## One-time setup (5 minutes, do it the night before)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare-demo.ps1
```

What it does, safely:
1. Stops the app (offline ingest requires the server down - single-writer rule).
2. Generates the demo PDFs into `demo\pdfs\` (matplotlib charts + pymupdf text).
3. Ingests them into the index.
4. Starts the app with `warmup_on_start: true` - the embedders load at boot, so
   the first question the audience sees answers instantly.

Then open http://127.0.0.1:8000 and click one of the quick-start chips to smoke-test.

## The scripted demo (about 4 minutes)

Each question below was verified to retrieve the exact page at rank 0. The three
marked **[chart]** questions are the showpieces: a text-only RAG system cannot
answer them.

| # | Question | Doc | Gold page | Why it impresses |
|---|----------|-----|-----------|------------------|
| 1 | What is the revisit requirement for the KESTREL constellation? | NanoSat-Constellation-Design-Spec.pdf | 1 | Block diagram page with a caption table |
| 2 | What is the orbit-average power of the edge AI processor? | NanoSat-Constellation-Design-Spec.pdf | 2 | Power-budget table page |
| 3 **[chart]** | What is the S-band link margin at 540 km altitude? | NanoSat-Constellation-Design-Spec.pdf | 3 | The answer lives inside a matplotlib line chart |
| 4 | Which risks exceed the red threshold? | NanoSat-Constellation-Design-Spec.pdf | 4 | Color-coded risk matrix page |
| 5 **[chart]** | What was month-6 retention for the April onboarding cohort? | SaaS-Cohort-Retention-Report.pdf | 1 | Multi-line retention chart |
| 6 **[chart]** | What is the largest drop in the activation funnel? | SaaS-Cohort-Retention-Report.pdf | 2 | Funnel bar chart |
| 7 | Which cohort retained best at month six? | SaaS-Cohort-Retention-Report.pdf | 3 | Numeric cohort table |
| 8 | What is the recommended next step for activation? | SaaS-Cohort-Retention-Report.pdf | 4 | Findings + recommendation page |

### Suggested flow

1. Land on the page - the hero shows the doc/page count ticking up as documents
   are indexed. Say: "Every page of these PDFs was rendered to an image and
   embedded with ColQwen2, plus a text layer with BGE-M3 and BM25."
2. Ask Q1 or Q2 first (easy win), then Q3 **[chart]**. While it runs, say:
   "Notice it found the page with the chart, not just text that mentions a
   chart. That is late-interaction visual retrieval - ColPali compares the
   question against the rendered page image."
3. Click the **retrieval internals** panel under the answer. Explain: three
   independent rankings (ColPali visual, BGE dense, BM25 sparse) are merged with
   Reciprocal Rank Fusion; the colored badges show which legs found each page.
4. Ask Q5 or Q6 for a second chart win, then Q8 to show a normal "answer page"
   query. Mention citations: every answer carries `source (page N)`.
5. Close with the numbers: eval dev set MRR@10 = 1.0, NDCG@5 = 1.0 (see
   `evals/baseline.json`), everything on a 6 GB laptop GPU with free-tier LLM
   answers through a local gateway.

## Failover plan (never let the demo die)

- **No internet / gateway down:** the app retries, then falls back to
  **extractive answers** (verbatim quotes from the retrieved pages). The UI
  shows the fallback note. Keep talking; citations still work.
- **Demo-safe defaults:** run with `.env` -> `GENERATION_MODE=extractive` on
  the day if you want zero network dependence, or keep `text` to show the LLM
  answers when the pool is up. Both are supported; the app tells you which it used.
- **Pre-warm:** `warmup_on_start: true` in `config.yaml` (already set). Do not
  restart the app with the audience waiting.
- **Keep the index trained:** run `prepare-demo.ps1` once before the event and
  do not wipe `/v1/collection` during the demo.

## Rebuild the PDFs

`demo\make_demo_pdfs.py` regenerates the corpus from scratch (it is ~100 lines
of matplotlib + pymupdf). Edit the data/labels and re-run
`prepare-demo.ps1` to re-ingest. The PDFs themselves are gitignored - the
generator is the source of truth, so nothing binary sits in the repo.