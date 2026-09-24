"""Evaluation harness: golden queries -> retrieval metrics -> MLflow run.

Metrics: hit@5, hit@10, MRR, NDCG@5 (binary relevance against expected page).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from .config import Settings, configure_hf_env
from .embeddings import ColPaliEmbedder, TextEmbedder
from .qdrant_store import QdrantStore
from .retrieve import Retriever

DEFAULT_GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden_queries.json"


def ndcg_at_k(relevant: list[int], k: int = 5) -> float:
    if not relevant:
        return 0.0
    dcg = sum((1.0 / math.log2(i + 2)) for i, r in enumerate(relevant[:k]) if r)
    idcg = sum((1.0 / math.log2(i + 2)) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def evaluate(retriever: Retriever, golden: list[dict[str, Any]], top_k: int = 10, mode: str = "hybrid") -> dict[str, float]:
    """golden items: {"query": str, "expected_src": str, "expected_page": int}"""
    hits5 = 0
    rr_sum = 0.0
    ndcg5_sum = 0.0
    n = len(golden)
    per_query: list[dict[str, Any]] = []
    for item in golden:
        q = item["query"]
        result = (
            retriever.retrieve(q, top_k) if mode == "hybrid" else retriever.retrieve_colpali_only(q, top_k)
        )
        pages = result.pages
        relevant = [
            1
            for p in pages
            if p.src == item["expected_src"] and p.page == int(item["expected_page"])
        ]
        hit = any(relevant)
        hits5 += hit
        rank = next(
            (i for i, p in enumerate(pages) if p.src == item["expected_src"] and p.page == int(item["expected_page"])),
            None,
        )
        rr = 1.0 / (rank + 1) if rank is not None and rank < top_k else 0.0
        rr_sum += rr
        ndcg5_sum += ndcg_at_k(relevant, 5)
        per_query.append({"query": q, "hit": bool(relevant), "recip_rank": rr})

    return {
        "n": float(n),
        "hit@5": hits5 / n,
        "mrr": rr_sum / n,
        "ndcg@5": ndcg5_sum / n,
        "mode": mode,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="colpali-eval", description="Run retrieval evaluation")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="golden queries JSON")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=["hybrid", "colpali"], default="hybrid")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings()
    settings.ensure_dirs()
    configure_hf_env(settings)
    store = QdrantStore(settings)
    colpali = ColPaliEmbedder(settings)
    text = TextEmbedder(settings)
    retriever = Retriever(settings, store, colpali, text)

    with Path(args.golden).open("r", encoding="utf-8") as fh:
        golden = json.load(fh)

    metrics = evaluate(retriever, golden, top_k=args.top_k, mode=args.mode)
    print(json.dumps(metrics, indent=2))

    if not args.no_mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri("file:" + str(settings.base_dir / "data" / "mlruns"))
            with mlflow.start_run(run_name=f"eval-{args.mode}"):
                mlflow.log_params({"mode": args.mode, "golden": str(args.golden), "n": len(golden)})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        except Exception as exc:  # pragma: no cover
            print(f"mlflow skipped: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())