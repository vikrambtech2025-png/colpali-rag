"""Evaluation harness: golden queries -> retrieval metrics -> optional MLflow run.

Metrics (computed from a single ranked list per query, no re-querying per k):
  - hit@3 / hit@5 / hit@10  (recall curve, binary relevance against the gold page)
  - mrr@10                  (reciprocal rank of the gold page)
  - ndcg@5                  (binary relevance, discounted)

Modes: hybrid (ColPali + dense + sparse RRF), colpali, dense — so the recall
curve can be compared leg by leg. Run baseline with:

    uv run python -m colpali_rag.eval --mode hybrid --out evals/baseline.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from .config import Settings, configure_hf_env, get_settings, mlflow_tracking_uri
from .embeddings import ColPaliEmbedder, TextEmbedder
from .qdrant_store import QdrantStore
from .retrieve import Retriever

DEFAULT_GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden_queries.json"
RECALL_K = (3, 5, 10)


def ndcg_at_k(relevant: list[int], k: int = 5) -> float:
    """relevant: binary relevance labels in ranked order.

    IDCG is computed from the *ideal* ordering of the same labels (relevant
    items first), not from assuming every rank is relevant — otherwise a
    perfect ranking scores < 1.0 on sparse relevance.
    """
    if not relevant:
        return 0.0
    dcg = sum((1.0 / math.log2(i + 2)) for i, r in enumerate(relevant[:k]) if r)
    ideal = sorted(relevant, reverse=True)[:k]
    idcg = sum((1.0 / math.log2(i + 2)) for i, r in enumerate(ideal) if r)
    return dcg / idcg if idcg else 0.0


def _rank_of(pages, expected_src: str, expected_page: int) -> int | None:
    return next(
        (i for i, p in enumerate(pages) if p.src == expected_src and p.page == int(expected_page)),
        None,
    )


def evaluate(retriever: Retriever, golden: list[dict[str, Any]], top_k: int = 10, mode: str = "hybrid") -> dict[str, Any]:
    """golden items: {"query": str, "expected_src": str, "expected_page": int}.

    Retrieves each query at most top_k and reports the recall curve, MRR and
    NDCG from that same ranked list.
    """
    if mode == "colpali":
        retrieve = retriever.retrieve_colpali_only
    elif mode == "dense":
        retrieve = retriever.retrieve_dense_only
    else:
        retrieve = retriever.retrieve

    max_k = min(top_k, max(RECALL_K))
    hits = {k: 0 for k in RECALL_K}
    rr_sum = 0.0
    ndcg5_sum = 0.0
    per_query: list[dict[str, Any]] = []
    for item in golden:
        pages = retrieve(item["query"], top_k=max_k).pages
        rank = _rank_of(pages, item["expected_src"], item["expected_page"])
        for k in RECALL_K:
            if rank is not None and rank < k:
                hits[k] += 1
        relevant = [1 if i == rank else 0 for i in range(len(pages))] if rank is not None else []
        rr_sum += 1.0 / (rank + 1) if rank is not None else 0.0
        ndcg5_sum += ndcg_at_k(relevant, 5)
        per_query.append({
            "query": item["query"],
            "expected": f"{item['expected_src']} p{item['expected_page']}",
            "rank": rank,
            "hit@5": bool(rank is not None and rank < 5),
        })

    n = float(len(golden))
    return {
        "n": n,
        "mode": mode,
        "hit@3": hits[3] / n,
        "hit@5": hits[5] / n,
        "hit@10": hits[10] / n,
        "mrr@10": rr_sum / n,
        "ndcg@5": ndcg5_sum / n,
        "per_query": per_query,
    }


def _print_summary(metrics: dict[str, Any]) -> None:
    print(f"mode     : {metrics['mode']}  (n={int(metrics['n'])})")
    print(f"hit@3    : {metrics['hit@3']:.3f}")
    print(f"hit@5    : {metrics['hit@5']:.3f}")
    print(f"hit@10   : {metrics['hit@10']:.3f}")
    print(f"mrr@10   : {metrics['mrr@10']:.3f}")
    print(f"ndcg@5   : {metrics['ndcg@5']:.3f}")
    print("per-query:")
    for pq in metrics["per_query"]:
        tag = "HIT" if pq["hit@5"] else "miss"
        print(f"  [{tag:4}] rank={pq['rank']}  {pq['expected']}  {pq['query']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="colpali-eval", description="Run retrieval evaluation")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="golden queries JSON")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=["hybrid", "colpali", "dense"], default="hybrid")
    parser.add_argument("--out", help="write results JSON to this path (e.g. evals/results/hybrid.json)")
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
    _print_summary(metrics)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")

    if not args.no_mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_tracking_uri(settings))
            with mlflow.start_run(run_name=f"eval-{args.mode}"):
                mlflow.log_params({"mode": args.mode, "golden": str(args.golden), "n": len(golden)})
                # MLflow forbids '@' in metric names; hit@5 -> hit_5, ndcg@5 -> ndcg_5
                safe = {k.replace("@", "_"): v for k, v in metrics.items() if isinstance(v, (int, float))}
                mlflow.log_metrics(safe)
        except Exception as exc:  # pragma: no cover
            print(f"mlflow skipped: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())