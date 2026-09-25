"""Retrieval pipeline: ColPali MaxSim + BGE-M3 dense + sparse, RRF-fused."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from .config import Settings
from .embeddings import ColPaliEmbedder, TextEmbedder
from .qdrant_store import PageResult, QdrantStore

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    query: str
    pages: list[PageResult] = field(default_factory=list)
    trace: dict[str, Any] | None = None  # serializable fusion trace (inspector UI)

    def best(self) -> PageResult | None:
        return self.pages[0] if self.pages else None


def _slim(hits: list[PageResult], limit: int = 8) -> list[dict[str, Any]]:
    """Compact per-leg ranking for the trace payload."""
    return [{"src": h.src, "page": h.page, "score": round(h.score, 4)} for h in hits[:limit]]


def _trace_dict(trace) -> dict[str, Any]:
    """Serialize a RetrievalTrace for the API/UI."""
    return {
        "encode_ms": round(trace.encode_ms, 1),
        "colpali_ms": round(trace.colpali_ms, 1),
        "dense_ms": round(trace.dense_ms, 1),
        "sparse_ms": round(trace.sparse_ms, 1),
        "fusion_ms": round(trace.fusion_ms, 1),
        "total_ms": round(trace.total_ms, 1),
        "legs": {
            "colpali": _slim(trace.colpali),
            "dense": _slim(trace.dense),
            "sparse": _slim(trace.sparse),
        },
        "fused": [
            {
                "src": h.src,
                "page": h.page,
                "score": round(h.score, 5),
                "legs": list(h.legs),
            }
            for h in trace.fused
        ],
    }


class Retriever:
    """Encodes the query with both embedders and fuses three retrieval legs."""

    def __init__(self, settings: Settings, store: QdrantStore, colpali: ColPaliEmbedder, text: TextEmbedder) -> None:
        self.settings = settings
        self.store = store
        self.colpali = colpali
        self.text = text

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or self.settings.top_k
        t0 = perf_counter()
        q_colpali: np.ndarray = self.colpali.embed_query(query)
        q_dense, q_sparse = self.text.embed_query(query)
        encode_ms = (perf_counter() - t0) * 1000.0
        pages, trace = self.store.hybrid_query_traced(q_colpali, q_dense, q_sparse, k)
        trace.encode_ms = encode_ms
        return RetrievalResult(query=query, pages=pages, trace=_trace_dict(trace))

    def retrieve_colpali_only(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """ColPali leg alone — handy for eval of the visual retriever."""
        k = top_k or self.settings.top_k
        q_colpali = self.colpali.embed_query(query)
        pages = self.store.query_colpali(q_colpali, k)
        return RetrievalResult(query=query, pages=pages)

    def retrieve_dense_only(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Dense-text leg alone — handy for eval of the text retriever."""
        k = top_k or self.settings.top_k
        q_dense, _ = self.text.embed_query(query)
        pages = self.store.query_dense(q_dense, k)
        return RetrievalResult(query=query, pages=pages)