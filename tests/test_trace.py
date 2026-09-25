"""Trace serialization tests (fake store + fake embedders, no models)."""
from __future__ import annotations

import numpy as np

from colpali_rag.qdrant_store import PageResult, RetrievalTrace
from colpali_rag.retrieve import Retriever


class FakeColpali:
    is_loaded = True

    def embed_query(self, text):
        return np.zeros((4, 128), dtype=np.float32)


class FakeText:
    is_loaded = True

    def embed_query(self, text):
        return np.zeros(1024, dtype=np.float32), {1: 0.5}


class FakeStore:
    def __init__(self) -> None:
        self.requested_top_k: list[int] = []

    def hybrid_query_traced(self, colpali_vecs, dense_vec, sparse, top_k):
        self.requested_top_k.append(top_k)
        p = lambda s, pg, sc: PageResult(
            score=sc, src=s, page=pg,
            leg={"colpali": sc, "dense": 0.5, "sparse": 0.1},
            legs=["colpali", "dense"],
        )
        colpali = [p("a.pdf", 1, 3.0), p("a.pdf", 2, 2.0)]
        dense = [p("a.pdf", 2, 1.5)]
        sparse = [p("a.pdf", 3, 4.0)]
        fused = [p("a.pdf", 2, 0.02), p("a.pdf", 1, 0.012)]
        trace = RetrievalTrace(
            colpali_ms=1.0, dense_ms=2.0, sparse_ms=3.0, fusion_ms=4.0,
            colpali=colpali, dense=dense, sparse=sparse, fused=fused,
        )
        return fused, trace


def _make_retriever() -> tuple[Retriever, FakeStore]:
    store = FakeStore()
    return Retriever(None, store, FakeColpali(), FakeText()), store


def test_retrieve_trace_shape() -> None:
    r, store = _make_retriever()
    res = r.retrieve("hello world", top_k=5)
    assert store.requested_top_k == [5]
    t = res.trace
    assert set(t) == {"encode_ms", "colpali_ms", "dense_ms", "sparse_ms", "fusion_ms", "total_ms", "legs", "fused"}
    assert t["encode_ms"] >= 0.0
    assert t["total_ms"] == t["encode_ms"] + 1.0 + 2.0 + 3.0 + 4.0
    assert t["legs"]["colpali"][0] == {"src": "a.pdf", "page": 1, "score": 3.0}
    assert t["fused"][0]["legs"] == ["colpali", "dense"]
    assert t["fused"][0]["score"] == 0.02


def test_retrieve_top_k_default_not_required() -> None:
    from types import SimpleNamespace

    r, store = _make_retriever()
    r.settings = SimpleNamespace(top_k=8)
    res = r.retrieve("q")
    assert store.requested_top_k == [8]
    assert res.pages[0].src == "a.pdf"