"""Regression tests for the eval metrics (fake retriever, no models)."""
from __future__ import annotations

import math

from colpali_rag.eval import evaluate, ndcg_at_k
from colpali_rag.qdrant_store import PageResult


def _page(src: str, page: int) -> PageResult:
    return PageResult(score=1.0, src=src, page=page)


class FakeRetriever:
    """Scripted ranked results per query (mapping query -> list of (src, page))."""

    def __init__(self, orders: dict[str, list[tuple[str, int]]]) -> None:
        self.orders = orders

    def _pages(self, query: str, top_k: int) -> list[PageResult]:
        return [_page(s, p) for s, p in self.orders[query][:top_k]]

    def retrieve(self, query: str, top_k: int | None = None):
        return type("R", (), {"pages": self._pages(query, top_k or 10)})


GOLDEN = [
    {"query": "a", "expected_src": "d.pdf", "expected_page": 1},  # rank 0 -> hit@3/5/10, mrr 1.0
    {"query": "b", "expected_src": "d.pdf", "expected_page": 3},  # rank 4 -> hit@5/10, mrr 0.2
    {"query": "c", "expected_src": "d.pdf", "expected_page": 9},  # rank 9 -> hit@10 only, mrr 0.1
    {"query": "d", "expected_src": "d.pdf", "expected_page": 1},  # not retrieved
]


def test_ndcg_at_k_basic() -> None:
    assert ndcg_at_k([], 5) == 0.0
    assert abs(ndcg_at_k([1, 1, 0], 5) - 1.0) < 1e-9
    single = ndcg_at_k([1, 0, 0, 0], 5)
    assert single == 1.0  # one relevant item ranked first is a perfect ranking
    second = ndcg_at_k([0, 1, 0, 0], 5)
    assert abs(second - (1.0 / math.log2(3))) < 1e-9  # relevant at rank 2 is discounted


def test_evaluate_recall_curve_and_mrr() -> None:
    orders = {
        "a": [("d.pdf", 1), ("d.pdf", 2), ("d.pdf", 3)],
        "b": [("d.pdf", 2), ("d.pdf", 4), ("d.pdf", 5), ("d.pdf", 6), ("d.pdf", 3)],
        "c": [("d.pdf", 2)] + [("d.pdf", i) for i in range(3, 9)] + [("d.pdf", 9)],  # gold at rank 7
        "d": [("d.pdf", 7)],
    }
    m = evaluate(FakeRetriever(orders), GOLDEN, top_k=10, mode="hybrid")
    assert m["n"] == 4.0
    assert m["hit@3"] == 0.25   # only query a within first 3
    assert m["hit@5"] == 0.5    # a + b
    assert m["hit@10"] == 0.75  # a + b + c
    assert abs(m["mrr@10"] - (1.0 + 0.2 + 0.125 + 0.0) / 4.0) < 1e-9  # c ranks 7th -> 1/8
    expected_ndcg = (ndcg_at_k([1], 5) + ndcg_at_k([0, 0, 0, 0, 1], 5)) / 4.0
    assert abs(m["ndcg@5"] - expected_ndcg) < 1e-9


def test_evaluate_passes_top_k_through() -> None:
    # gold (d.pdf, 1) sits at index 9 of the scripted order
    orders = {"a": [("d.pdf", 2)] + [("d.pdf", i) for i in range(3, 11)] + [("d.pdf", 1)]}
    big = evaluate(FakeRetriever(orders), [GOLDEN[0]], top_k=10, mode="hybrid")
    assert big["hit@10"] == 1.0
    assert big["hit@5"] == 0.0
    assert abs(big["mrr@10"] - 0.1) < 1e-9

    small = evaluate(FakeRetriever(orders), [GOLDEN[0]], top_k=5, mode="hybrid")
    assert small["hit@3"] == 0.0 and small["hit@5"] == 0.0 and small["hit@10"] == 0.0