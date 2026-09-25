"""Qdrant access layer.

One collection "docs" with three vector views per page-point:
  - colpali : multi-vector (per image-patch, 128d) -> late-interaction MaxSim
  - dense   : BGE-M3 dense (1024d)
  - sparse  : BGE-M3 sparse (SPLADE-style, stored as a named sparse vector)

Backend: local (embedded) mode OR a hosted Qdrant cluster. Local mode means a
single writer process (the API server); hosted mode (qdrant_url/qdrant_api_key
from .env) gives normal concurrent access with server-side MaxSim. All
query/upsert code below is a thin, testable wrapper around qdrant-client.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from .utils import point_id

log = logging.getLogger(__name__)


@dataclass
class PageResult:
    score: float
    src: str
    page: int
    text: str = ""
    image: str = ""  # relative path under pages_dir
    payload: dict[str, Any] = field(default_factory=dict)
    # per-leg retrieval scores for the fusion inspector: {"colpali": ..., "dense": ..., "sparse": ...}
    leg: dict[str, float] = field(default_factory=dict)
    legs: list[str] = field(default_factory=list)  # which legs retrieved this page

    @property
    def citation(self) -> str:
        return f"{self.src} (page {self.page})"


@dataclass
class RetrievalTrace:
    """Timings + per-leg ranked lists for the retrieval inspector UI."""

    encode_ms: float = 0.0
    colpali_ms: float = 0.0
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    fusion_ms: float = 0.0
    colpali: list[PageResult] = field(default_factory=list)
    dense: list[PageResult] = field(default_factory=list)
    sparse: list[PageResult] = field(default_factory=list)
    fused: list[PageResult] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return self.encode_ms + self.colpali_ms + self.dense_ms + self.sparse_ms + self.fusion_ms


def rrf_merge(lists: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion over ranked point ids (ints)."""
    scores: dict[int, float] = {}
    for ranked in lists:
        for rank, pid in enumerate(ranked, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return scores


class QdrantStore:
    """Owns the collection schema and performs hybrid queries."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self._client: QdrantClient | None = None
        self._native_late_interaction: bool | None = None

    # -- client -----------------------------------------------------------
    @property
    def is_cloud(self) -> bool:
        """True when backed by a hosted Qdrant cluster."""
        return self.settings.is_cloud

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            if self.is_cloud:
                self._client = QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key or None,
                    timeout=self.settings.qdrant_timeout,
                    check_compatibility=False,  # skip extra version round-trip + warning
                )
                log.info("Qdrant hosted mode: %s (collection %r)", self.settings.qdrant_url, self.settings.collection)
            else:
                path = Path(self.settings.qdrant_path)
                path.mkdir(parents=True, exist_ok=True)
                log.info("Qdrant local mode at %s", path)
                self._client = QdrantClient(path=str(path))
        return self._client

    # -- schema -----------------------------------------------------------
    def ensure_collection(self) -> None:
        coll = self.settings.collection
        if self.client.collection_exists(coll):
            return
        try:
            self.client.create_collection(
                collection_name=coll,
                vectors_config={
                    "colpali": models.VectorParams(
                        size=self.settings.colpali_dim,
                        distance=models.Distance.DOT,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                    ),
                    "dense": models.VectorParams(
                        size=self.settings.dense_dim,
                        distance=models.Distance.DOT,
                    ),
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )
        except Exception as exc:  # pragma: no cover - infra dependent
            where = "hosted" if self.is_cloud else "local"
            raise RuntimeError(
                f"Failed to create collection {coll!r} on {where} Qdrant. "
                "Late-interaction (MaxSim) multivector collections require Qdrant "
                ">= ~1.11 (cloud clusters tick this automatically). Original error: "
                f"{exc}"
            ) from exc
        log.info("Collection %r created (colpali multivector + dense + sparse)", coll)

    def count(self) -> int:
        if not self.client.collection_exists(self.settings.collection):
            return 0
        return self.client.count(self.settings.collection, exact=True).count

    def list_sources(self) -> list[dict[str, Any]]:
        """Distinct document sources (filename) with page counts, sorted by name.

        Payload-only scroll — no vectors loaded, so it is cheap even for big
        collections.
        """
        if not self.client.collection_exists(self.settings.collection):
            return []
        counts: dict[str, int] = {}
        offset: Any = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.settings.collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                src = (pt.payload or {}).get("src")
                if src:
                    counts[src] = counts.get(src, 0) + 1
            if next_offset is None:
                break
            offset = next_offset
        return [{"src": s, "pages": c} for s, c in sorted(counts.items())]

    # -- writes -----------------------------------------------------------
    def upsert_page(
        self,
        src: str,
        page: int,
        colpali_vecs: np.ndarray,
        dense_vec: np.ndarray,
        sparse: dict[int, float],
        payload: dict[str, Any],
    ) -> None:
        pid = point_id(src, page)
        self.client.upsert(
            collection_name=self.settings.collection,
            points=[
                models.PointStruct(
                    id=pid,
                    vector={
                        "colpali": colpali_vecs.astype(np.float32).tolist(),
                        "dense": dense_vec.astype(np.float32).tolist(),
                        # real named sparse vector (queryable with using="sparse"),
                        # NOT a payload field
                        "sparse": models.SparseVector(
                            indices=[int(k) for k in sparse.keys()],
                            values=[float(v) for v in sparse.values()],
                        ),
                    },
                    payload=payload,
                )
            ],
        )

    # -- reads ------------------------------------------------------------
    def _payload_to_result(self, scored) -> PageResult:
        p = scored.payload or {}
        return PageResult(
            score=float(scored.score),
            src=p.get("src", ""),
            page=int(p.get("page", 0)),
            text=p.get("text", ""),
            image=p.get("image", ""),
            payload=p,
        )

    @staticmethod
    def _colpali_query(vecs: np.ndarray) -> list[list[float]]:
        """Late-interaction queries are raw nested lists in current qdrant-client."""
        return vecs.astype(np.float32).tolist()

    def _supports_native_late(self) -> bool:
        """Probe once whether this qdrant core supports multivector (MaxSim) queries."""
        if self._native_late_interaction is not None:
            return self._native_late_interaction
        ok = False
        try:
            coll = self.settings.collection
            if not self.client.collection_exists(coll):
                ok = True  # created fresh by us => fine
            else:
                self.client.query_points(
                    collection_name=coll,
                    query=[[0.0] * self.settings.colpali_dim],
                    using="colpali",
                    limit=1,
                )
                ok = True
        except Exception as exc:  # pragma: no cover - qdrant version dependent
            log.warning("Server-side late interaction unsupported (%s); using numpy MaxSim fallback", exc)
            ok = False
        self._native_late_interaction = ok
        return ok

    def query_colpali(self, query_vecs: np.ndarray, limit: int) -> list[PageResult]:
        """ColPali leg: native MaxSim when possible, numpy fallback otherwise."""
        if not self.client.collection_exists(self.settings.collection):
            return []
        if self._supports_native_late():
            hits = self.client.query_points(
                collection_name=self.settings.collection,
                query=self._colpali_query(query_vecs),
                using="colpali",
                limit=limit,
                with_payload=True,
            ).points
            return [self._payload_to_result(h) for h in hits]
        return self._query_colpali_numpy(query_vecs, limit)

    def query_dense(self, dense_vec: np.ndarray, limit: int) -> list[PageResult]:
        """Dense-text leg alone (used by eval to compare retrieval legs)."""
        if not self.client.collection_exists(self.settings.collection):
            return []
        return self._query_plain(dense_vec.astype(np.float32), "dense", limit)

    def _query_colpali_numpy(self, query_vecs: np.ndarray, limit: int) -> list[PageResult]:
        """MaxSim over the whole collection (fallback for embedded dev cores)."""
        q = query_vecs.astype(np.float32)  # (n_tokens, D)
        results: list[PageResult] = []
        scroll = True
        offset: Any = None
        while scroll:
            page, scroll = self.client.scroll(
                collection_name=self.settings.collection,
                limit=100,
                with_vectors=True,
                with_payload=True,
                offset=offset,
            )
            offset = page[-1].id if page else None
            for p in page:
                vecs = np.asarray(p.vector.get("colpali", []), dtype=np.float32)
                if vecs.ndim != 2 or vecs.shape[0] == 0:
                    continue
                dot = q @ vecs.T  # (n_tokens, n_patches)
                score = float(dot.max(axis=1).sum())
                pr = self._payload_to_result(p)
                pr.score = score
                results.append(pr)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def hybrid_query(self, colpali_vecs: np.ndarray, dense_vec: np.ndarray, sparse: dict[int, float], top_k: int) -> list[PageResult]:
        """Fuse three legs with RRF and return the top-k pages."""
        pages, _trace = self.hybrid_query_traced(colpali_vecs, dense_vec, sparse, top_k)
        return pages

    def hybrid_query_traced(
        self, colpali_vecs: np.ndarray, dense_vec: np.ndarray, sparse: dict[int, float], top_k: int
    ) -> tuple[list[PageResult], RetrievalTrace]:
        """Run the three legs separately, fuse with RRF, and return the fusion
        trace (timings + per-leg rankings) alongside the top-k pages.

        Fusion keys are (src, page) so a point retrieved by multiple legs
        contributes to one fused result (the old id()-keyed merge could emit
        duplicate PageResults for the same point).
        """
        from time import perf_counter

        trace = RetrievalTrace()
        coll = self.settings.collection
        if not self.client.collection_exists(coll):
            return [], trace

        t = perf_counter()
        colpali_hits = self.query_colpali(colpali_vecs, self.settings.colpali_prefetch)
        trace.colpali_ms = (perf_counter() - t) * 1000.0
        trace.colpali = colpali_hits

        t = perf_counter()
        dense_hits = self.query_dense(dense_vec, self.settings.dense_prefetch)
        trace.dense_ms = (perf_counter() - t) * 1000.0
        trace.dense = dense_hits

        t = perf_counter()
        sparse_hits = self._query_plain(
            models.SparseVector(indices=list(sparse.keys()), values=list(sparse.values())),
            "sparse",
            self.settings.sparse_prefetch,
        )
        trace.sparse_ms = (perf_counter() - t) * 1000.0
        trace.sparse = sparse_hits

        t = perf_counter()
        legs = {"colpali": colpali_hits, "dense": dense_hits, "sparse": sparse_hits}
        fused_scores: dict[tuple[str, int], float] = {}
        for name, hits in legs.items():
            for rank, h in enumerate(hits, start=1):
                key = (h.src, h.page)
                fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (self.settings.rrf_k + rank)
        # one PageResult per point; prefer the colpali hit (best visual evidence)
        best: dict[tuple[str, int], PageResult] = {}
        for name in ("colpali", "dense", "sparse"):
            for h in legs[name]:
                best.setdefault((h.src, h.page), h)
        for h in legs["colpali"]:
            best[(h.src, h.page)] = h
        ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results: list[PageResult] = []
        for key, score in ordered:
            h = best[key]
            per_leg = {name: float(next((x.score for x in hits if (x.src, x.page) == key), 0.0)) for name, hits in legs.items()}
            found = [name for name, hits in legs.items() if any((x.src, x.page) == key for x in hits)]
            results.append(
                PageResult(
                    score=score,
                    src=h.src,
                    page=h.page,
                    text=h.text,
                    image=h.image,
                    payload=h.payload,
                    leg=per_leg,
                    legs=found,
                )
            )
        trace.fusion_ms = (perf_counter() - t) * 1000.0
        trace.fused = results
        return results, trace

    def _query_plain(self, query: Any, using: str, limit: int) -> list[PageResult]:
        hits = self.client.query_points(
            collection_name=self.settings.collection,
            query=query,
            using=using,
            limit=limit,
            with_payload=True,
        ).points
        return [self._payload_to_result(h) for h in hits]

    def delete_point(self, src: str, page: int) -> None:
        self.client.delete(collection_name=self.settings.collection, points_selector=[point_id(src, page)])

    def delete_collection(self) -> None:
        if self.client.collection_exists(self.settings.collection):
            self.client.delete_collection(self.settings.collection)