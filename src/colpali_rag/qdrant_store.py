"""Qdrant access layer.

One collection "docs" with three vector views per page-point:
  - colpali : multi-vector (per image-patch, 128d) -> late-interaction MaxSim
  - dense   : BGE-M3 dense (1024d)
  - sparse  : BGE-M3 sparse (SPLADE-style)

Qdrant runs in local (embedded) mode => a single writer process (the API server).
All query/upsert code below is a thin, testable wrapper around qdrant-client.
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

    @property
    def citation(self) -> str:
        return f"{self.src} (page {self.page})"


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
    def client(self) -> QdrantClient:
        if self._client is None:
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
        log.info("Collection %r created (colpali multivector + dense + sparse)", coll)

    def count(self) -> int:
        if not self.client.collection_exists(self.settings.collection):
            return 0
        return self.client.count(self.settings.collection, exact=True).count

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
        sparse = {int(k): float(v) for k, v in sparse.items()}
        self.client.upsert(
            collection_name=self.settings.collection,
            points=[
                models.PointStruct(
                    id=pid,
                    vector={
                        "colpali": colpali_vecs.astype(np.float32).tolist(),
                        "dense": dense_vec.astype(np.float32).tolist(),
                    },
                    payload={**payload, "sparse": sparse},
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
        """Fuse three legs with RRF. Uses Qdrant server fusion when supported."""
        coll = self.settings.collection
        if not self.client.collection_exists(coll):
            return []
        prefetch = [
            models.Prefetch(query=self._colpali_query(colpali_vecs), using="colpali", limit=self.settings.colpali_prefetch),
            models.Prefetch(query=dense_vec.astype(np.float32).tolist(), using="dense", limit=self.settings.dense_prefetch),
            models.Prefetch(
                query=models.SparseVector(indices=list(sparse.keys()), values=list(sparse.values())),
                using="sparse",
                limit=self.settings.sparse_prefetch,
            ),
        ]
        try:
            scored = self.client.query_points(
                collection_name=coll,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            ).points
            return [self._payload_to_result(h) for h in scored]
        except Exception as exc:  # pragma: no cover
            log.warning("Server fusion failed (%s); falling back to python RRF", exc)
            return self.hybrid_query_python(colpali_vecs, dense_vec, sparse, top_k)

    def hybrid_query_python(
        self, colpali_vecs: np.ndarray, dense_vec: np.ndarray, sparse: dict[int, float], top_k: int
    ) -> list[PageResult]:
        """Python RRF fallback: three independent queries, merged locally."""
        colpali_hits = self.query_colpali(colpali_vecs, self.settings.colpali_prefetch)
        dense_hits = [
            h for h in self._query_plain(dense_vec, "dense", self.settings.dense_prefetch)
        ]
        sparse_hits = [
            h for h in self._query_plain(models.SparseVector(indices=list(sparse.keys()), values=list(sparse.values())), "sparse", self.settings.sparse_prefetch)
        ]
        merged = rrf_merge(
            [[id(h) for h in colpali_hits], [id(h) for h in dense_hits], [id(h) for h in sparse_hits]],
            k=self.settings.rrf_k,
        )
        by_id = {id(h): h for h in [*colpali_hits, *dense_hits, *sparse_hits]}
        ordered = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results: list[PageResult] = []
        for pid, score in ordered:
            h = by_id[pid]
            h.score = score
            results.append(h)
        return results

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