"""Retrieval pipeline: ColPali MaxSim + BGE-M3 dense + sparse, RRF-fused."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .config import Settings
from .embeddings import ColPaliEmbedder, TextEmbedder
from .qdrant_store import PageResult, QdrantStore

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    query: str
    pages: list[PageResult] = field(default_factory=list)

    def best(self) -> PageResult | None:
        return self.pages[0] if self.pages else None


class Retriever:
    """Encodes the query with both embedders and fuses three retrieval legs."""

    def __init__(self, settings: Settings, store: QdrantStore, colpali: ColPaliEmbedder, text: TextEmbedder) -> None:
        self.settings = settings
        self.store = store
        self.colpali = colpali
        self.text = text

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or self.settings.top_k
        q_colpali: np.ndarray = self.colpali.embed_query(query)
        q_dense, q_sparse = self.text.embed_query(query)
        pages = self.store.hybrid_query(q_colpali, q_dense, q_sparse, k)
        return RetrievalResult(query=query, pages=pages)

    def retrieve_colpali_only(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """ColPali leg alone — handy for eval of the visual retriever."""
        k = top_k or self.settings.top_k
        q_colpali = self.colpali.embed_query(query)
        pages = self.store.query_colpali(q_colpali, k)
        return RetrievalResult(query=query, pages=pages)