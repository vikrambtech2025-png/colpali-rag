"""Embedding engines.

- ColPaliEmbedder: Vision-Language retriever producing per-patch 128-d vectors,
  loaded transformers-native (ColQwen2ForRetrieval / ColPaliForRetrieval), used
  with Qdrant late-interaction MaxSim.
- TextEmbedder: BGE-M3 dense + sparse (SPLADE-style) via fastembed.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Iterable

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ColPali / ColQwen2 (visual, late interaction)
# ---------------------------------------------------------------------------


class ColPaliEmbedder:
    """Lazy-loads the visual retriever. Produces per-page/per-query token vectors."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any = None
        self._processor: Any = None
        self._load_lock = threading.Lock()

    def _model_classes(self) -> tuple[type, type]:
        """Pick transformers-native classes by model family.

        Imported inside a lock (see load()) because transformers lazy-loads
        these modules; importing them concurrently from two threads can fail
        intermittently with 'cannot import name'.
        """
        low = self.settings.colpali_model.lower()
        if "qwen" in low:
            from transformers import ColQwen2ForRetrieval, ColQwen2Processor

            return ColQwen2ForRetrieval, ColQwen2Processor
        from transformers import ColPaliForRetrieval, ColPaliProcessor

        return ColPaliForRetrieval, ColPaliProcessor

    # -- lifecycle --------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:  # serialize first load across ingest/query threads
            if self._model is not None:
                return
            import torch

            model_cls, proc_cls = self._model_classes()
            model_id = self.settings.colpali_model
            log.info(
                "Loading %s from %s (device=%s, dtype=float16)",
                model_cls.__name__,
                model_id,
                self.settings.resolved_device,
            )
            self._model = model_cls.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=self.settings.resolved_device,
                trust_remote_code=True,
            )
            self._model.eval()
            self._processor = proc_cls.from_pretrained(model_id, trust_remote_code=True)
            log.info("ColPali embedder ready: %s", model_id)

    def unload(self) -> None:
        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _extract(outputs: Any) -> list[np.ndarray]:
        """Normalize model output to a list of (n_tokens, D) fp32 arrays."""
        import torch

        if isinstance(outputs, (list, tuple)):
            items = list(outputs)
        elif hasattr(outputs, "embeddings"):
            items = list(outputs.embeddings)
        else:
            raise TypeError(f"unexpected model output: {type(outputs)}")
        out = []
        for t in items:
            v = t.detach().float().cpu().numpy().astype(np.float32)
            if v.ndim == 1:
                v = v.reshape(1, -1)
            out.append(v)
        return out

    # -- inference ---------------------------------------------------------
    def embed_images(self, images: Iterable[Any], batch_size: int | None = None) -> list[np.ndarray]:
        """images: PIL Images. Returns list of (n_patches, D) float arrays (fp32)."""
        self.load()
        import torch

        bs = batch_size or self.settings.embed_batch_size
        out: list[np.ndarray] = []
        imgs = list(images)
        for i in range(0, len(imgs), bs):
            chunk = imgs[i : i + bs]
            batches = self._processor.process_images(chunk).to(self._model.device)
            with torch.inference_mode():
                outputs = self._model(**batches)
            out.extend(self._extract(outputs))
        return out

    def embed_query(self, text: str) -> np.ndarray:
        """Returns (n_tokens, D) float array for late-interaction scoring."""
        self.load()
        import torch

        queries = self._processor.process_queries([text]).to(self._model.device)
        with torch.inference_mode():
            outputs = self._model(**queries)
        return self._extract(outputs)[0]


# ---------------------------------------------------------------------------
# Text: dense + sparse
# ---------------------------------------------------------------------------


class TextEmbedder:
    """Dense (BGE) + sparse (BM25) legs via fastembed, both CPU-friendly.

    fastembed 0.8 dropped BGE-M3, so we use two models:
      - dense : BAAI/bge-large-en-v1.5 (1024-d)
      - sparse: Qdrant/bm25 (fastembed BM25, no GPU, downloads small vocab/stats)
    `embed_query` / `embed_documents` both return (dense_1024, sparse_dict) pairs.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._dense: Any = None
        self._sparse: Any = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        if self._dense is not None:
            return
        with self._load_lock:  # serialize first load across threads
            if self._dense is not None:
                return
            from fastembed import SparseTextEmbedding, TextEmbedding

            log.info("Loading dense %s", self.settings.text_embed_model)
            self._dense = TextEmbedding(
                model_name=self.settings.text_embed_model,
                cache_dir=str(self.settings.hf_cache),
            )
            log.info("Loading sparse %s", self.settings.text_sparse_model)
            self._sparse = SparseTextEmbedding(
                model_name=self.settings.text_sparse_model,
                cache_dir=str(self.settings.hf_cache),
            )
            log.info("Text embedder ready")

    @property
    def is_loaded(self) -> bool:
        return self._dense is not None

    def embed_texts(self, texts: list[str], is_query: bool = False) -> list[tuple[np.ndarray, dict[int, float]]]:
        self.load()
        dense_fn = self._dense.query_embed if is_query else self._dense.embed
        sparse_fn = self._sparse.query_embed if is_query else self._sparse.embed
        dense_out = list(dense_fn(texts))
        sparse_out = list(sparse_fn(texts))
        out: list[tuple[np.ndarray, dict[int, float]]] = []
        for d, s in zip(dense_out, sparse_out):
            dense = np.asarray(d, dtype=np.float32).reshape(-1)
            if hasattr(s, "indices"):
                sparse = {int(i): float(v) for i, v in zip(s.indices, s.values)}
            else:
                sparse = {int(k): float(v) for k, v in dict(s).items()}
            out.append((dense, sparse))
        return out

    def embed_query(self, text: str) -> tuple[np.ndarray, dict[int, float]]:
        return self.embed_texts([text], is_query=True)[0]

    def embed_documents(self, texts: list[str]) -> list[tuple[np.ndarray, dict[int, float]]]:
        return self.embed_texts(texts, is_query=False)