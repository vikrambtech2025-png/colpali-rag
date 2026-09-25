"""Embedder lazy-loading regression tests (no real models, no download).

Covers the atomic-load invariant: a mid-load failure (e.g. flaky HF network
while downloading the processor) must leave the embedder clean so the next
call retries the whole load instead of poisoning it (model set, processor None
-> every later query 500s).
"""
from __future__ import annotations

import pytest

from colpali_rag.config import Settings
from colpali_rag.embeddings import ColPaliEmbedder


def test_colpali_load_atomic_when_processor_fails_then_retries(monkeypatch) -> None:
    calls = {"n": 0}

    class FakeModel:
        eval_called = False

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            FakeModel.eval_called = True

        @property
        def device(self):
            return "cpu"

    class FlakyProc:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("network flake: server disconnected")
            return object()

    emb = ColPaliEmbedder(
        Settings(colpali_model="fake/model", text_embed_model="x", text_sparse_model="y")
    )
    monkeypatch.setattr(
        ColPaliEmbedder,
        "_model_classes",
        staticmethod(lambda: (FakeModel, FlakyProc)),
    )

    with pytest.raises(RuntimeError, match="network flake"):
        emb.load()

    # Not poisoned: attrs are reset, so the next call retries the whole load.
    assert emb.is_loaded is False
    assert emb._model is None
    assert emb._processor is None

    emb.load()  # clean retry succeeds from scratch
    assert emb.is_loaded is True
    assert emb._model is not None
    assert emb._processor is not None
    assert FakeModel.eval_called is True