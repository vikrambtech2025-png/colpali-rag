"""Hosted-Qdrant (cloud) mode: client selection and named sparse vectors."""
from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient

from colpali_rag.config import Settings
from colpali_rag.qdrant_store import QdrantStore


def test_cloud_mode_selected_when_url_set(tmp_path) -> None:
    s = Settings(
        qdrant_url="https://stub.cloud.qdrant.io:6333",
        qdrant_api_key="secret",
        qdrant_path=tmp_path / "qdrant",
    )
    store = QdrantStore(s)
    assert store.is_cloud
    assert isinstance(store.client, QdrantClient)  # no network at construction


def test_local_mode_when_no_url(tmp_path) -> None:
    s = Settings(qdrant_path=tmp_path / "qdrant", qdrant_url="")
    store = QdrantStore(s)
    assert not store.is_cloud


class _FakeClient:
    def __init__(self) -> None:
        self.upserted: list = []

    def upsert(self, collection_name=None, points=None):
        self.upserted = list(points or [])
        return None


def test_sparse_stored_as_named_vector_not_payload(tmp_path) -> None:
    """Late-interaction collections need a real named sparse vector for
    using='sparse' queries; payload storage silently returned nothing."""
    s = Settings(qdrant_path=tmp_path / "qdrant", qdrant_url="")
    store = QdrantStore(s)
    fake = _FakeClient()
    store._client = fake  # bypass the real client for this pure-shape test
    store.upsert_page(
        src="a.pdf",
        page=1,
        colpali_vecs=np.zeros((4, 128), dtype=np.float32),
        dense_vec=np.zeros(1024, dtype=np.float32),
        sparse={3: 1.5, 9: 0.25},
        payload={"src": "a.pdf", "page": 1, "text": "hi", "image": ""},
    )
    pt = fake.upserted[0]
    v = pt.vector
    assert set(v) == {"colpali", "dense", "sparse"}
    sv = v["sparse"]
    assert list(sv.indices) == [3, 9]
    assert list(sv.values) == [1.5, 0.25]
    assert "sparse" not in pt.payload