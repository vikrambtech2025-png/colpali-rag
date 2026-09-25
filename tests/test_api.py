"""API tests that avoid model loading: monkeypatched embedders + a temp qdrant path."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from colpali_rag import api as api_mod
from colpali_rag.config import Settings


class FakeColpali:
    is_loaded = True

    def embed_query(self, text):
        return np.zeros((4, 128), dtype=np.float32)

    def embed_images(self, images, batch_size=None):
        return [np.zeros((16, 128), dtype=np.float32) for _ in images]


class FakeText:
    is_loaded = True

    def embed_query(self, text):
        return np.zeros(1024, dtype=np.float32), {1: 0.5}

    def embed_documents(self, texts):
        return [(np.zeros(1024, dtype=np.float32), {1: 0.5}) for _ in texts]


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        qdrant_path=tmp_path / "qdrant",
        qdrant_url="",  # hermetic: never inherit the hosted-cluster .env creds
        pages_dir=tmp_path / "pages",
        hf_cache=tmp_path / "hf",
        manifests_dir=tmp_path / "manifests",
        corpus_dir=tmp_path / "corpus",
        jobs_db=tmp_path / "jobs.db",
        generation_mode="text",
        omniroute_model="kilo-test",
    )
    settings.ensure_dirs()
    app = api_mod.create_app(settings)
    rt = app.state.runtime
    rt.colpali = FakeColpali()
    rt.text = FakeText()
    rt.retriever = api_mod.Retriever(settings, rt.store, rt.colpali, rt.text)
    rt.pipeline = api_mod.IngestPipeline(settings, rt.store, rt.colpali, rt.text)
    return TestClient(app)


def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["points"] == 0


def test_collection_info(client):
    r = client.get("/v1/collection")
    assert r.status_code == 200
    assert r.json()["points"] == 0


def test_query_empty_index(client):
    r = client.post("/v1/query", json={"query": "hello", "top_k": 5, "generate": False})
    assert r.status_code == 200
    body = r.json()
    assert body["pages"] == []


def test_query_rejects_blank(client):
    r = client.post("/v1/query", json={"query": "   "})
    assert r.status_code == 422


def test_query_returns_trace(client):
    r = client.post("/v1/query", json={"query": "hello", "top_k": 5, "generate": False})
    assert r.status_code == 200
    body = r.json()
    assert body["latency_ms"] >= 0
    assert body["trace"] is not None
    assert "legs" in body["trace"] and "fused" in body["trace"]
    assert body["trace"]["total_ms"] >= 0


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "ColPali RAG" in r.text
    assert 'id="chat"' in r.text
    assert "/v1/ingest" in r.text


def test_sources_empty(client):
    r = client.get("/v1/sources")
    assert r.status_code == 200
    assert r.json() == []


def test_ingest_requires_pdf(client):
    r = client.post("/v1/ingest", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 400