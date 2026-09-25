"""Production-hardening tests: auth, rate limits, upload caps, readiness.

All tests avoid model loading: they exercise dependency layers (auth / rate
limiting / size caps) or read endpoints that never touch the embedders.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from colpali_rag import api as api_mod
from colpali_rag.config import Settings


def _make_client(tmp_path, **overrides) -> TestClient:
    settings = Settings(
        qdrant_path=tmp_path / "qdrant",
        pages_dir=tmp_path / "pages",
        hf_cache=tmp_path / "hf",
        manifests_dir=tmp_path / "manifests",
        corpus_dir=tmp_path / "corpus",
        jobs_db=tmp_path / "jobs.db",
        generation_mode="text",
        omniroute_model="kilo-test",
        **overrides,
    )
    settings.ensure_dirs()
    app = api_mod.create_app(settings)
    return TestClient(app)


def test_auth_required_when_key_set(tmp_path) -> None:
    client = _make_client(tmp_path, api_key="sekret")
    assert client.get("/v1/sources").status_code == 401
    assert client.get("/v1/collection").status_code == 401
    assert client.get("/v1/ingest/nope").status_code == 401
    assert client.post("/v1/query", json={"query": "hi"}).status_code == 401
    # correct key passes
    ok = client.get("/v1/sources", headers={"X-API-Key": "sekret"})
    assert ok.status_code == 200
    # wrong key is rejected
    assert client.get("/v1/sources", headers={"X-API-Key": "nope"}).status_code == 401


def test_health_and_ready_are_public(tmp_path) -> None:
    client = _make_client(tmp_path, api_key="sekret")
    assert client.get("/v1/health").status_code == 200
    r = client.get("/v1/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_rate_limit_ingest_per_client(tmp_path) -> None:
    client = _make_client(tmp_path, rate_limit_ingest_per_minute=2)
    # non-PDF uploads: rejected at the handler (400), but the rate-limit
    # dependency still counts them; request 3 must be throttled (429).
    for _ in range(3):
        r = client.post("/v1/ingest", files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 429
    codes = [client.post("/v1/ingest", files={"file": ("x.txt", b"hi", "text/plain")}).status_code for _ in range(2)]
    assert codes == [429, 429]  # still throttled this minute


def test_upload_size_cap(tmp_path) -> None:
    client = _make_client(tmp_path, max_upload_mb=1)
    big = b"x" * (2 * 1024 * 1024)
    r = client.post("/v1/ingest", files={"file": ("big.pdf", big, "application/pdf")})
    assert r.status_code == 413
    assert "cap" in r.json()["detail"]


def test_health_payload_has_observability(tmp_path) -> None:
    client = _make_client(tmp_path)
    body = client.get("/v1/health").json()
    assert "uptime_s" in body
    assert "ingests" in body
    assert body["settings"]["auth"] == "off"
    assert body["settings"]["rate_limit_per_minute"] >= 1