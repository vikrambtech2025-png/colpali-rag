"""Durable ingest job store: restart survival + crash recovery."""
from __future__ import annotations

from colpali_rag.jobs import IngestGate, JobStore


def test_job_roundtrip_and_restart_survival(tmp_path) -> None:
    db = tmp_path / "jobs.db"
    store = JobStore(db)
    store.create("abc123", "doc.pdf")
    store.update("abc123", "running")
    store.update("abc123", "done", {"pages": 2, "duration_s": 1.5})
    row = store.get("abc123")
    assert row["status"] == "done"
    assert row["report"]["pages"] == 2
    assert row["src"] == "doc.pdf"
    store.close()
    # simulate a server restart: a fresh store on the same db still sees it
    reopened = JobStore(db)
    assert reopened.get("abc123")["status"] == "done"
    assert reopened.get("abc123")["report"]["pages"] == 2


def test_stranded_jobs_marked_failed_on_reopen(tmp_path) -> None:
    db = tmp_path / "jobs.db"
    s1 = JobStore(db)
    s1.create("j1", "a.pdf")
    assert s1.get("j1")["status"] == "queued"
    s1.close()
    s2 = JobStore(db)  # crash happened while j1 was queued/running
    assert s2.get("j1")["status"] == "failed"
    assert "restarted" in s2.get("j1")["report"]["errors"][0]


def test_recent_orders_newest_first(tmp_path) -> None:
    db = tmp_path / "jobs.db"
    store = JobStore(db)
    for i in range(5):
        store.create(f"job{i}", f"d{i}.pdf")
    recent = store.recent(2)
    assert [r["job_id"] for r in recent] == ["job4", "job3"]


def test_ingest_gate_snapshot(tmp_path) -> None:
    gate = IngestGate(2)
    assert gate.snapshot() == {"active": 0, "queued": 0}
    gate.acquire()
    gate.acquire()
    assert gate.snapshot()["active"] == 2
    gate.release()
    assert gate.snapshot()["active"] == 1