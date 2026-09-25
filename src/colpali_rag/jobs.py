"""Durable ingest job store (SQLite) + concurrency gates.

Jobs survive process restarts: every status change is written to a SQLite
table, and on boot any job left in "queued"/"running" by a crash is marked
"failed". A semaphore gate caps concurrent ingests so a burst of uploads
cannot monopolize the GPU (extra jobs wait, status stays "queued").
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    src         TEXT,
    report      TEXT,
    created_at  REAL NOT NULL,
    finished_at REAL
);
"""

_TERMINAL = {"done", "partial", "error", "failed"}
_COLUMNS = "job_id, status, src, report, created_at, finished_at"


class JobStore:
    """SQLite-backed job log; safe for the single API process."""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()
            self._fail_stranded()

    def _fail_stranded(self) -> None:
        """Mark jobs a crash left in flight as failed (restart recovery)."""
        now = time.time()
        self._conn.execute(
            "UPDATE ingest_jobs SET status='failed', finished_at=?, report=? "
            "WHERE status IN ('queued','running')",
            (now, json.dumps({"errors": ["server restarted while this job was in flight"]})),
        )
        self._conn.commit()

    def create(self, job_id: str, src: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO ingest_jobs (job_id, status, src, created_at) VALUES (?,?,?,?)",
                (job_id, "queued", src, time.time()),
            )
            self._conn.commit()

    def update(self, job_id: str, status: str, report: dict[str, Any] | None = None) -> None:
        finished = time.time() if status in _TERMINAL else None
        with self._lock:
            self._conn.execute(
                "UPDATE ingest_jobs SET status=?, report=?, finished_at=COALESCE(finished_at, ?) "
                "WHERE job_id=?",
                (status, json.dumps(report or {}), finished, job_id),
            )
            self._conn.commit()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM ingest_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return self._row(row) if row else None

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM ingest_jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(r) for r in rows]

    @staticmethod
    def _row(row: tuple) -> dict[str, Any]:
        return {
            "job_id": row[0],
            "status": row[1],
            "src": row[2],
            "report": json.loads(row[3]) if row[3] else None,
            "created_at": row[4],
            "finished_at": row[5],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class IngestGate:
    """Cap concurrent ingests; expose queue depth for health payloads."""

    def __init__(self, max_concurrent: int) -> None:
        self._sem = threading.Semaphore(max(1, max_concurrent))
        self._active = 0
        self._queued = 0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            self._queued += 1
        self._sem.acquire()
        with self._lock:
            self._queued -= 1
            self._active += 1

    def release(self) -> None:
        with self._lock:
            self._active -= 1
        self._sem.release()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"active": self._active, "queued": self._queued}


class QueryGate:
    """Cap concurrent GPU retrievals so parallel queries don't thrash VRAM."""

    def __init__(self, max_concurrent: int) -> None:
        self._sem = threading.Semaphore(max(1, max_concurrent))

    def __enter__(self) -> None:
        self._sem.acquire()

    def __exit__(self, *exc: object) -> None:
        self._sem.release()