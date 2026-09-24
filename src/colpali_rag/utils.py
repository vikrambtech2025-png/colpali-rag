"""Small shared helpers: ids, timing, text normalization."""
from __future__ import annotations

import hashlib
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def slugify(text: str, max_len: int = 64) -> str:
    """Filesystem-safe slug (keeps dots inside names)."""
    slug = re.sub(r"[^a-zA-Z0-9._]+", "-", text)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.replace("-.", ".").replace(".-", ".")
    slug = slug.strip(".-_")
    return slug[:max_len] or "doc"


def point_id(src: str, page: int) -> str:
    """Deterministic UUID for a (source, page) pair (Qdrant requires UUIDs for
    string ids in local mode; deterministic ids make re-ingest idempotent)."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{src}::{int(page)}"))


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


@contextmanager
def timed(name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        print(f"[timing] {name}: {elapsed:.2f}s")


def now_iso() -> str:
    import datetime

    return datetime.datetime.now().isoformat(timespec="seconds")