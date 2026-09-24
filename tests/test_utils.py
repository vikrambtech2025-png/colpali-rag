from __future__ import annotations

import numpy as np

from colpali_rag.qdrant_store import rrf_merge
from colpali_rag.utils import point_id, slugify


def test_slugify_safe() -> None:
    assert slugify("My  File (final)!.pdf") == "My-File-final.pdf"
    assert slugify("!!!") == "doc"


def test_point_id() -> None:
    import uuid

    pid = point_id("reports/Q3 2025 report.pdf", 2)
    # must be a valid UUID (Qdrant local mode requirement) and deterministic
    assert uuid.UUID(pid)
    assert point_id("reports/Q3 2025 report.pdf", 2) == pid


def test_rrf_prefers_overlap() -> None:
    merged = rrf_merge([[1, 2, 3], [9, 2, 1]], k=60)
    assert merged[1] > merged[2] > merged[3]
    assert merged[1] > merged[9]


def test_rrf_single_list() -> None:
    merged = rrf_merge([[7, 5]], k=60)
    assert merged[7] > merged[5]


def test_numpy_maxsim_math() -> None:
    """Mirror of the fallback MaxSim used in qdrant_store."""
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    doc = np.array([[0.5, 0.0], [0.0, 0.25], [1.0, 1.0]])
    dot = q @ doc.T
    score = float(dot.max(axis=1).sum())
    assert score == 2.0  # 1*1 (q0 vs doc2) + 1*1 (q1 vs doc2)