"""Spike: validate the full stack on real hardware.

1. ColPali model load + VRAM measurement
2. Qdrant local-mode collection with multivector late-interaction
3. Embed 2 images -> upsert -> LateInteraction query -> hybrid query

Run:  uv run python -m scripts.spike
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from colpali_rag.config import Settings, configure_hf_env  # noqa: E402
from colpali_rag.embeddings import ColPaliEmbedder, TextEmbedder  # noqa: E402
from colpali_rag.qdrant_store import QdrantStore  # noqa: E402
from colpali_rag.utils import timed  # noqa: E402


def vr_mb() -> float:
    import torch

    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def nvidia() -> str:
    import subprocess

    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    except Exception:
        return "nvidia-smi unavailable"


def main() -> int:
    settings = Settings()
    settings.ensure_dirs()
    configure_hf_env(settings)
    print("nvidia-smi:", nvidia())
    print("device:", settings.resolved_device)

    # 1) model load
    colpali = ColPaliEmbedder(settings)
    t0 = time.perf_counter()
    with timed("colqwen2 load"):
        colpali.load()
    print(f"VRAM after load: {vr_mb():.0f} MB")

    # 2) sample images
    demo = Path(settings.corpus_dir)
    imgs = [Image.open(p).convert("RGB") for p in sorted(demo.glob("*.png"))[:2]]
    print(f"sample images: {len(imgs)}")
    per = colpali.embed_images(imgs)
    for i, v in enumerate(per):
        print(f"  image {i}: shape={v.shape} dtype={v.dtype}")

    q = colpali.embed_query("what does the revenue chart show?")
    print(f"query tokens: {q.shape}")

    # 3) qdrant multivector collection
    store = QdrantStore(settings)
    with timed("ensure_collection"):
        store.ensure_collection()
    print("collection exists:", store.client.collection_exists(settings.collection))
    print("native late interaction supported:", store._supports_native_late())

    # text embedder
    text = TextEmbedder(settings)
    with timed("bge-m3 load"):
        text.load()
    d1, s1 = text.embed_documents(["revenue chart product lines quarterly"])[0]
    print(f"dense shape={d1.shape} sparse_tokens={len(s1)}")

    store.upsert_page(
        src="spike.pdf", page=1,
        colpali_vecs=per[0], dense_vec=d1, sparse=s1,
        payload={"src": "spike.pdf", "page": 1, "text": "revenue chart product lines quarterly", "image": "spike/page-001.png"},
    )
    store.upsert_page(
        src="spike.pdf", page=2,
        colpali_vecs=per[1], dense_vec=d1, sparse=s1,
        payload={"src": "spike.pdf", "page": 2, "text": "revenue chart product lines quarterly", "image": "spike/page-002.png"},
    )
    print("points:", store.count())

    with timed("colpali-only query"):
        r1 = store.query_colpali(q, 5)
    print("colpali legs:", [(p.page, round(p.score, 3)) for p in r1])

    with timed("hybrid query (server fusion)"):
        try:
            r2 = store.hybrid_query(q, d1, s1, 5)
            mode = "server fusion"
        except Exception as exc:
            print("server fusion failed:", exc)
            r2 = store.hybrid_query_python(q, d1, s1, 5)
            mode = "python RRF"
    print(f"hybrid ({mode}):", [(p.page, round(p.score, 3)) for p in r2])

    print("\nSPIKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())