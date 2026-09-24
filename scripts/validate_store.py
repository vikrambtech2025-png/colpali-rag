"""Scratch validation of Qdrant store (multivector late-interaction + hybrid fusion)
using synthetic vectors against a temporary Qdrant. No torch/GPU needed."""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from colpali_rag.config import Settings
from colpali_rag.qdrant_store import QdrantStore

scratch = Path(__file__).resolve().parents[1] / "data" / "qdrant-scratch"
if scratch.exists():
    shutil.rmtree(scratch)
s = Settings(qdrant_path=scratch)
st = QdrantStore(s)

st.ensure_collection()
print("collection exists:", st.client.collection_exists("docs"))

rng = np.random.default_rng(0)
# 4 fake "pages": 3 irrelevant, 1 similar to our synthetic query
docs = [rng.standard_normal((24, 128)) * 0.1 for _ in range(3)]
docs.append(rng.standard_normal((24, 128)) * 0.1)
q = rng.standard_normal((6, 128)) * 0.1
docs[-1][: len(q)] = q * 2.0  # page 4 shares the query's patch vectors -> strong MaxSim

for i, v in enumerate(docs, start=1):
    st.upsert_page(
        src="scratch.pdf",
        page=i,
        colpali_vecs=v,
        dense_vec=np.zeros(1024, dtype=np.float32),
        sparse={i: 1.0},
        payload={"src": "scratch.pdf", "page": i, "text": f"page {i}", "image": ""},
    )
print("count:", st.count())
print("native late interaction:", st._supports_native_late())

res = st.query_colpali(q, 4)
print("colpali leg  ->", [(p.page, round(p.score, 3)) for p in res])
assert res[0].page == 4, "native MaxSim should rank the crafted match first"

res2 = st.hybrid_query(q, np.zeros(1024, dtype=np.float32), {42: 1.0}, 4)
print("hybrid leg   ->", [(p.page, round(p.score, 4)) for p in res2])
assert res2, "hybrid query returned nothing"

res3 = st.hybrid_query_python(q, np.zeros(1024, dtype=np.float32), {42: 1.0}, 4)
print("py-RRF leg   ->", [(p.page, round(p.score, 4)) for p in res3])
assert res3, "python RRF returned nothing"

print("\nQDRANT STORE VALIDATION OK")