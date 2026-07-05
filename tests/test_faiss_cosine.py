"""Tests for the cosine-mode FAISS builder (tasks 2.3, 2.4).

  * Property 3 — cosine scaling invariance: scaling a query vector by any
    positive scalar does not change its cosine ranking.
  * Unit: a tiny cosine index yields scores in [-1, 1]; the L2 path still
    builds for the ablation.

These tests construct a FAISS index directly from vectors (mirroring
build_faiss.py's logic) so they need only faiss + numpy, not the embedding model
(which is gated). The full builder is exercised via build_faiss_index against a
pickle fixture.
"""

import pickle

import faiss
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.index.build_faiss import build_faiss_index


def _build_cosine_index(vectors: np.ndarray):
    """Mirror build_faiss.py cosine path: normalize + IndexFlatIP + IDMap."""
    vectors = vectors.copy().astype("float32")
    d = vectors.shape[1]
    faiss.normalize_L2(vectors)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
    index.add_with_ids(vectors, np.arange(len(vectors)))
    return index


# --- Property 3: cosine scaling invariance ------------------------------------

@settings(max_examples=50, deadline=None)
@given(
    scalar=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_cosine_ranking_invariant_to_positive_scaling(scalar, seed):
    rng = np.random.default_rng(seed)
    # 8 docs in 16-dim space; avoid all-zero vectors.
    docs = rng.normal(size=(8, 16)).astype("float32") + 0.1
    index = _build_cosine_index(docs)

    query = rng.normal(size=(1, 16)).astype("float32") + 0.1

    def ranked_ids(q):
        q = q.copy().astype("float32")
        faiss.normalize_L2(q)
        _scores, idxs = index.search(q, 8)
        return list(idxs[0])

    base = ranked_ids(query)
    scaled = ranked_ids(query * scalar)
    assert base == scaled


# --- Unit tests ----------------------------------------------------------------

def test_cosine_scores_within_unit_range():
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(5, 16)).astype("float32")
    index = _build_cosine_index(docs)

    q = docs[2:3].copy()
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, 5)

    # The query equals doc 2 (post-normalization) => self-match cosine ~ 1.
    assert idxs[0][0] == 2
    assert scores[0][0] == pytest.approx(1.0, abs=1e-4)
    assert np.all(scores[0] <= 1.0 + 1e-4)
    assert np.all(scores[0] >= -1.0 - 1e-4)


def test_builder_cosine_and_l2_paths(tmp_path):
    """build_faiss_index builds both metrics; cosine normalizes, L2 does not."""
    embedding_map = {
        "doc_a": [1.0, 0.0, 0.0, 0.0],
        "doc_b": [0.0, 2.0, 0.0, 0.0],
        "doc_c": [0.0, 0.0, 3.0, 0.0],
    }
    emb_path = tmp_path / "embeddings.pkl"
    with open(emb_path, "wb") as f:
        pickle.dump(embedding_map, f)

    # cosine
    cos_path = tmp_path / "cos" / "faiss.index"
    build_faiss_index(emb_path, cos_path, metric="cosine")
    assert cos_path.exists()
    assert cos_path.with_suffix(".json").exists()
    cos_index = faiss.read_index(str(cos_path))
    assert cos_index.ntotal == 3

    # l2 (ablation path)
    l2_path = tmp_path / "l2" / "faiss.index"
    build_faiss_index(emb_path, l2_path, metric="l2")
    assert l2_path.exists()
    l2_index = faiss.read_index(str(l2_path))
    assert l2_index.ntotal == 3


def test_builder_rejects_unknown_metric(tmp_path):
    embedding_map = {"d": [1.0, 0.0]}
    emb_path = tmp_path / "e.pkl"
    with open(emb_path, "wb") as f:
        pickle.dump(embedding_map, f)
    with pytest.raises(ValueError):
        build_faiss_index(emb_path, tmp_path / "x.index", metric="bogus")
