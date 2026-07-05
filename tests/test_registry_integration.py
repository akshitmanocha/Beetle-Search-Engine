"""Integration tests for the registry + registry-backed retrievers + reranker.

These load the real models (embeddinggemma, SPLADE, cross-encoder) and build
in-memory indexes over a tiny fixture corpus, so they are slow and require the
gated embedder to be accessible (HF token in the cache). They are marked
``integration`` so the fast unit/property suite can be run alone with
``-m "not integration"``.

Covers:
  * Registry build over a corpus dict (task 5.1).
  * Each registry-backed retriever returns (doc_id, score) and finds the
    on-topic doc (task 5.2).
  * SPLADE returns (doc_id, score) tuples — same shape as the others (R3).
  * Three-way hybrid fusion runs and returns fused results (task 3.2).
  * Reranker truncates and does not mutate input order (task 4.1).
"""

import pytest

from src.config import CONFIG
from src.serving.registry import build_eval_registry
from src.search.search_bm25 import search_bm25_registry
from src.search.search_faiss import search_faiss_registry
from src.search.search_splade import search_splade_registry
from src.search.hybrid_search import hybrid_search
from src.models.reranker import rerank

pytestmark = pytest.mark.integration

# A tiny, clearly-separable corpus: a transformers doc, a gardening doc, a cooking doc.
FIXTURE_CORPUS = {
    "transformers": {
        "title": "The Transformer Architecture",
        "text": "Attention is all you need. Self-attention and multi-head attention "
                "power large language models for natural language processing.",
    },
    "gardening": {
        "title": "Growing Tomatoes",
        "text": "Plant tomato seedlings after the last frost. Water regularly and "
                "provide plenty of sunlight for a healthy harvest.",
    },
    "cooking": {
        "title": "Sourdough Bread",
        "text": "Mix flour and water to make a starter. Ferment, knead, proof, and "
                "bake at high temperature for a crusty loaf.",
    },
}


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("eval_indexes")
    return build_eval_registry(
        FIXTURE_CORPUS, CONFIG, index_dir, metric="cosine",
        with_splade=True, with_reranker=True,
    )


def test_registry_builds(registry):
    assert registry.is_ready()
    assert registry.bm25_index is not None
    assert registry.faiss_index is not None
    assert registry.faiss_index.ntotal == 3
    assert registry.splade_index is not None
    assert registry.embed_model is not None
    assert registry.reranker is not None
    assert registry.missing_artifacts() == []


def test_bm25_finds_on_topic(registry):
    results = search_bm25_registry("transformer attention models", registry, top_k=3)
    assert results
    assert all(len(r) == 2 for r in results)
    assert results[0][0] == "transformers"


def test_dense_finds_on_topic(registry):
    results = search_faiss_registry("how do neural language models use attention", registry, top_k=3)
    assert results
    # Cosine scores are bounded by 1.
    assert all(-1.01 <= score <= 1.01 for _, score in results)
    assert results[0][0] == "transformers"


def test_splade_returns_tuples_and_finds_on_topic(registry):
    results = search_splade_registry("natural language processing attention", registry, top_k=3)
    assert results
    # Critical: SPLADE returns (doc_id, score) tuples, same shape as others.
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert results[0][0] == "transformers"


def test_hybrid_fuses_three_retrievers(registry):
    results = hybrid_search("attention in language models", registry, top_k=3)
    assert results
    assert all(len(r) == 2 for r in results)
    assert results[0][0] == "transformers"
    # Fused scores are descending.
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_weight_zero_disables_retriever(registry):
    # With only dense enabled, still finds the on-topic doc.
    results = hybrid_search("attention models", registry, top_k=3, weights=(0.0, 1.0, 0.0))
    assert results
    assert results[0][0] == "transformers"


def test_reranker_truncates_and_preserves_input(registry):
    candidates = [("gardening", 0.1), ("transformers", 0.05), ("cooking", 0.02)]
    original = list(candidates)

    reranked = rerank("transformer attention for NLP", candidates, registry, max_chars=100)

    # Input list is not mutated.
    assert candidates == original
    # Output is (doc_id, score) sorted descending.
    assert all(len(r) == 2 for r in reranked)
    scores = [s for _, s in reranked]
    assert scores == sorted(scores, reverse=True)
    # The on-topic doc should rise to the top after reranking.
    assert reranked[0][0] == "transformers"


def test_reranker_empty_candidates(registry):
    assert rerank("anything", [], registry) == []
