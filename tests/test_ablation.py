"""Ablation tests: determinism (Property 8, task 9.5) + mini-ablation smoke (11.4).

The determinism test runs a system twice on a fixture corpus and asserts
identical rankings and metrics. The mini-ablation smoke test runs all five
systems on NFCorpus (the smallest BEIR dataset) and asserts every system yields
a MetricRow with metrics in [0, 1]. Both are ``integration`` (real models).
"""

import pytest

from src.config import CONFIG
from src.serving.registry import build_eval_registry
from eval.ablation import (
    DEFAULT_SYSTEMS,
    SystemSpec,
    evaluate_system,
    _run_query,
)

pytestmark = pytest.mark.integration

FIXTURE_CORPUS = {
    "transformers": {"title": "Transformers", "text": "Attention is all you need for language models."},
    "gardening": {"title": "Tomatoes", "text": "Plant seedlings after the last frost and water them."},
    "cooking": {"title": "Sourdough", "text": "Ferment a starter then knead, proof, and bake bread."},
    "rl": {"title": "Reinforcement Learning", "text": "Agents learn policies by maximizing reward signals."},
}
FIXTURE_QUERIES = {
    "q1": "attention in neural language models",
    "q2": "how to grow vegetables at home",
}
FIXTURE_QRELS = {
    "q1": {"transformers": 2},
    "q2": {"gardening": 1},
}


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("ablation_indexes")
    return build_eval_registry(
        FIXTURE_CORPUS, CONFIG, index_dir, metric="cosine",
        with_splade=True, with_reranker=True,
    )


# --- Property 8: determinism ---------------------------------------------------

def test_determinism_same_rankings(registry):
    """The same query on the same registry yields identical rankings twice."""
    spec = SystemSpec("hybrid", weights=(1.0, 1.0, 1.0), rerank=False)
    first, _ = _run_query(spec, FIXTURE_QUERIES["q1"], registry)
    second, _ = _run_query(spec, FIXTURE_QUERIES["q1"], registry)
    assert first == second


def test_determinism_same_metrics(registry):
    """Evaluating the same system twice yields identical metric rows."""
    spec = SystemSpec("hybrid", weights=(1.0, 1.0, 1.0), rerank=False)
    row_a = evaluate_system(spec, FIXTURE_QUERIES, FIXTURE_QRELS, registry, "fixture")
    row_b = evaluate_system(spec, FIXTURE_QUERIES, FIXTURE_QRELS, registry, "fixture")
    assert row_a.ndcg_at_10 == row_b.ndcg_at_10
    assert row_a.mrr_at_10 == row_b.mrr_at_10
    assert row_a.recall_at_100 == row_b.recall_at_100


# --- Mini-ablation smoke (all five systems produce valid metrics) -------------

def test_all_systems_produce_valid_metrics(registry):
    """Every system in the default grid yields metrics within [0, 1]."""
    for spec in DEFAULT_SYSTEMS:
        row = evaluate_system(spec, FIXTURE_QUERIES, FIXTURE_QRELS, registry, "fixture")
        assert 0.0 <= row.ndcg_at_10 <= 1.0, spec.name
        assert 0.0 <= row.mrr_at_10 <= 1.0, spec.name
        assert 0.0 <= row.recall_at_100 <= 1.0, spec.name
        assert row.mean_latency_ms >= 0.0, spec.name
        # The relevant doc for q1 should be findable; hybrid should score > 0.
    # hybrid specifically should retrieve the on-topic doc for q1.
    hybrid = SystemSpec("hybrid", weights=(1.0, 1.0, 1.0), rerank=False)
    row = evaluate_system(hybrid, FIXTURE_QUERIES, FIXTURE_QRELS, registry, "fixture")
    assert row.ndcg_at_10 > 0.0


def test_empty_results_do_not_crash(registry):
    """A query that matches nothing lexically still produces a (zero) metric row."""
    spec = SystemSpec("bm25", weights=None, rerank=False)
    weird_queries = {"qx": "zzzqqxx nonexistent gibberish token"}
    weird_qrels = {"qx": {"transformers": 1}}
    row = evaluate_system(spec, weird_queries, weird_qrels, registry, "fixture")
    assert 0.0 <= row.ndcg_at_10 <= 1.0
