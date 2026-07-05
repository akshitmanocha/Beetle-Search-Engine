"""Tests for retrieval metrics: property-based (Properties 4-6) + hand-computed.

Property tests (tasks 7.2-7.4):
  * Property 4 — nDCG bounds:        0 <= ndcg_at_k <= 1; == 1 in ideal order.
  * Property 5 — MRR bounds:          MRR@k in [0, 1]; 1/rank of first relevant.
  * Property 6 — Recall monotonicity: recall_at_k non-decreasing in k.

Unit tests (task 7.5): hand-computed nDCG/MRR/Recall on tiny fixtures, plus the
no-relevant-docs and perfect-ranking boundaries.
"""

import math

from hypothesis import given
from hypothesis import strategies as st

from eval.metrics import aggregate, mrr_at_k, ndcg_at_k, recall_at_k


# --- Hypothesis strategies -----------------------------------------------------

# A small universe of doc ids keeps rankings and qrels overlapping meaningfully.
doc_ids = st.sampled_from([f"d{i}" for i in range(12)])

rankings = st.lists(doc_ids, max_size=12, unique=True)

# qrels: doc_id -> non-negative graded relevance.
qrels = st.dictionaries(
    keys=doc_ids,
    values=st.integers(min_value=0, max_value=3),
    max_size=12,
)

cutoffs = st.integers(min_value=1, max_value=15)


# --- Property 4: nDCG bounds ---------------------------------------------------

@given(ranking=rankings, qrel=qrels, k=cutoffs)
def test_ndcg_within_bounds(ranking, qrel, k):
    score = ndcg_at_k(ranking, qrel, k)
    assert 0.0 <= score <= 1.0 + 1e-9


@given(qrel=qrels, k=cutoffs)
def test_ndcg_ideal_order_is_one(qrel, k):
    """A ranking placing docs in descending-relevance order scores nDCG == 1."""
    relevant = [doc_id for doc_id, rel in qrel.items() if rel > 0]
    ideal_ranking = sorted(relevant, key=lambda d: qrel[d], reverse=True)
    if not ideal_ranking:
        # No relevant docs => nDCG is 0 by definition.
        assert ndcg_at_k(ideal_ranking, qrel, k) == 0.0
    else:
        assert ndcg_at_k(ideal_ranking, qrel, k) == 1.0


# --- Property 5: MRR bounds ----------------------------------------------------

@given(ranking=rankings, qrel=qrels, k=cutoffs)
def test_mrr_within_bounds(ranking, qrel, k):
    score = mrr_at_k(ranking, qrel, k)
    assert 0.0 <= score <= 1.0


@given(ranking=rankings, qrel=qrels, k=cutoffs)
def test_mrr_equals_reciprocal_of_first_relevant(ranking, qrel, k):
    """MRR@k equals 1/rank of the first relevant doc in top-k, else 0."""
    expected = 0.0
    for rank, doc_id in enumerate(ranking[:k]):
        if qrel.get(doc_id, 0) > 0:
            expected = 1.0 / (rank + 1)
            break
    assert mrr_at_k(ranking, qrel, k) == expected


# --- Property 6: Recall monotonicity in k -------------------------------------

@given(
    ranking=rankings,
    qrel=qrels,
    k_pair=st.tuples(cutoffs, cutoffs),
)
def test_recall_monotonic_in_k(ranking, qrel, k_pair):
    k1, k2 = sorted(k_pair)
    assert recall_at_k(ranking, qrel, k1) <= recall_at_k(ranking, qrel, k2) + 1e-9


@given(ranking=rankings, qrel=qrels, k=cutoffs)
def test_recall_within_bounds(ranking, qrel, k):
    assert 0.0 <= recall_at_k(ranking, qrel, k) <= 1.0


# --- Hand-computed unit fixtures (task 7.5) -----------------------------------

def test_ndcg_hand_computed():
    """Two relevant docs (rel=1) at ranks 1 and 3 (0-indexed 0 and 2)."""
    ranking = ["a", "x", "b", "y"]
    qrel = {"a": 1, "b": 1}
    # DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG (ideal: a,b at ranks 0,1) = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(ranking, qrel, 10) == dcg / idcg


def test_ndcg_graded_relevance():
    """Graded gain uses 2**rel - 1."""
    ranking = ["a", "b"]
    qrel = {"a": 3, "b": 1}
    dcg = (2 ** 3 - 1) / math.log2(2) + (2 ** 1 - 1) / math.log2(3)
    idcg = dcg  # already ideal order
    assert math.isclose(ndcg_at_k(ranking, qrel, 10), dcg / idcg)
    assert ndcg_at_k(ranking, qrel, 10) == 1.0


def test_ndcg_no_relevant_docs_is_zero():
    assert ndcg_at_k(["a", "b", "c"], {}, 10) == 0.0
    assert ndcg_at_k(["a", "b", "c"], {"z": 1}, 10) == 0.0


def test_mrr_hand_computed():
    # First relevant ("b") at 0-indexed rank 1 => 1/2.
    assert mrr_at_k(["a", "b", "c"], {"b": 1, "c": 1}, 10) == 0.5
    # First relevant at rank 0 => 1.0.
    assert mrr_at_k(["b", "a"], {"b": 1}, 10) == 1.0
    # None in top-k => 0.
    assert mrr_at_k(["a", "c"], {"z": 1}, 10) == 0.0


def test_mrr_respects_cutoff():
    # Relevant doc is at rank 3 (0-indexed 2) but k=2 excludes it.
    assert mrr_at_k(["a", "b", "rel"], {"rel": 1}, 2) == 0.0
    assert mrr_at_k(["a", "b", "rel"], {"rel": 1}, 3) == 1.0 / 3


def test_recall_hand_computed():
    qrel = {"a": 1, "b": 1, "c": 1, "d": 1}
    # 2 of 4 relevant docs retrieved within top-3.
    assert recall_at_k(["a", "x", "b"], qrel, 3) == 0.5
    # All 4 retrieved within top-10.
    assert recall_at_k(["a", "b", "c", "d"], qrel, 10) == 1.0


def test_recall_no_relevant_docs_is_zero():
    assert recall_at_k(["a", "b"], {}, 10) == 0.0


def test_aggregate_is_mean():
    assert aggregate({"q1": 1.0, "q2": 0.0}) == 0.5
    assert aggregate({}) == 0.0
