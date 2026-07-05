"""Property-based + unit tests for weighted Reciprocal Rank Fusion.

Property tests (tasks 3.3-3.5):
  * Property 1 — rank-monotonicity: for a single list with equal weights,
    earlier-ranked docs get a fused score >= later-ranked docs.
  * Property 2 — weight zeroing: a retriever with weight 0 yields fused scores
    identical to fusing only the remaining lists.
  * Property 7 — fusion completeness: every doc in any input list appears in
    the fused output.
"""

from hypothesis import given
from hypothesis import strategies as st

from src.search.hybrid_search import reciprocal_rank_fusion


# --- Strategies ----------------------------------------------------------------

doc_ids = st.text(alphabet="abcdefghij", min_size=1, max_size=3)


def ranked_list(max_size=10):
    """A rank-ordered list of (doc_id, score) with unique doc ids."""
    return st.lists(doc_ids, max_size=max_size, unique=True).map(
        lambda ids: [(d, 0.0) for d in ids]
    )


list_of_ranked_lists = st.lists(ranked_list(), min_size=1, max_size=4)


# --- Property 1: rank-monotonicity --------------------------------------------

@given(results=ranked_list(max_size=12))
def test_rrf_rank_monotonicity(results):
    """For a single equal-weight list, fused score is non-increasing with rank."""
    fused = reciprocal_rank_fusion([results])
    for i in range(len(results) - 1):
        doc_i = results[i][0]
        doc_j = results[i + 1][0]
        assert fused[doc_i] >= fused[doc_j]


# --- Property 2: weight zeroing -----------------------------------------------

@given(lists=st.lists(ranked_list(), min_size=2, max_size=4), data=st.data())
def test_rrf_weight_zeroing(lists, data):
    """A list with weight 0 contributes the same as omitting it entirely."""
    t = data.draw(st.integers(min_value=0, max_value=len(lists) - 1))

    weights = [1.0] * len(lists)
    weights[t] = 0.0
    with_zero = reciprocal_rank_fusion(lists, weights=weights)

    remaining = [lst for idx, lst in enumerate(lists) if idx != t]
    without_t = reciprocal_rank_fusion(remaining) if remaining else {}

    # Docs that ONLY appeared in the zeroed list now have fused score 0, but
    # still appear (completeness). Compare scores on the shared doc universe.
    for doc_id in set(without_t) | set(with_zero):
        assert with_zero.get(doc_id, 0.0) == without_t.get(doc_id, 0.0)


# --- Property 7: fusion completeness ------------------------------------------

@given(lists=list_of_ranked_lists)
def test_rrf_completeness(lists):
    """Every doc appearing in any input list appears in the fused output."""
    fused = reciprocal_rank_fusion(lists)
    all_docs = {doc_id for lst in lists for doc_id, _ in lst}
    assert set(fused.keys()) == all_docs


# --- Unit tests ----------------------------------------------------------------

def test_rrf_equal_weights_hand_computed():
    k = 60
    a = [("d1", 0.9), ("d2", 0.8)]
    b = [("d2", 0.7), ("d3", 0.6)]
    fused = reciprocal_rank_fusion([a, b], k=k)
    # d1: rank0 in a => 1/61
    # d2: rank1 in a + rank0 in b => 1/62 + 1/61
    # d3: rank1 in b => 1/62
    assert fused["d1"] == 1 / 61
    assert fused["d2"] == 1 / 62 + 1 / 61
    assert fused["d3"] == 1 / 62
    # d2 fused highest.
    assert max(fused, key=fused.get) == "d2"


def test_rrf_weights_scale_contribution():
    a = [("d1", 0.0)]
    b = [("d1", 0.0)]
    base = reciprocal_rank_fusion([a, b], weights=[1.0, 1.0])
    weighted = reciprocal_rank_fusion([a, b], weights=[2.0, 1.0])
    assert weighted["d1"] > base["d1"]


def test_rrf_mismatched_weights_raises():
    import pytest
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[("d1", 0.0)]], weights=[1.0, 1.0])


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[]]) == {}
