"""Retrieval-quality metrics: nDCG@k, MRR@k, Recall@k, and an aggregator.

All functions take a ranking (a best-first list of doc ids) and a ``qrel``
mapping (``doc_id -> relevance``, relevance a non-negative int; absent doc ids
are treated as relevance 0). They are pure, deterministic, and dependency-free
so they can be exercised by Hypothesis property tests without any ML stack.

The implementations follow Algorithm 3 in the design doc and satisfy the
correctness properties:
  * Property 4 — nDCG bounds:        0 <= ndcg_at_k <= 1, == 1 in ideal order.
  * Property 5 — MRR bounds:          MRR@k in [0, 1]; 1/rank of first relevant.
  * Property 6 — Recall monotonicity: recall_at_k is non-decreasing in k.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping


def _is_relevant(relevance: int) -> bool:
    """A judgment counts as relevant when its graded relevance is positive."""
    return relevance > 0


def ndcg_at_k(ranking: List[str], qrel: Mapping[str, int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at cutoff ``k``.

    Graded gain ``2**rel - 1`` with a ``log2(rank + 2)`` discount (rank
    0-indexed), normalized by the ideal DCG. Returns ``0.0`` when the query has
    no relevant documents (ideal DCG is 0), and a value in ``[0, 1]`` otherwise.
    """
    if k <= 0:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(ranking[:k]):
        rel = qrel.get(doc_id, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(rank + 2)

    ideal_gains = sorted((r for r in qrel.values() if r > 0), reverse=True)[:k]
    idcg = sum(
        (2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_gains)
    )

    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranking: List[str], qrel: Mapping[str, int], k: int = 10) -> float:
    """Mean Reciprocal Rank contribution for a single query at cutoff ``k``.

    Returns ``1 / rank`` (1-indexed) of the first relevant document within the
    top ``k``, or ``0.0`` if no relevant document appears in the top ``k``.
    """
    if k <= 0:
        return 0.0

    for rank, doc_id in enumerate(ranking[:k]):
        if _is_relevant(qrel.get(doc_id, 0)):
            return 1.0 / (rank + 1)
    return 0.0


def recall_at_k(ranking: List[str], qrel: Mapping[str, int], k: int = 100) -> float:
    """Fraction of the query's relevant documents retrieved within the top ``k``.

    Returns ``0.0`` when the query has no relevant documents. Non-decreasing in
    ``k`` by construction (a larger window can only retrieve more relevant docs).
    """
    if k <= 0:
        return 0.0

    relevant = {doc_id for doc_id, rel in qrel.items() if _is_relevant(rel)}
    if not relevant:
        return 0.0

    retrieved_relevant = sum(1 for doc_id in ranking[:k] if doc_id in relevant)
    return retrieved_relevant / len(relevant)


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean of an iterable of floats; ``0.0`` when empty."""
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def aggregate(per_query: Mapping[str, float]) -> float:
    """Mean of per-query metric values; ``0.0`` over an empty set of queries."""
    return mean(per_query.values())
