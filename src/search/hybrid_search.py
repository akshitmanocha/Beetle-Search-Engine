from collections import defaultdict

# NOTE: the search backends are imported lazily inside ``hybrid_search`` so that
# ``reciprocal_rank_fusion`` — a pure, dependency-free function — can be imported
# and property-tested without the ML stack present.


def reciprocal_rank_fusion(
    ranked_lists: "list[list[tuple]]",
    weights: "list[float] | None" = None,
    k: int = 60,
) -> dict:
    """Weighted Reciprocal Rank Fusion over several ranked lists.

    Each list is a rank-ordered sequence of ``(doc_id, score)`` (the score is
    ignored; only rank is used). Every document contributes
    ``weight * 1 / (k + rank + 1)`` to its fused score, with ``rank`` 0-indexed.

    Args:
        ranked_lists: one rank-ordered ``(doc_id, score)`` list per retriever.
        weights: per-list weights; defaults to equal weight ``1.0`` for each.
            A weight of ``0`` makes a list contribute nothing.
        k: the RRF constant (default 60).

    Returns:
        ``dict[doc_id -> fused_score]`` covering every doc in any input list.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match the number of "
            f"ranked lists ({len(ranked_lists)})."
        )

    rrf_scores: dict = defaultdict(float)
    for weight, results in zip(weights, ranked_lists):
        for rank, (doc_id, _score) in enumerate(results):
            rrf_scores[doc_id] += weight * (1.0 / (k + rank + 1))
    return dict(rrf_scores)


def hybrid_search(
    query: str,
    registry,
    top_k: int = 10,
    weights: "tuple" = (1.0, 1.0, 1.0),
):
    """Weighted hybrid search over BM25, dense (cosine), and SPLADE.

    Fuses the retrievers' ranked lists via weighted Reciprocal Rank Fusion
    (SPLADE included — fixes R3) and returns the top-k ``(doc_id, fused_score)``
    sorted descending.

    Args:
        query: the search query.
        registry: a built :class:`~src.serving.registry.Registry`.
        top_k: number of fused results to return (each retriever is queried at
            this depth).
        weights: ``(w_bm25, w_dense, w_splade)``. A weight of 0 disables that
            retriever's contribution.
    """
    from src.search.search_bm25 import search_bm25_registry
    from src.search.search_faiss import search_faiss_registry
    from src.search.search_splade import search_splade_registry

    ranked_lists = [
        search_bm25_registry(query, registry, top_k=top_k),
        search_faiss_registry(query, registry, top_k=top_k),
        search_splade_registry(query, registry, top_k=top_k),
    ]
    w = list(weights[:3])

    fused = reciprocal_rank_fusion(ranked_lists, weights=w, k=60)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]