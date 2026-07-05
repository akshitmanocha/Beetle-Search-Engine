def rerank(query: str, candidates, registry, max_chars: int = 2000):
    """Re-rank ``candidates`` with the registry's cross-encoder (task 4.1).

    Uses the startup-loaded reranker (no per-request instantiation — fixes S2),
    concatenates ``title + " " + text`` and truncates to ``max_chars`` before
    scoring (fixes R7), and returns candidates re-sorted by cross-encoder score
    descending WITHOUT mutating the input ``candidates`` list.

    Args:
        query: the search query.
        candidates: list of ``(doc_id, score)`` pairs to re-rank.
        registry: a built Registry whose ``corpus`` contains the candidate docs.
        max_chars: max characters of concatenated title+text fed to the model.

    Returns:
        A new list of ``(doc_id, rerank_score)`` sorted by score descending.
    """
    if not candidates:
        return []

    doc_ids = [doc_id for doc_id, _ in candidates]
    pairs = []
    for doc_id in doc_ids:
        doc = registry.corpus.get(doc_id, {})
        title = doc.get("title", "") or ""
        text = doc.get("text", doc.get("body_text", "")) or ""
        combined = f"{title} {text}"[:max_chars]
        pairs.append((query, combined))

    scores = registry.reranker.predict(pairs, show_progress_bar=False)
    # Build a NEW list; do not mutate the caller's `candidates`.
    reranked = sorted(
        zip(doc_ids, (float(s) for s in scores)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return reranked
