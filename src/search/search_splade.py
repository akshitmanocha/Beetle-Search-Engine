def search_splade_registry(query: str, registry, top_k: int = 10):
    """Search the registry's in-memory SPLADE index using the loaded model.

    Uses the startup-loaded SPLADE model + tokenizer and the in-memory inverted
    index (no per-query model reload or JSON re-read — fixes R2). Returns up to
    ``top_k`` ``(doc_id, score)`` tuples — the SAME shape as the other
    retrievers, so it can be fused by RRF directly (fixes the R3 adapter gap).
    """
    from src.serving.registry import generate_splade_vector

    inverted_index = registry.splade_index
    doc_id_map = registry.splade_doc_map
    if inverted_index is None or doc_id_map is None:
        return []

    query_vec = generate_splade_vector(
        query, registry.splade_model, registry.splade_tokenizer, registry.device()
    )

    scores: dict = {}
    for term_id, q_weight in query_vec.items():
        postings = inverted_index.get(term_id)
        if not postings:
            continue
        for doc_idx, d_weight in postings:
            scores[doc_idx] = scores.get(doc_idx, 0.0) + q_weight * d_weight

    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(doc_id_map[doc_idx], float(score)) for doc_idx, score in sorted_docs[:top_k]]
