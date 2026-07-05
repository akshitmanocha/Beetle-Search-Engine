import faiss
import numpy as np


def _normalize_query(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a query vector in place using the same op as the index."""
    vec = np.ascontiguousarray(vec, dtype="float32")
    faiss.normalize_L2(vec)
    return vec


def search_faiss_registry(query: str, registry, top_k: int = 10):
    """Cosine search using the startup-loaded embedding model and FAISS index.

    Encodes the query with the registry's embedding model (no per-query model
    reload — fixes R4), L2-normalizes the query vector to match the cosine
    index, searches, and returns up to ``top_k`` ``(doc_id, score)`` pairs,
    skipping the FAISS sentinel id ``-1``. Higher score = more similar.
    """
    model = registry.embed_model

    # embeddinggemma is prompt-based: use encode_query when available so the
    # query side gets the query prompt (see beetle-embedder-facts memory).
    if hasattr(model, "encode_query"):
        q = model.encode_query(query, convert_to_numpy=True)
    else:
        q = model.encode([query], convert_to_numpy=True)
    q = np.asarray(q, dtype="float32").reshape(1, -1)

    if registry.faiss_metric == "cosine":
        q = _normalize_query(q)

    scores, idxs = registry.faiss_index.search(q, top_k)
    results = []
    for i, s in zip(idxs[0], scores[0]):
        if i != -1:
            results.append((registry.faiss_doc_ids[i], float(s)))
    return results
