from pathlib import Path
import yaml
from collections import defaultdict

from src.search.search_bm25 import search_bm25
from src.search.search_faiss import search_faiss

def reciprocal_rank_fusion(search_results: list[list[tuple]], k: int = 60) -> dict:
    """
    Performs Reciprocal Rank Fusion on a list of search results.

    Args:
        search_results: A list of lists of tuples, where each inner list represents
                        the search results from a different system, and each tuple
                        contains the document ID and the score.
        k: A constant used in the RRF formula.

    Returns:
        A dictionary where keys are document IDs and values are their RRF scores.
    """
    rrf_scores = defaultdict(float)

    for results in search_results:
        for rank, (doc_id, _) in enumerate(results):
            rrf_scores[doc_id] += 1 / (k + rank + 1)

    return rrf_scores

def hybrid_search(query: str, project_root: Path, top_k: int = 10):
    """
    Performs a hybrid search using BM25 and FAISS, and combines the results
    using Reciprocal Rank Fusion.

    Args:
        query: The query string.
        project_root: The root directory of the project.
        top_k: The number of top results to return.

    Returns:
        A list of tuples, where each tuple contains the document ID and its RRF score,
        sorted in descending order of score.
    """
    # Define paths
    bm25_index_path = project_root / "data" / "bm25_index"
    faiss_index_path = project_root / "data" / "faiss_index" / "faiss.index"
    doc_id_map_path = faiss_index_path.with_suffix('.json')

    # Load embedding model parameters
    with open(project_root / "params.yaml", "r") as f:
        embedding_params = yaml.safe_load(f)['models']['embedding']

    # Perform searches
    bm25_results = search_bm25(query, bm25_index_path, top_k=top_k)
    faiss_results = search_faiss(
        query,
        embedding_params['model_name'],
        embedding_params['device'],
        faiss_index_path,
        doc_id_map_path,
        top_k=top_k
    )

    # Combine results using RRF
    rrf_scores = reciprocal_rank_fusion([bm25_results, faiss_results])

    # Sort results by score
    sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    return sorted_results

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent
    sample_query = "transformer models for NLP"

    print(f"Performing hybrid search for: '{sample_query}'")
    results = hybrid_search(sample_query, project_root)

    print("\nTop search results:")
    for doc_id, score in results:
        print(f"  - Document ID: {doc_id}, Score: {score:.4f}")
