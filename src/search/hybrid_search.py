import os
import sys
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import yaml
from collections import defaultdict
import torch
import json
from src.search.search_bm25 import search_bm25
from src.search.search_faiss import search_faiss

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

def reciprocal_rank_fusion(search_results: list[list[tuple]], k: int = 60) -> dict:
    """
    Performs Reciprocal Rank Fusion on a list of search results.
    """
    rrf_scores = defaultdict(float)
    for results in search_results:
        for rank, (doc_id, _) in enumerate(results):
            rrf_scores[doc_id] += 1 / (k + rank + 1)
    return rrf_scores

def hybrid_search(query: str, top_k: int = 10):
    """
    Performs a hybrid search.
    """
    # Define paths
    bm25_index_path = project_root / "data" / "bm25_index"
    faiss_index_path = project_root / "data" / "faiss_index" / "faiss.index"
    doc_id_map_path = faiss_index_path.with_suffix('.json')

    # Load parameters
    with open(project_root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)['models']
    embedding_params = params['embedding']

    print(f"Using device: {device}")

    # Perform initial search
    bm25_results = search_bm25(query, bm25_index_path, top_k=top_k)
    faiss_results = search_faiss(
        query,
        embedding_params['model_name'],
        device,
        faiss_index_path,
        doc_id_map_path,
        top_k=top_k
    )

    # Fuse results
    rrf_scores = reciprocal_rank_fusion([bm25_results, faiss_results])
    sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    return sorted_results[:top_k]

if __name__ == '__main__':
    sample_query = "transformer models for NLP"

    print(f"Performing hybrid search for: '{sample_query}'")
    results = hybrid_search(sample_query)

    # Load metadata from parsed.json
    metadata_path = project_root / "data" / "parsed.json"
    metadata = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for item in json.load(f):
            metadata[item['id']] = {'title': item['title'], 'url': item['url']}

    print("\nTop search results:")
    for doc_id, _ in results:
        doc_info = metadata.get(doc_id)
        if doc_info:
            print(f"  - Title: {doc_info['title']}\n    Link: {doc_info['url']}")
        else:
            print(f"  - Metadata not found for ID: {doc_id}")