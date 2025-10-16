import argparse
import json
from pathlib import Path
import yaml
import torch

from src.search.hybrid_search import hybrid_search
from src.search.search_bm25 import search_bm25
from src.search.search_faiss import search_faiss
from src.search.search_splade import search_splade
from src.models.reranker import Reranker


def search_and_rerank(query: str, top_k: int, rerank_k: int, search_method: str, reranker_enabled: bool):
    project_root = Path(__file__).resolve().parent

    with open(project_root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)
    reranker_params = params['models']['reranker']
    embedding_params = params['models']['embedding']

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Performing {search_method} search for: '{query}'")

    if search_method == "hybrid":
        results = hybrid_search(query, top_k=top_k)
    elif search_method == "bm25":
        index_path = project_root / "data" / "bm25_index"
        results = search_bm25(query, index_path, top_k=top_k)
    elif search_method == "faiss":
        index_path = project_root / "data" / "faiss_index" / "faiss.index"
        doc_id_map_path = index_path.with_suffix('.json')
        results = search_faiss(
            query,
            embedding_params['model_name'],
            device,
            index_path,
            doc_id_map_path,
            top_k=top_k,
        )
    elif search_method == "splade":
        index_path = project_root / "data" / "splade_index" / "inverted_index.json"
        doc_map_path = project_root / "data" / "splade_index" / "doc_map.json"
        results = search_splade(query, index_path, doc_map_path, top_k=top_k)

    metadata_path = project_root / "data" / "parsed.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = {item['id']: item for item in json.load(f)}

    result_docs = [metadata[doc_id] for doc_id, _ in results if doc_id in metadata]

    if reranker_enabled:
        print("Reranking results...")
        reranker = Reranker(reranker_params['model_name'], device=device)
        reranked_docs = reranker.rerank(query, result_docs)
        return reranked_docs[:rerank_k]
    else:
        return result_docs[:rerank_k]

def main():
    parser = argparse.ArgumentParser(description="Perform a search with reranking.")
    parser.add_argument("query", type=str, help="The search query.")
    args = parser.parse_args()

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)['search']
    
    top_k = params.get('top_k', 20)
    rerank_k = params.get('rerank_k', 5)
    search_method = params.get('method', 'hybrid')
    reranker_enabled = params.get('reranker', False)

    results = search_and_rerank(args.query, top_k, rerank_k, search_method, reranker_enabled)

    if reranker_enabled:
        print("\nTop search results (reranked):")
        for i, doc in enumerate(results):
            print(f"{i+1}. Title: {doc['title']}")
            print(f"   Link: {doc['url']}")
            print(f"   Score: {doc['rerank_score']:.4f}")
            print("-" * 20)
    else:
        print("\nTop search results:")
        for i, doc in enumerate(results):
            print(f"{i+1}. Title: {doc['title']}")
            print(f"   Link: {doc['url']}")
            print("-" * 20)

if __name__ == '__main__':
    main()
