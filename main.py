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

def get_docs_from_jsonl(doc_ids_to_find: set, jsonl_path: Path) -> dict:
    """
    Efficiently retrieves specific documents from a JSONL file by their IDs.
    Returns a dictionary mapping doc_id to the full document object.
    """
    found_docs = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not doc_ids_to_find:
                break  # Stop if we've found all the docs we're looking for
            doc = json.loads(line)
            doc_id = doc.get('id')
            if doc_id in doc_ids_to_find:
                found_docs[doc_id] = doc
                doc_ids_to_find.remove(doc_id)
    return found_docs

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

    results = []
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

    # Efficiently get full document data from the JSONL file
    doc_ids_to_retrieve = {res['id'] for res in results}
    jsonl_path = project_root / "data" / "clean" / "blogs.jsonl"
    docs_map = get_docs_from_jsonl(doc_ids_to_retrieve, jsonl_path)

    # Reconstruct the list of documents in the order of the initial search results
    result_docs = [docs_map[res['id']] for res in results if res['id'] in docs_map]

    if reranker_enabled:
        print("Reranking results...")
        reranker = Reranker(reranker_params['model_name'], device=device)
        reranked_docs = reranker.rerank(query, result_docs)
        # The reranker already sorts the documents. We just need to add the original search score.
        # For simplicity, we will return the reranked list directly.
        # The reranker adds 'rerank_score' to each doc dictionary.
        return reranked_docs[:rerank_k]
    else:
        # Add the search score to the document object for consistent output
        for doc in result_docs:
            # Find the original score from the initial search results
            original_res = next((r for r in results if r['id'] == doc['id']), None)
            doc['search_score'] = original_res['score'] if original_res else 0
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

    print(f"\n--- Top {len(results)} results for '{args.query}' ---")
    if reranker_enabled:
        print("(Reranked)")
        for i, doc in enumerate(results):
            print(f"{i+1}. Title: {doc['title']}")
            print(f"   Link: {doc['url']}")
            print(f"   Score: {doc['rerank_score']:.4f}")
            print("-" * 20)
    else:
        for i, doc in enumerate(results):
            print(f"{i+1}. Title: {doc['title']}")
            print(f"   Link: {doc['url']}")
            print(f"   Score: {doc.get('search_score', 'N/A'):.4f}")
            print("-" * 20)

if __name__ == '__main__':
    main()