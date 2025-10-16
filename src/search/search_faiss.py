import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import faiss
import json
import torch
from src.models.embed import generate_embeddings
import yaml

def search_faiss(query_str: str, model_name: str, device: str, index_path: Path, doc_id_map_path: Path, top_k: int = 10):
    """
    Searches the FAISS index.

    Args:
        query_str: The query string.
        model_name: The name of the SentenceTransformer model to use.
        device: The device to run the model on ('cpu', 'cuda', or 'mps').
        index_path: Path to the FAISS index.
        doc_id_map_path: Path to the document ID mapping file.
        top_k: The number of top results to return.

    Returns:
        A list of tuples, where each tuple contains the document ID and the score.
    """
    # Load the FAISS index
    index = faiss.read_index(str(index_path))

    # Load the document ID mapping
    with open(doc_id_map_path, 'r') as f:
        doc_ids = json.load(f)

    # Generate the embedding for the query
    query_embedding_map = generate_embeddings(
        documents=[{"id": "query", "body_text": query_str}],
        model_name=model_name,
        batch_size=1,
        device=device,
    )
    query_embedding = query_embedding_map["query"]

    query_embedding = np.array([query_embedding], dtype='float32')

    # Perform the search
    distances, indices = index.search(query_embedding, top_k)

    # Extract results
    results = []
    for i in range(len(indices[0])):
        doc_id_index = indices[0][i]
        if doc_id_index != -1:
            results.append((doc_ids[doc_id_index], distances[0][i]))

    return results


if __name__ == '__main__':
    index_path = project_root / "data" / "faiss_index" / "faiss.index"
    doc_id_map_path = index_path.with_suffix('.json')

    print("\nPerforming a sample search...")
    with open(project_root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)['models']['embedding']

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    sample_query = "transformer models"
    results = search_faiss(
        sample_query,
        params['model_name'],
        device,
        index_path,
        doc_id_map_path
    )
    print(f"Top {len(results)} results for '{sample_query}':")
    for doc_id, score in results:
        print(f"  - Document ID: {doc_id}, Score: {score:.4f}")
