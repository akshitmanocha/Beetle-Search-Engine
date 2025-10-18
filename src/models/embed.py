#!/usr/bin/env python3
"""
Generate sentence embeddings for blog posts using a SentenceTransformer model.
"""

import json
import os
import pickle
from pathlib import Path

import torch
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_data(data_path: Path) -> list:
    """Load parsed blog data from a JSON file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    with open(data_path, "r") as f:
        return json.load(f)


def generate_embeddings(
    documents: list[dict],
    model_name: str,
    batch_size: int,
    device: str,
) -> dict[str, list[float]]:
    """
    Generate embeddings for a list of documents.

    Args:
        documents: A list of dictionaries, where each dict represents a blog post.
        model_name: The name of the SentenceTransformer model to use.
        batch_size: The batch size for encoding.
        device: The device to run the model on ('cpu', 'cuda', or 'mps').

    Returns:
        A dictionary mapping document IDs to their embeddings.
    """
    # Initialize the SentenceTransformer model
    model = SentenceTransformer(model_name, device=device,trust_remote_code=True)

    # Prepare texts and corresponding IDs
    texts = [
        doc.get("body_text", "") for doc in documents if doc.get("body_text")
    ]
    doc_ids = [
        doc["id"] for doc in documents if doc.get("body_text")
    ]

    print(f"Generating embeddings for {len(texts)} documents with non-empty body text...")

    # Generate embeddings in batches
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        
    )

    # Create a mapping from doc_id to embedding
    embedding_map = {doc_id: emb.tolist() for doc_id, emb in zip(doc_ids, embeddings)}
    return embedding_map


def save_embeddings(embedding_map: dict, output_path: Path):
    """Save the embedding map to a pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embedding_map, f)
    print(f"✓ Saved embeddings for {len(embedding_map)} documents to: {output_path}")


def main():
    """Main function to run the embedding generation process."""

    project_root = Path(__file__).parent.parent.parent

    # Load parameters
    with open(project_root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)['models']['embedding']

    model_name = params['model_name']
    batch_size = params['batch_size']

    # Proper device selection for macOS, CUDA, and CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Define paths
    data_path = project_root / "data" / "clean" / "blogs.json"
    output_path = project_root / "data" / "embeddings" / "embeddings.pkl"

    print(f"Using device: {device}")

    # Run the process
    documents = load_data(data_path)
    embedding_map = generate_embeddings(documents, model_name, batch_size, device)
    save_embeddings(embedding_map, output_path)


if __name__ == "__main__":
    main()
