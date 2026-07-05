import pickle
from pathlib import Path
import numpy as np
import faiss
import json

def build_faiss_index(embedding_path: Path, index_path: Path, metric: str = "cosine"):
    """
    Builds and saves a FAISS index from a pickle file of embeddings.

    Args:
        embedding_path: Path to the pickle file containing the embeddings.
        index_path: Path to save the FAISS index.
        metric: ``"cosine"`` (default) L2-normalizes vectors and uses an
            inner-product index, so inner product equals cosine similarity —
            the correct metric for SentenceTransformer embeddings. ``"l2"``
            keeps the original Euclidean index and is retained only for the
            cosine-vs-L2 ablation (Requirement 3.4).
    """
    # Load embeddings
    with open(embedding_path, "rb") as f:
        embedding_map = pickle.load(f)

    doc_ids = list(embedding_map.keys())
    embeddings = np.array(list(embedding_map.values()), dtype='float32')

    if len(embeddings) == 0:
        print("No embeddings found to index.")
        return

    # Get the dimension of the embeddings
    d = embeddings.shape[1]

    # Build the base index according to the requested metric.
    if metric == "cosine":
        faiss.normalize_L2(embeddings)   # in-place -> unit vectors
        base_index = faiss.IndexFlatIP(d)  # inner product == cosine on unit vectors
    elif metric == "l2":
        base_index = faiss.IndexFlatL2(d)  # kept only for the cosine-vs-L2 ablation
    else:
        raise ValueError(f"Unknown metric '{metric}'; expected 'cosine' or 'l2'.")

    index = faiss.IndexIDMap(base_index)  # Mapping from internal id to position

    # Add vectors to the index
    index.add_with_ids(embeddings, np.array(range(len(doc_ids))))

    # Save the index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    # Save the document IDs mapping
    with open(index_path.with_suffix('.json'), 'w') as f:
        json.dump(doc_ids, f)

    print(f"✓ FAISS index ({metric}) built and saved to {index_path}")
    print(f"✓ Document ID mapping saved to {index_path.with_suffix('.json')}")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[2]
    embedding_path = project_root / "data" / "embeddings" / "embeddings.pkl"
    index_path = project_root / "data" / "faiss_index" / "faiss.index"

    build_faiss_index(embedding_path, index_path, metric="cosine")
