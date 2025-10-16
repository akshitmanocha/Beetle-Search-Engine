import pickle
from pathlib import Path
import numpy as np
import faiss
import json

def build_faiss_index(embedding_path: Path, index_path: Path):
    """
    Builds and saves a FAISS index from a pickle file of embeddings.

    Args:
        embedding_path: Path to the pickle file containing the embeddings.
        index_path: Path to save the FAISS index.
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

    # Build the Faiss index
    index = faiss.IndexFlatL2(d)  # Using L2 distance
    index = faiss.IndexIDMap(index) # Mapping from index to document ID

    # Add vectors to the index
    index.add_with_ids(embeddings, np.array(range(len(doc_ids))))

    # Save the index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    # Save the document IDs mapping
    with open(index_path.with_suffix('.json'), 'w') as f:
        json.dump(doc_ids, f)

    print(f"✓ FAISS index built and saved to {index_path}")
    print(f"✓ Document ID mapping saved to {index_path.with_suffix('.json')}")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[2]
    embedding_path = project_root / "data" / "embeddings" / "embeddings.pkl"
    index_path = project_root / "data" / "faiss_index" / "faiss.index"
    
    build_faiss_index(embedding_path, index_path)
