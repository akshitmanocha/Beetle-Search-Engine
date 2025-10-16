import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np

# Disable tokenizers parallelism to avoid multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Model and Tokenizer Loading ---
# Proper device selection for macOS, CUDA, and CPU
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

MODEL_ID = "naver/splade-cocondenser-ensembledistil"
tokenizer = None
model = None

def _get_model_and_tokenizer():
    """Lazy load model and tokenizer on first use."""
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval()
    return tokenizer, model

def _generate_splade_vector(text: str) -> dict[int, float]:
    """Generates a sparse SPLADE vector for a given text."""
    tokenizer, model = _get_model_and_tokenizer()
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        logits = model(**tokens).logits

    # Aggregation step (max pooling over sequence length)
    vec, _ = torch.max(torch.log(1 + torch.relu(logits)) * tokens.attention_mask.unsqueeze(-1), dim=1)
    
    # Filter out zero-weight terms and format
    cols = vec.nonzero().squeeze().cpu().tolist()
    weights = vec[0, cols].cpu().tolist()
    
    if not isinstance(cols, list):
        cols = [cols]
        weights = [weights]

    return dict(zip(cols, weights))

def search_splade(query: str, index_path: Path, doc_map_path: Path, top_k: int = 10):
    """Searches the SPLADE index for a given query."""
    print("Loading SPLADE index...")
    with open(index_path, 'r') as f:
        inverted_index = {int(k): v for k, v in json.load(f).items()}
    with open(doc_map_path, 'r') as f:
        doc_id_map = {int(k): v for k, v in json.load(f).items()}

    print(f"Searching for query: '{query}'")
    query_vec = _generate_splade_vector(query)
    scores = {}

    for term_id, q_weight in query_vec.items():
        if term_id in inverted_index:
            for doc_idx, d_weight in inverted_index[term_id]:
                scores[doc_idx] = scores.get(doc_idx, 0) + q_weight * d_weight

    # Sort documents by score
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    # Map integer indices back to original document IDs
    results = [
        {"id": doc_id_map[doc_idx], "score": score}
        for doc_idx, score in sorted_docs[:top_k]
    ]
    return results

def main():
    """Main function to run a sample search."""
    project_root = Path(__file__).parent.parent.parent
    index_path = project_root / "data" / "splade_index" / "inverted_index.json"
    doc_map_path = project_root / "data" / "splade_index" / "doc_map.json"

    # --- Perform a search ---
    print("\n" + "="*20)
    print(f"Using device: {DEVICE}")
    print("Performing a sample search...")
    sample_query = "transformer models for NLP"
    results = search_splade(sample_query, index_path, doc_map_path)

    print("\nTop search results:")
    for res in results:
        print(f"  - Document ID: {res['id']}, Score: {res['score']:.4f}")

if __name__ == "__main__":
    main()
