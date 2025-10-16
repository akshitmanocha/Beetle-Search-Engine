import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

# --- Model and Tokenizer Loading ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

def _generate_splade_vector(text: str) -> dict[int, float]:
    """Generates a sparse SPLADE vector for a given text."""
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

def build_splade_index(documents: list[dict], index_path: Path, doc_map_path: Path):
    """Builds an inverted index from documents and saves it."""
    if index_path.exists():
        print(f"Index already exists at {index_path}. Skipping build.")
        return

    print(f"Building SPLADE inverted index for {len(documents)} documents...")
    inverted_index = {}
    doc_id_map = {}

    for i, doc in enumerate(tqdm(documents)):
        doc_id = doc.get("id")
        if not doc_id:
            continue
        
        doc_id_map[i] = doc_id
        text_to_embed = doc.get("title", "") + " " + doc.get("body_text", "")
        sparse_vec = _generate_splade_vector(text_to_embed)

        for term_id, weight in sparse_vec.items():
            if term_id not in inverted_index:
                inverted_index[term_id] = []
            inverted_index[term_id].append((i, weight))

    print(f"Saving inverted index to {index_path}...")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, 'w') as f:
        json.dump(inverted_index, f)
        
    with open(doc_map_path, 'w') as f:
        json.dump(doc_id_map, f)

    print("✓ Index build complete.")

def main():
    """Main function to build the index."""
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "parsed.json"
    index_path = project_root / "data" / "splade_index" / "inverted_index.json"
    doc_map_path = project_root / "data" / "splade_index" / "doc_map.json"

    # Load documents
    with open(data_path, "r") as f:
        documents = json.load(f)

    # Build the index
    build_splade_index(documents, index_path, doc_map_path)

if __name__ == "__main__":
    main()
