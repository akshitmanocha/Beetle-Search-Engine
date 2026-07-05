"""DVC stage: build the on-disk SPLADE inverted index from the clean corpus.

The SPLADE encoding logic lives once in ``src.serving.registry``
(``load_splade`` + ``generate_splade_vector``); this stage just reuses it and
persists the inverted index to disk. All heavy work happens inside ``main()``,
not at import time.
"""

import json
from pathlib import Path

from tqdm import tqdm


def build_splade_index(documents: list, index_path: Path, doc_map_path: Path):
    """Encode ``documents`` into a SPLADE inverted index and save it."""
    if index_path.exists():
        print(f"Index already exists at {index_path}. Skipping build.")
        return

    from src.config import CONFIG
    from src.serving.registry import doc_text, generate_splade_vector, load_splade

    model, tokenizer = load_splade(CONFIG)
    device = CONFIG.device()

    print(f"Building SPLADE inverted index for {len(documents)} documents...")
    inverted_index: dict = {}
    doc_id_map: dict = {}

    for i, doc in enumerate(tqdm(documents)):
        if not doc.get("id"):
            continue
        doc_id_map[i] = doc["id"]
        text = doc_text(doc)
        for term_id, weight in generate_splade_vector(text, model, tokenizer, device).items():
            inverted_index.setdefault(term_id, []).append((i, weight))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(inverted_index, f)
    with open(doc_map_path, "w") as f:
        json.dump(doc_id_map, f)
    print("✓ Index build complete.")


def main():
    root = Path(__file__).resolve().parents[2]
    with open(root / "data" / "clean" / "blogs.json") as f:
        documents = json.load(f)
    build_splade_index(
        documents,
        root / "data" / "splade_index" / "inverted_index.json",
        root / "data" / "splade_index" / "doc_map.json",
    )


if __name__ == "__main__":
    main()
