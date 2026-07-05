#!/usr/bin/env python3
"""Prepare the demo corpus + indexes from a BEIR dataset (task 13.1 support).

The live demo searches a small BEIR corpus (default: SciFact) instead of the
deferred blog corpus. This script downloads the dataset and writes exactly the
on-disk artifacts that ``src.serving.registry.build_registry`` loads:

  data/clean/blogs.json                      (list of {id, title, body_text, url})
  data/bm25_index/                           (Whoosh index dir)
  data/faiss_index/faiss.index   + .json     (cosine FAISS + doc-id list)
  data/splade_index/inverted_index.json + doc_map.json

Run once at image-build time (needs an HF token for the gated embedder):

    HF_TOKEN=... python scripts/prepare_demo_index.py --dataset scifact

After this, ``build_registry`` works unchanged and the container starts fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_demo_index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the demo corpus + indexes from a BEIR dataset.")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name (default: scifact)")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Cap the corpus size for a lighter demo (0 = all docs)")
    args = parser.parse_args()

    import faiss
    import numpy as np

    from src.config import CONFIG
    from eval.datasets import load_beir_dataset
    from src.serving.registry import (
        load_embed_model, load_splade, build_bm25_index,
        build_splade_index_in_memory, doc_text,
    )

    data_dir = CONFIG.path("data", "beir")
    logger.info("Loading BEIR dataset '%s'...", args.dataset)
    corpus, _queries, _qrels = load_beir_dataset(args.dataset, data_dir)

    if args.max_docs and len(corpus) > args.max_docs:
        kept = dict(list(corpus.items())[: args.max_docs])
        logger.info("Capping corpus from %d to %d docs", len(corpus), len(kept))
        corpus = kept

    # 1. blogs.json in the shape build_registry expects.
    clean_dir = CONFIG.path("data", "clean")
    clean_dir.mkdir(parents=True, exist_ok=True)
    blogs = [
        {
            "id": doc_id,
            "title": doc.get("title", ""),
            "body_text": doc.get("text", ""),
            "url": f"https://example.org/{args.dataset}/{doc_id}",
        }
        for doc_id, doc in corpus.items()
    ]
    with open(clean_dir / "blogs.json", "w", encoding="utf-8") as f:
        json.dump(blogs, f)
    logger.info("Wrote %d docs to %s", len(blogs), clean_dir / "blogs.json")

    # 2. BM25 (Whoosh) index.
    bm25_dir = CONFIG.path("data", "bm25_index")
    if bm25_dir.exists():
        import shutil
        shutil.rmtree(bm25_dir)
    build_bm25_index(corpus, bm25_dir)
    logger.info("Built BM25 index at %s", bm25_dir)

    # 3. FAISS cosine index (normalize + IndexFlatIP + IDMap) using encode_document.
    embed_model = load_embed_model(CONFIG)
    doc_ids = list(corpus.keys())
    texts = [doc_text(corpus[d]) for d in doc_ids]
    if hasattr(embed_model, "encode_document"):
        vectors = embed_model.encode_document(texts, convert_to_numpy=True, show_progress_bar=True)
    else:
        vectors = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(vectors, np.arange(len(doc_ids)))

    faiss_dir = CONFIG.path("data", "faiss_index")
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = faiss_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    with open(faiss_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    logger.info("Built FAISS cosine index (%d x %d) at %s", len(doc_ids), vectors.shape[1], faiss_path)

    # 4. SPLADE inverted index.
    splade_model, splade_tokenizer = load_splade(CONFIG)
    inverted_index, doc_map = build_splade_index_in_memory(
        corpus, splade_model, splade_tokenizer, CONFIG.device()
    )
    splade_dir = CONFIG.path("data", "splade_index")
    splade_dir.mkdir(parents=True, exist_ok=True)
    # Keys must be JSON-serializable; term ids are ints -> str on disk, parsed back in build_registry.
    with open(splade_dir / "inverted_index.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in inverted_index.items()}, f)
    with open(splade_dir / "doc_map.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in doc_map.items()}, f)
    logger.info("Built SPLADE inverted index at %s", splade_dir)

    logger.info("Demo index preparation complete for '%s'.", args.dataset)


if __name__ == "__main__":
    main()
