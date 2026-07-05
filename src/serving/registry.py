"""Model/index registry: load every heavy artifact exactly once.

This module is the single place that owns the expensive objects — the embedding
model, the SPLADE model + tokenizer, the cross-encoder reranker, the retrieval
indexes (BM25, FAISS, SPLADE), and the corpus as an in-memory ``id -> doc`` dict.
Loading them once (at FastAPI startup, or once per dataset in the eval harness)
fixes the per-query model reloads and corpus rescans (Requirements 2.1-2.4).

Two builders share the same model loaders and index builders:

  * ``build_registry(config)`` — serving: load on-disk artifacts produced by the
    DVC pipeline (BM25 dir, FAISS index, SPLADE inverted index, corpus json).
  * ``build_eval_registry(corpus, config, ...)`` — evaluation/demo: build the
    indexes in memory from a corpus dict (e.g. a BEIR dataset). Used by the
    ablation runner and the deployed demo.

Models are cached at module level so that building several eval registries (one
per dataset) reuses the already-loaded models rather than reloading them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache (load once, reuse across registries).
# ---------------------------------------------------------------------------
_EMBED_MODEL = None
_SPLADE_MODEL = None
_SPLADE_TOKENIZER = None
_RERANKER = None


def load_embed_model(config):
    """Load the SentenceTransformer embedding model once and cache it."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer

        model_name = config.params["models"]["embedding"]["model_name"]
        device = config.device()
        logger.info("Loading embedding model '%s' on %s", model_name, device)
        _EMBED_MODEL = SentenceTransformer(
            model_name, device=device, trust_remote_code=True
        )
    return _EMBED_MODEL


def load_splade(config):
    """Load the SPLADE model + tokenizer once and cache them."""
    global _SPLADE_MODEL, _SPLADE_TOKENIZER
    if _SPLADE_MODEL is None or _SPLADE_TOKENIZER is None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        model_id = config.params["models"].get("splade", {}).get(
            "model_name", "naver/splade-cocondenser-ensembledistil"
        )
        device = config.device()
        logger.info("Loading SPLADE model '%s' on %s", model_id, device)
        _SPLADE_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _SPLADE_MODEL = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
        _SPLADE_MODEL.eval()
    return _SPLADE_MODEL, _SPLADE_TOKENIZER


def load_reranker(config):
    """Load the cross-encoder reranker once and cache it."""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        model_name = config.params["models"]["reranker"]["model_name"]
        device = config.device()
        logger.info("Loading reranker '%s' on %s", model_name, device)
        _RERANKER = CrossEncoder(model_name, device=device, trust_remote_code=True)
    return _RERANKER


# ---------------------------------------------------------------------------
# SPLADE sparse-vector generation (single source of truth).
# ---------------------------------------------------------------------------

def generate_splade_vector(text: str, model, tokenizer, device: str,
                           max_length: int = 512) -> Dict[int, float]:
    """Encode ``text`` into a sparse ``{term_id: weight}`` SPLADE vector.

    Single-text convenience wrapper over :func:`generate_splade_vectors_batch`.
    """
    return generate_splade_vectors_batch([text], model, tokenizer, device, max_length)[0]


def generate_splade_vectors_batch(texts, model, tokenizer, device: str,
                                  max_length: int = 512, batch_size: int = 32):
    """Encode a list of texts into sparse ``{term_id: weight}`` SPLADE vectors.

    Batched for MPS/GPU throughput. Each vector is the max-pooled
    ``log(1 + relu(logits))`` over the sequence, masked by attention.

    IMPORTANT (MPS correctness): inputs are padded to a fixed ``max_length``
    (``padding="max_length"``), NOT just to the batch max (``padding=True``).
    On Apple ``mps``, ``torch.max(dim=1)`` over a *short, lightly-padded*
    sequence returns garbage (e.g. the query 'Foods for Glaucoma' expanded to
    'is/,/g' instead of 'foods/glaucoma/diet'), which silently halved NFCorpus
    SPLADE nDCG (0.17 vs the published 0.35). Forcing a fixed pad length makes
    MPS match CPU exactly. Terms are extracted per-row (``row[nonzero]``).
    """
    import torch

    out: List[Dict[int, float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        tokens = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=max_length, padding="max_length",
        ).to(device)
        with torch.no_grad():
            logits = model(**tokens).logits

        # Max-pool log(1 + relu(logits)) over the sequence, masked by attention.
        vecs, _ = torch.max(
            torch.log(1 + torch.relu(logits)) * tokens.attention_mask.unsqueeze(-1),
            dim=1,
        )  # [B, vocab]
        for r in range(vecs.shape[0]):
            row = vecs[r]
            nz = row.nonzero().squeeze(-1)
            cols = nz.cpu().tolist()
            weights = row[nz].cpu().tolist()
            out.append(dict(zip(cols, weights)))
    return out


# ---------------------------------------------------------------------------
# Registry data structure.
# ---------------------------------------------------------------------------

@dataclass
class Registry:
    """Holds every loaded model/index plus the in-memory corpus."""

    config: Any
    corpus: Dict[str, Dict[str, str]] = field(default_factory=dict)

    bm25_index: Any = None
    faiss_index: Any = None
    faiss_doc_ids: List[str] = field(default_factory=list)
    faiss_metric: str = "cosine"
    splade_index: Optional[Dict[int, list]] = None
    splade_doc_map: Optional[Dict[int, str]] = None

    embed_model: Any = None
    splade_model: Any = None
    splade_tokenizer: Any = None
    reranker: Any = None

    errors: List[str] = field(default_factory=list)

    def device(self) -> str:
        return self.config.device()

    def is_ready(self) -> bool:
        """True when no load errors occurred and the core artifacts are present."""
        return not self.errors and self.corpus is not None and len(self.corpus) > 0

    def missing_artifacts(self) -> List[str]:
        """Names of artifacts that failed to load (for the /readyz error body)."""
        missing = []
        if not self.corpus:
            missing.append("corpus")
        if self.bm25_index is None:
            missing.append("bm25_index")
        if self.faiss_index is None:
            missing.append("faiss_index")
        if self.splade_index is None:
            missing.append("splade_index")
        if self.embed_model is None:
            missing.append("embed_model")
        if self.reranker is None:
            missing.append("reranker")
        return missing


# ---------------------------------------------------------------------------
# Index builders (shared by eval and demo).
# ---------------------------------------------------------------------------

def doc_text(doc: Dict[str, str]) -> str:
    """The text used for indexing/reranking: title + body, robust to schema.

    The single source of truth for turning a corpus document into the string fed
    to every indexer/encoder (BM25, FAISS, SPLADE). Accepts either the BEIR
    ``text`` key or the blog-pipeline ``body_text`` key.
    """
    title = doc.get("title", "") or ""
    body = doc.get("text", doc.get("body_text", "")) or ""
    return f"{title} {body}".strip()


def build_bm25_index(corpus: Dict[str, Dict[str, str]], index_dir: Path):
    """Build (or open) a Whoosh BM25 index for ``corpus`` under ``index_dir``."""
    from whoosh import index as whoosh_index
    from whoosh.analysis import StemmingAnalyzer
    from whoosh.fields import ID, TEXT, Schema

    schema = Schema(
        id=ID(stored=True, unique=True),
        title=TEXT(stored=True, analyzer=StemmingAnalyzer(), field_boost=2.0),
        body_text=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    )

    index_dir.mkdir(parents=True, exist_ok=True)
    ix = whoosh_index.create_in(str(index_dir), schema)
    writer = ix.writer()
    for doc_id, doc in corpus.items():
        writer.add_document(
            id=str(doc_id),
            title=doc.get("title", "") or "",
            body_text=doc.get("text", doc.get("body_text", "")) or "",
        )
    writer.commit()
    return ix


def build_faiss_index_in_memory(
    corpus: Dict[str, Dict[str, str]], embed_model, metric: str = "cosine",
    batch_size: int = 32,
) -> Tuple[Any, List[str]]:
    """Embed the corpus and build an in-memory FAISS index. Returns (index, doc_ids)."""
    import faiss
    import numpy as np

    doc_ids = list(corpus.keys())
    texts = [doc_text(corpus[d]) for d in doc_ids]

    # embeddinggemma is prompt-based: use encode_document when available so the
    # corpus side gets the document prompt (see beetle-embedder-facts memory).
    if hasattr(embed_model, "encode_document"):
        vectors = embed_model.encode_document(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )
    else:
        vectors = embed_model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )
    vectors = np.asarray(vectors, dtype="float32")

    d = vectors.shape[1]
    if metric == "cosine":
        faiss.normalize_L2(vectors)
        base = faiss.IndexFlatIP(d)
    elif metric == "l2":
        base = faiss.IndexFlatL2(d)
    else:
        raise ValueError(f"Unknown metric '{metric}'.")

    index = faiss.IndexIDMap(base)
    index.add_with_ids(vectors, np.arange(len(doc_ids)))
    return index, doc_ids


def build_splade_index_in_memory(
    corpus: Dict[str, Dict[str, str]], splade_model, splade_tokenizer, device: str,
) -> Tuple[Dict[int, list], Dict[int, str]]:
    """Build an in-memory SPLADE inverted index. Returns (inverted_index, doc_map)."""
    inverted_index: Dict[int, list] = {}
    doc_map: Dict[int, str] = {}

    doc_ids = list(corpus.keys())
    texts = [doc_text(corpus[d]) for d in doc_ids]
    vectors = generate_splade_vectors_batch(texts, splade_model, splade_tokenizer, device)
    for i, (doc_id, sparse) in enumerate(zip(doc_ids, vectors)):
        doc_map[i] = doc_id
        for term_id, weight in sparse.items():
            inverted_index.setdefault(term_id, []).append((i, weight))

    return inverted_index, doc_map


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------

def build_eval_registry(
    corpus: Dict[str, Dict[str, str]],
    config,
    index_dir: Path,
    metric: str = "cosine",
    with_splade: bool = True,
    with_reranker: bool = True,
) -> Registry:
    """Build a fully in-memory registry from a corpus dict (eval/demo path).

    Args:
        corpus: ``{doc_id: {"title", "text"}}``.
        config: the ``Config`` singleton.
        index_dir: directory for the on-disk Whoosh index (other indexes are
            held in memory).
        metric: FAISS metric, ``"cosine"`` (default) or ``"l2"`` (ablation).
        with_splade: build the SPLADE index (slowest step; skip to save time).
        with_reranker: load the cross-encoder reranker.
    """
    reg = Registry(config=config, corpus=corpus, faiss_metric=metric)
    device = config.device()

    # Models (cached across datasets).
    reg.embed_model = load_embed_model(config)
    if with_splade:
        reg.splade_model, reg.splade_tokenizer = load_splade(config)
    if with_reranker:
        reg.reranker = load_reranker(config)

    # Indexes.
    reg.bm25_index = build_bm25_index(corpus, Path(index_dir) / "bm25")
    reg.faiss_index, reg.faiss_doc_ids = build_faiss_index_in_memory(
        corpus, reg.embed_model, metric=metric
    )
    if with_splade:
        reg.splade_index, reg.splade_doc_map = build_splade_index_in_memory(
            corpus, reg.splade_model, reg.splade_tokenizer, device
        )

    return reg


def build_registry(config) -> Registry:
    """Build a serving registry from on-disk artifacts (DVC pipeline outputs).

    Loads each artifact exactly once, recording a load error rather than raising
    so the server can report not-ready (Requirement 2.4) instead of crashing.
    """
    reg = Registry(config=config)

    # Corpus (id -> doc), held in memory to kill per-query file scans.
    try:
        corpus_path = config.path("data", "clean", "blogs.json")
        with open(corpus_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        reg.corpus = {
            doc["id"]: {
                "title": doc.get("title", "") or "",
                "text": doc.get("body_text", doc.get("text", "")) or "",
                "url": doc.get("url", ""),
            }
            for doc in docs
            if doc.get("id")
        }
    except Exception as exc:
        reg.errors.append(f"corpus: {exc}")

    # BM25 (Whoosh).
    try:
        from whoosh.index import open_dir

        reg.bm25_index = open_dir(str(config.path("data", "bm25_index")))
    except Exception as exc:
        reg.errors.append(f"bm25_index: {exc}")

    # FAISS index + doc-id map.
    try:
        import faiss

        faiss_path = config.path("data", "faiss_index", "faiss.index")
        reg.faiss_index = faiss.read_index(str(faiss_path))
        with open(faiss_path.with_suffix(".json"), "r", encoding="utf-8") as f:
            reg.faiss_doc_ids = json.load(f)
    except Exception as exc:
        reg.errors.append(f"faiss_index: {exc}")

    # SPLADE inverted index + doc map.
    try:
        splade_dir = config.path("data", "splade_index")
        with open(splade_dir / "inverted_index.json", "r", encoding="utf-8") as f:
            reg.splade_index = {int(k): v for k, v in json.load(f).items()}
        with open(splade_dir / "doc_map.json", "r", encoding="utf-8") as f:
            reg.splade_doc_map = {int(k): v for k, v in json.load(f).items()}
    except Exception as exc:
        reg.errors.append(f"splade_index: {exc}")

    # Models.
    try:
        reg.embed_model = load_embed_model(config)
    except Exception as exc:
        reg.errors.append(f"embed_model: {exc}")
    try:
        reg.splade_model, reg.splade_tokenizer = load_splade(config)
    except Exception as exc:
        reg.errors.append(f"splade_model: {exc}")
    try:
        reg.reranker = load_reranker(config)
    except Exception as exc:
        reg.errors.append(f"reranker: {exc}")

    if reg.errors:
        logger.error("Registry build encountered errors: %s", reg.errors)

    return reg
