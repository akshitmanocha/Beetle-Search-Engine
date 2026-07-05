"""Unit tests for the BEIR dataset loader (task 8.2).

Builds a tiny on-disk dataset in BEIR's file layout (corpus.jsonl,
queries.jsonl, qrels/<split>.tsv) so the loader is exercised without any large
download (Requirement 12.4). Asserts uniform shape and the missing-doc-judgment
skip behavior (Requirements 6.4, 6.5).
"""

import json

import pytest

from eval.datasets import load_beir_dataset


def _write_fixture(root, name, split="test"):
    """Create a minimal BEIR-format dataset under ``root/name`` and return its path."""
    ds = root / name
    (ds / "qrels").mkdir(parents=True)

    corpus = [
        {"_id": "d1", "title": "Transformers", "text": "Attention is all you need."},
        {"_id": "d2", "title": "BM25", "text": "Probabilistic lexical retrieval."},
        {"_id": "d3", "title": "SPLADE", "text": "Learned sparse expansion."},
    ]
    with open(ds / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")

    queries = [
        {"_id": "q1", "text": "what is attention"},
        {"_id": "q2", "text": "lexical retrieval"},
    ]
    with open(ds / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    # qrels TSV: header, then query_id\tcorpus_id\tscore.
    # q2 references "d_missing" which is NOT in the corpus -> must be skipped.
    with open(ds / "qrels" / f"{split}.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.write("q1\td1\t2\n")
        f.write("q2\td2\t1\n")
        f.write("q2\td_missing\t1\n")

    return ds


def test_loader_uniform_shape(tmp_path):
    _write_fixture(tmp_path, "tiny")
    corpus, queries, qrels = load_beir_dataset("tiny", tmp_path)

    # Corpus: doc_id -> {title, text}
    assert set(corpus.keys()) == {"d1", "d2", "d3"}
    assert corpus["d1"] == {"title": "Transformers", "text": "Attention is all you need."}
    assert all({"title", "text"} <= set(v.keys()) for v in corpus.values())

    # Queries: query_id -> text
    assert queries == {"q1": "what is attention", "q2": "lexical retrieval"}

    # Qrels: query_id -> {doc_id: relevance}
    assert qrels["q1"] == {"d1": 2}


def test_loader_skips_missing_doc_judgments(tmp_path):
    """A qrel referencing a doc absent from the corpus is dropped, not crashed on."""
    _write_fixture(tmp_path, "tiny")
    _corpus, _queries, qrels = load_beir_dataset("tiny", tmp_path)

    # q2 had judgments for d2 (present) and d_missing (absent).
    assert qrels["q2"] == {"d2": 1}
    assert "d_missing" not in qrels["q2"]


def test_loader_relevance_values_are_ints(tmp_path):
    _write_fixture(tmp_path, "tiny")
    _corpus, _queries, qrels = load_beir_dataset("tiny", tmp_path)
    for judgments in qrels.values():
        for rel in judgments.values():
            assert isinstance(rel, int)


def test_loader_fails_fast_with_cache_path(tmp_path):
    """A missing dataset fails with a message naming the dataset and cache path."""
    with pytest.raises(RuntimeError) as exc_info:
        # Use a name BEIR's downloader will reject quickly; assert the message
        # names the dataset and the expected local cache path.
        load_beir_dataset("definitely_not_a_real_dataset_xyz", tmp_path)
    msg = str(exc_info.value)
    assert "definitely_not_a_real_dataset_xyz" in msg
    assert str(tmp_path / "definitely_not_a_real_dataset_xyz") in msg
