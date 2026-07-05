"""BEIR dataset loading into a uniform ``(corpus, queries, qrels)`` shape.

Defaults to the two small datasets reported for the sprint — SciFact and
NFCorpus — each of which downloads in minutes and indexes quickly on an M4 Pro.
The loader is dataset-name-agnostic, so any other BEIR dataset (e.g. ArguAna,
which is supported but excluded from the reported runs because its 1,406-query
reranking pass is impractically slow on a single-laptop MPS setup) works too.

Design contract (Requirement 6):
  * uniform shape: ``corpus: {doc_id: {"title", "text"}}``,
    ``queries: {query_id: text}``, ``qrels: {query_id: {doc_id: relevance}}``;
  * fail fast with a message naming the dataset and its expected local cache
    path when it cannot be downloaded or located;
  * skip any qrel judgment that references a document absent from the corpus,
    with a warning, so it does not count toward ideal scoring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# The sprint's reported benchmark datasets (small, fast, widely cited).
DEFAULT_DATASETS = ("scifact", "nfcorpus")

# BEIR's public download host. Datasets are distributed as zip archives named
# ``<dataset>.zip`` under this prefix.
_BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

Corpus = Dict[str, Dict[str, str]]
Queries = Dict[str, str]
Qrels = Dict[str, Dict[str, int]]


def load_beir_dataset(
    name: str,
    data_dir: Path,
    split: str = "test",
) -> Tuple[Corpus, Queries, Qrels]:
    """Load a BEIR dataset into the uniform ``(corpus, queries, qrels)`` shape.

    Args:
        name: dataset name (e.g. ``"scifact"``, ``"nfcorpus"``, ``"arguana"``).
        data_dir: directory under which datasets are cached/extracted.
        split: relevance-judgment split to load (BEIR convention: ``"test"``).

    Returns:
        ``(corpus, queries, qrels)``. Judgments referencing a doc id absent from
        the corpus are dropped (with a warning) and not counted toward scoring.

    Raises:
        RuntimeError: if the dataset cannot be downloaded or located, with a
            message naming the dataset and its expected local cache path.
    """
    data_dir = Path(data_dir)
    expected_path = data_dir / name

    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as exc:  # pragma: no cover - import-time environment issue
        raise RuntimeError(
            "The 'beir' package is required to load BEIR datasets. "
            "Install it via `pip install -r requirements.txt`."
        ) from exc

    # Download + unzip if the dataset is not already cached locally.
    if not expected_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        url = _BEIR_URL.format(name=name)
        try:
            logger.info("Downloading BEIR dataset '%s' from %s", name, url)
            util.download_and_unzip(url, str(data_dir))
        except Exception as exc:
            raise RuntimeError(
                f"Could not download or locate BEIR dataset '{name}'. "
                f"Expected a local cache at: {expected_path}. "
                f"Download URL was: {url}. Underlying error: {exc}"
            ) from exc

    if not expected_path.exists():
        raise RuntimeError(
            f"BEIR dataset '{name}' is still not present after download. "
            f"Expected a local cache at: {expected_path}."
        )

    raw_corpus, raw_queries, raw_qrels = GenericDataLoader(
        data_folder=str(expected_path)
    ).load(split=split)

    return _normalize(name, raw_corpus, raw_queries, raw_qrels)


def _normalize(
    name: str,
    raw_corpus: Dict[str, Dict[str, str]],
    raw_queries: Dict[str, str],
    raw_qrels: Dict[str, Dict[str, int]],
) -> Tuple[Corpus, Queries, Qrels]:
    """Coerce BEIR's loader output into Beetle's uniform shape and drop dangling qrels."""
    corpus: Corpus = {
        doc_id: {
            "title": doc.get("title", "") or "",
            "text": doc.get("text", "") or "",
        }
        for doc_id, doc in raw_corpus.items()
    }
    queries: Queries = dict(raw_queries)

    qrels: Qrels = {}
    dropped = 0
    for query_id, judgments in raw_qrels.items():
        kept: Dict[str, int] = {}
        for doc_id, relevance in judgments.items():
            if doc_id in corpus:
                kept[doc_id] = int(relevance)
            else:
                dropped += 1
        # Keep the query even if all of its judgments were dropped, so callers
        # can still iterate the full query set; scoring treats it as 0-relevant.
        qrels[query_id] = kept

    if dropped:
        logger.warning(
            "Dataset '%s': skipped %d qrel judgment(s) referencing docs absent "
            "from the corpus (excluded from ideal scoring).",
            name,
            dropped,
        )

    return corpus, queries, qrels
