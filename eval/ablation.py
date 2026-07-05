"""Ablation runner: evaluate retrieval systems across datasets and emit tables.

Implements the research deliverable's core (Requirement 8 / design Algorithm 4):
builds per-dataset eval indexes once, evaluates BM25-only / dense-only /
SPLADE-only / hybrid / hybrid+rerank, records a :class:`MetricRow` per
(dataset, system), treats empty retriever results as zero-metric rankings
without crashing, and writes ``eval/results/<dataset>_metrics.csv``.

Also provides the fusion-weight sweep (task 9.2) and the cosine-vs-L2 comparison
(task 9.3).

Determinism (Property 8): models run in eval mode and indexes are built once per
dataset; rankings are a pure function of the (corpus, query) pair, so re-running
yields identical rankings and metrics. ``set_seed`` pins torch/numpy RNG for
defense in depth.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval.metrics import mean, mrr_at_k, ndcg_at_k, recall_at_k

logger = logging.getLogger(__name__)

RERANK_K = 50          # candidate depth fed to the reranker
RETRIEVAL_DEPTH = 100  # top_k each retriever is queried at (for Recall@100)


def set_seed(seed: int = 42) -> None:
    """Pin RNG for reproducibility (Property 8)."""
    import numpy as np

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:
        pass


@dataclass
class SystemSpec:
    """One row of the ablation grid."""

    name: str  # "bm25" | "dense" | "splade" | "hybrid" | "hybrid+rerank"
    weights: Optional[Tuple[float, float, float]] = None
    rerank: bool = False


@dataclass
class MetricRow:
    system: str
    dataset: str
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_100: float
    mean_latency_ms: float
    reranker_calls: int = 0


DEFAULT_SYSTEMS: List[SystemSpec] = [
    SystemSpec("bm25", weights=None, rerank=False),
    SystemSpec("dense", weights=None, rerank=False),
    SystemSpec("splade", weights=None, rerank=False),
    SystemSpec("hybrid", weights=(1.0, 1.0, 1.0), rerank=False),
    SystemSpec("hybrid+rerank", weights=(1.0, 1.0, 1.0), rerank=True),
]


def _run_query(spec: SystemSpec, query: str, registry) -> Tuple[List[str], int]:
    """Run one query through one system. Returns (ranked_doc_ids, reranker_calls)."""
    from src.search.search_bm25 import search_bm25_registry
    from src.search.search_faiss import search_faiss_registry
    from src.search.search_splade import search_splade_registry
    from src.search.hybrid_search import hybrid_search
    from src.models.reranker import rerank

    reranker_calls = 0
    if spec.name == "bm25":
        ranked = search_bm25_registry(query, registry, top_k=RETRIEVAL_DEPTH)
    elif spec.name == "dense":
        ranked = search_faiss_registry(query, registry, top_k=RETRIEVAL_DEPTH)
    elif spec.name == "splade":
        ranked = search_splade_registry(query, registry, top_k=RETRIEVAL_DEPTH)
    else:  # hybrid / hybrid+rerank
        weights = spec.weights or (1.0, 1.0, 1.0)
        ranked = hybrid_search(query, registry, top_k=RETRIEVAL_DEPTH, weights=weights)
        if spec.rerank and ranked:
            reranked = rerank(query, ranked[:RERANK_K], registry)
            reranker_calls = len(reranked)
            # Reranked head, then the untouched tail (for Recall@100 coverage).
            tail = ranked[RERANK_K:]
            ranked = reranked + tail

    return [doc_id for doc_id, _ in ranked], reranker_calls


def evaluate_system(
    spec: SystemSpec,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    registry,
    dataset: str,
) -> MetricRow:
    """Evaluate one system over all queries of one dataset."""
    rankings: Dict[str, List[str]] = {}
    latencies: List[float] = []
    total_rerank_calls = 0

    for qid, qtext in queries.items():
        t0 = time.perf_counter()
        try:
            ranked_ids, rcalls = _run_query(spec, qtext, registry)
        except Exception as exc:
            logger.warning("Query %r failed for system %s: %s", qid, spec.name, exc)
            ranked_ids, rcalls = [], 0
        latencies.append((time.perf_counter() - t0) * 1000.0)
        rankings[qid] = ranked_ids
        total_rerank_calls += rcalls

    # Score over queries that have judgments (others would be all-zero anyway).
    judged = [q for q in queries if q in qrels and qrels[q]]
    row = MetricRow(
        system=spec.name,
        dataset=dataset,
        ndcg_at_10=mean(ndcg_at_k(rankings[q], qrels[q], 10) for q in judged),
        mrr_at_10=mean(mrr_at_k(rankings[q], qrels[q], 10) for q in judged),
        recall_at_100=mean(recall_at_k(rankings[q], qrels[q], 100) for q in judged),
        mean_latency_ms=mean(latencies),
        reranker_calls=total_rerank_calls,
    )
    return row


def write_metric_rows(rows: List[MetricRow], path: Path) -> None:
    """Write a list of MetricRow to a CSV at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system", "dataset", "ndcg_at_10", "mrr_at_10",
        "recall_at_100", "mean_latency_ms", "reranker_calls",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_ablation(
    datasets: List[str],
    systems: Optional[List[SystemSpec]] = None,
    out_dir: Path = Path("eval/results"),
    data_dir: Optional[Path] = None,
    seed: int = 42,
) -> List[MetricRow]:
    """Run the full ablation grid over ``datasets`` and write per-dataset CSVs.

    Builds the per-dataset eval registry once, evaluates every system against
    it, writes ``<out_dir>/<dataset>_metrics.csv`` per dataset, and returns all
    MetricRows.
    """
    from src.config import CONFIG
    from src.serving.registry import build_eval_registry
    from eval.datasets import load_beir_dataset

    set_seed(seed)
    systems = systems or DEFAULT_SYSTEMS
    out_dir = Path(out_dir)
    data_dir = Path(data_dir) if data_dir else CONFIG.path("data", "beir")

    all_rows: List[MetricRow] = []
    for ds_name in datasets:
        logger.info("Loading dataset %s", ds_name)
        corpus, queries, qrels = load_beir_dataset(ds_name, data_dir)

        index_dir = CONFIG.path("data", "eval_indexes", ds_name)
        need_splade = any(s.name in ("splade", "hybrid", "hybrid+rerank") for s in systems)
        need_reranker = any(s.rerank for s in systems)
        registry = build_eval_registry(
            corpus, CONFIG, index_dir, metric="cosine",
            with_splade=need_splade, with_reranker=need_reranker,
        )

        ds_rows = [evaluate_system(spec, queries, qrels, registry, ds_name) for spec in systems]
        write_metric_rows(ds_rows, out_dir / f"{ds_name}_metrics.csv")
        all_rows.extend(ds_rows)

    # Combined table across all datasets.
    if all_rows:
        write_metric_rows(all_rows, out_dir / "metrics.csv")
    return all_rows


def sweep_fusion_weights(
    dataset: str,
    grid: Optional[List[Tuple[float, float, float]]] = None,
    out_dir: Path = Path("eval/results"),
    data_dir: Optional[Path] = None,
    seed: int = 42,
) -> List[dict]:
    """Evaluate a grid of RRF weights on one dataset (task 9.2).

    Writes ``<out_dir>/ablation_fusion_weights.csv`` with one row per weight
    combination.
    """
    from src.config import CONFIG
    from src.serving.registry import build_eval_registry
    from eval.datasets import load_beir_dataset

    set_seed(seed)
    grid = grid or [
        (1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
        (0, 1, 1), (1, 0, 1), (1, 1, 0),
    ]
    out_dir = Path(out_dir)
    data_dir = Path(data_dir) if data_dir else CONFIG.path("data", "beir")

    corpus, queries, qrels = load_beir_dataset(dataset, data_dir)
    index_dir = CONFIG.path("data", "eval_indexes", dataset)
    registry = build_eval_registry(
        corpus, CONFIG, index_dir, metric="cosine", with_splade=True, with_reranker=False
    )

    rows: List[dict] = []
    for weights in grid:
        spec = SystemSpec("hybrid", weights=tuple(float(w) for w in weights), rerank=False)
        metric = evaluate_system(spec, queries, qrels, registry, dataset)
        rows.append({
            "w_bm25": weights[0], "w_dense": weights[1], "w_splade": weights[2],
            "ndcg_at_10": metric.ndcg_at_10,
            "mrr_at_10": metric.mrr_at_10,
            "recall_at_100": metric.recall_at_100,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation_fusion_weights.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["w_bm25", "w_dense", "w_splade", "ndcg_at_10", "mrr_at_10", "recall_at_100"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def compare_metrics(
    dataset: str,
    out_dir: Path = Path("eval/results"),
    data_dir: Optional[Path] = None,
    seed: int = 42,
) -> dict:
    """Cosine-vs-L2 dense-retrieval comparison on one dataset (task 9.3).

    Builds the dense index both ways and reports the metric delta. Writes
    ``<out_dir>/cosine_vs_l2.csv``.
    """
    from src.config import CONFIG
    from src.serving.registry import build_eval_registry
    from eval.datasets import load_beir_dataset

    set_seed(seed)
    out_dir = Path(out_dir)
    data_dir = Path(data_dir) if data_dir else CONFIG.path("data", "beir")

    corpus, queries, qrels = load_beir_dataset(dataset, data_dir)
    dense_spec = SystemSpec("dense", weights=None, rerank=False)

    results = {}
    for metric_name in ("cosine", "l2"):
        index_dir = CONFIG.path("data", "eval_indexes", f"{dataset}_{metric_name}")
        registry = build_eval_registry(
            corpus, CONFIG, index_dir, metric=metric_name,
            with_splade=False, with_reranker=False,
        )
        row = evaluate_system(dense_spec, queries, qrels, registry, dataset)
        results[metric_name] = row

    out = {
        "dataset": dataset,
        "ndcg_cosine": results["cosine"].ndcg_at_10,
        "ndcg_l2": results["l2"].ndcg_at_10,
        "ndcg_delta": results["cosine"].ndcg_at_10 - results["l2"].ndcg_at_10,
        "mrr_cosine": results["cosine"].mrr_at_10,
        "mrr_l2": results["l2"].mrr_at_10,
        "recall_cosine": results["cosine"].recall_at_100,
        "recall_l2": results["l2"].recall_at_100,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cosine_vs_l2.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out.keys()))
        writer.writeheader()
        writer.writerow(out)
    return out
