"""Unit tests for figure generation (task 9.4) — no models, CSV fixture only."""

import csv

from eval.figures import plot_pareto, plot_per_dataset_bars, generate_all_figures


def _write_metrics_csv(path):
    rows = [
        {"system": "bm25", "dataset": "scifact", "ndcg_at_10": 0.60, "mrr_at_10": 0.55, "recall_at_100": 0.85, "mean_latency_ms": 5.0, "reranker_calls": 0},
        {"system": "dense", "dataset": "scifact", "ndcg_at_10": 0.62, "mrr_at_10": 0.58, "recall_at_100": 0.88, "mean_latency_ms": 12.0, "reranker_calls": 0},
        {"system": "hybrid", "dataset": "scifact", "ndcg_at_10": 0.68, "mrr_at_10": 0.63, "recall_at_100": 0.92, "mean_latency_ms": 20.0, "reranker_calls": 0},
        {"system": "hybrid+rerank", "dataset": "scifact", "ndcg_at_10": 0.71, "mrr_at_10": 0.67, "recall_at_100": 0.92, "mean_latency_ms": 120.0, "reranker_calls": 50},
        {"system": "bm25", "dataset": "nfcorpus", "ndcg_at_10": 0.31, "mrr_at_10": 0.45, "recall_at_100": 0.25, "mean_latency_ms": 4.0, "reranker_calls": 0},
        {"system": "hybrid", "dataset": "nfcorpus", "ndcg_at_10": 0.34, "mrr_at_10": 0.50, "recall_at_100": 0.29, "mean_latency_ms": 18.0, "reranker_calls": 0},
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_plot_pareto_creates_file(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    _write_metrics_csv(csv_path)
    out = tmp_path / "figures" / "pareto.png"
    plot_pareto(csv_path, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_bars_creates_file(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    _write_metrics_csv(csv_path)
    out = tmp_path / "figures" / "bars.png"
    plot_per_dataset_bars(csv_path, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_all_figures(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_metrics_csv(results_dir / "metrics.csv")
    figures_dir = tmp_path / "figures"
    generate_all_figures(results_dir, figures_dir)
    assert (figures_dir / "pareto.png").exists()
    assert (figures_dir / "per_dataset_bars.png").exists()
