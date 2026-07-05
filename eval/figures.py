"""Figure generation from the ablation result CSVs (task 9.4).

Renders:
  * ``eval/figures/pareto.png`` — quality (nDCG@10) vs compute (mean query
    latency), one point per (dataset, system).
  * ``eval/figures/per_dataset_bars.png`` — grouped bar chart of nDCG@10 per
    system per dataset.

Reads ``eval/results/metrics.csv`` (the combined table written by
``run_ablation``). Uses a non-interactive matplotlib backend so it works
headless (CI / container).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read_rows(metrics_csv: Path) -> List[dict]:
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_pareto(metrics_csv: Path, out_path: Path) -> None:
    """Quality (nDCG@10) vs compute (mean latency ms) scatter, labeled by system."""
    rows = _read_rows(metrics_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    # One marker shape per dataset, color per system.
    datasets = sorted({r["dataset"] for r in rows})
    systems = sorted({r["system"] for r in rows})
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    cmap = plt.get_cmap("tab10")
    sys_color = {s: cmap(i % 10) for i, s in enumerate(systems)}
    ds_marker = {d: markers[i % len(markers)] for i, d in enumerate(datasets)}

    for r in rows:
        x = float(r["mean_latency_ms"])
        y = float(r["ndcg_at_10"])
        ax.scatter(
            x, y, color=sys_color[r["system"]], marker=ds_marker[r["dataset"]],
            s=90, edgecolors="black", linewidths=0.5,
        )
        ax.annotate(r["system"], (x, y), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")

    ax.set_xlabel("Mean query latency (ms)  —  compute")
    ax.set_ylabel("nDCG@10  —  quality")
    ax.set_title("Quality vs Compute (upper-left is better)")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Legends: color=system, marker=dataset.
    sys_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=sys_color[s],
                   markersize=9, label=s) for s in systems
    ]
    ds_handles = [
        plt.Line2D([0], [0], marker=ds_marker[d], color="black", linestyle="",
                   markersize=8, label=d) for d in datasets
    ]
    leg1 = ax.legend(handles=sys_handles, title="System", loc="lower right", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=ds_handles, title="Dataset", loc="upper center", fontsize=8)

    # Give the data a little headroom so point labels don't collide with the legend.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_dataset_bars(metrics_csv: Path, out_path: Path) -> None:
    """Grouped bar chart of nDCG@10 per system per dataset."""
    rows = _read_rows(metrics_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = sorted({r["dataset"] for r in rows})
    systems = sorted({r["system"] for r in rows})

    # ndcg[system][dataset]
    ndcg: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        ndcg[r["system"]][r["dataset"]] = float(r["ndcg_at_10"])

    n_sys = len(systems)
    width = 0.8 / max(n_sys, 1)
    x_base = range(len(datasets))

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(systems):
        xs = [x + i * width for x in x_base]
        ys = [ndcg[s].get(d, 0.0) for d in datasets]
        ax.bar(xs, ys, width=width, label=s, color=cmap(i % 10),
               edgecolor="black", linewidth=0.4)

    ax.set_xticks([x + (n_sys - 1) * width / 2 for x in x_base])
    ax.set_xticklabels(datasets)
    ax.set_ylabel("nDCG@10")
    ax.set_title("nDCG@10 by system and dataset")
    ax.set_ylim(0, 1)
    ax.legend(title="System", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all_figures(
    results_dir: Path = Path("eval/results"),
    figures_dir: Path = Path("eval/figures"),
) -> None:
    """Render both figures from ``<results_dir>/metrics.csv``."""
    metrics_csv = Path(results_dir) / "metrics.csv"
    plot_pareto(metrics_csv, Path(figures_dir) / "pareto.png")
    plot_per_dataset_bars(metrics_csv, Path(figures_dir) / "per_dataset_bars.png")


if __name__ == "__main__":
    generate_all_figures()
