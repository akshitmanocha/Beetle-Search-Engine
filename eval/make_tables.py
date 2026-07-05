"""Generate LaTeX table fragments from the evaluation result CSVs (task 14.2).

Writes ``\\input``-able fragments under ``paper/tables/`` so the paper's tables
stay sourced from the actual run:

  paper/tables/main_metrics.tex          (from eval/results/metrics.csv)
  paper/tables/fusion_weights.tex        (from eval/results/ablation_fusion_weights.csv)
  paper/tables/cosine_vs_l2.tex          (from eval/results/cosine_vs_l2.csv)

Each fragment is a ``tabular`` body (rows only would be brittle; we emit a full
``tabular`` so the paper just does ``\\input{tables/main_metrics}``).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List


def _read(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(x: str, places: int = 4) -> str:
    try:
        return f"{float(x):.{places}f}"
    except (TypeError, ValueError):
        return str(x)


def main_metrics_table(results_dir: Path, out_path: Path) -> None:
    rows = _read(results_dir / "metrics.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"System & Dataset & nDCG@10 & MRR@10 & Recall@100 & Latency (ms) & Rerank calls \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['system']} & {r['dataset']} & {_fmt(r['ndcg_at_10'])} & "
            f"{_fmt(r['mrr_at_10'])} & {_fmt(r['recall_at_100'])} & "
            f"{_fmt(r['mean_latency_ms'], 1)} & {r.get('reranker_calls', '0')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fusion_weights_table(results_dir: Path, out_path: Path) -> None:
    rows = _read(results_dir / "ablation_fusion_weights.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$(w_{bm25}, w_{dense}, w_{splade})$ & nDCG@10 & MRR@10 & Recall@100 \\",
        r"\midrule",
    ]
    for r in rows:
        w = f"({r['w_bm25']}, {r['w_dense']}, {r['w_splade']})"
        lines.append(
            f"{w} & {_fmt(r['ndcg_at_10'])} & {_fmt(r['mrr_at_10'])} & {_fmt(r['recall_at_100'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cosine_vs_l2_table(results_dir: Path, out_path: Path) -> None:
    rows = _read(results_dir / "cosine_vs_l2.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & nDCG@10 (cosine) & nDCG@10 (L2) & $\Delta$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['dataset']} & {_fmt(r['ndcg_cosine'])} & {_fmt(r['ndcg_l2'])} & "
            f"{_fmt(r['ndcg_delta'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_all_tables(
    results_dir: Path = Path("eval/results"),
    tables_dir: Path = Path("paper/tables"),
) -> None:
    results_dir = Path(results_dir)
    tables_dir = Path(tables_dir)
    main_metrics_table(results_dir, tables_dir / "main_metrics.tex")
    fusion_weights_table(results_dir, tables_dir / "fusion_weights.tex")
    cosine_vs_l2_table(results_dir, tables_dir / "cosine_vs_l2.tex")


if __name__ == "__main__":
    generate_all_tables()
