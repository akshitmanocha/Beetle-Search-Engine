#!/usr/bin/env python3
"""
Weak Labeling Script - Generate heuristic-based labels for blog classification.
"""

import csv
import json
import re
from typing import Tuple
from urllib.parse import urlparse
import yaml
from pathlib import Path

def score_url_patterns(url: str) -> tuple[int, str]:
    """
    Score URL based on common blog patterns using regex.
    Returns (delta_score, reason).
    """
    score = 0
    reasons = []

    parsed = urlparse(url)
    path = parsed.path.lower()

    # Negative indicators (not blogs)
    non_blog_patterns = [
        (r'/docs?/', -8, 'documentation'),
        (r'/api/', -8, 'API docs'),
        (r'/documentation/', -8, 'documentation'),
        (r'/reference/', -6, 'reference docs'),
        (r'/product[s]?/', -5, 'product page'),
        (r'/pricing/', -8, 'pricing page'),
        (r'/feature[s]?/', -5, 'features page'),
        (r'/about/', -8, 'about page'),
        (r'/contact/', -8, 'contact page'),
        (r'/terms/', -8, 'terms page'),
        (r'/privacy/', -8, 'privacy page'),
        (r'/author[s]?/', -6, 'author listing'),
        (r'/tag[s]?/', -5, 'tag page'),
        (r'/categor(y|ies)/', -5, 'category page'),
        (r'/archive[s]?/', -4, 'archive page'),
        (r'/search/', -6, 'search page'),
    ]

    # Check for non-blog patterns
    for pattern, points, reason in non_blog_patterns:
        if re.search(pattern, path):
            score += points
            reasons.append(f'{points} {reason}')
            break  # Only count the first matching negative pattern

    # Base URL or homepage (no meaningful path)
    if not path or path == '/' or path.strip('/') == '':
        score -= 8
        reasons.append('-8 base URL')

    reason_str = '; '.join(reasons) if reasons else 'neutral URL pattern'
    return score, reason_str


def calculate_blog_score(data: dict) -> Tuple[int, str]:
    """
    Calculate blog score based on simplified heuristics.
    Focuses on: URL patterns, authors, and title.

    Returns:
        (score, reasoning) tuple
    """
    score = 0
    reasons = []

    # 1. Author Information (HIGH WEIGHT)
    if len(data.get("authors", [])) > 0:
        score += 3
        reasons.append(f"+3 has authors ({len(data['authors'])})")

    if data.get("has_author_bio"):
        score += 3
        reasons.append("+3 has author bio")

    # 2. Publication Date (supporting signal)
    if data.get("publish_date"):
        score += 3
        reasons.append("+3 has publish date")

    # 3. Word Count (minimal check - only penalize very short content)
    word_count = data.get("word_count", 0)
    if word_count < 300:
        score -= 20
        reasons.append(f"-3 too short ({word_count} words)")

    # 4. arXiv citations (positive for technical blogs)
    if data.get("has_arxiv_citation") or data.get("has_references_section"):
        score += 3
        reasons.append("+3 has reference and citation")

    # 5. Code blocks (positive for technical blogs)
    code_blocks = data.get("code_blocks_count", 0)
    if code_blocks > 0:
        score += 3
        reasons.append(f"+2 has code blocks ({code_blocks})")

    reasons = [r for r in reasons if r]  # Remove empty strings
    if not reasons:
        return score, ""
    reasoning = "; ".join(reasons)
    if len(reasoning) > 200:
        reasoning = reasoning[:200] + "..."
    return score, reasoning


def classify_blog(score: int, score_threshold: int) -> str:
    """
    Classify as blog/not-blog based on score.

    Returns:
        label (blog or not-blog)
    """
    if score >= score_threshold:
        return "blog"
    else:
        return "not-blog"


def generate_weak_labels(parsed_file: Path, weak_labels_file: Path, score_threshold: int):
    """Generate weak labels from parsed data."""
    if not parsed_file.exists():
        print(f"✗ Error: {parsed_file} does not exist")
        print(f"  Please run the parser first: dvc repro parse")
        return

    weak_labels_file.parent.mkdir(parents=True, exist_ok=True)

    # Load all parsed data from single JSON file
    print(f"Loading parsed data from {parsed_file}...")
    with open(parsed_file, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)

    print(f"Processing {len(parsed_data)} parsed websites...")

    results = []

    for i, data in enumerate(parsed_data, 1):
        try:
            score, reasoning = calculate_blog_score(data)
            label = classify_blog(score, score_threshold)

            results.append(
                {
                    "id": data.get("id", ""),
                    "url": data.get("url", ""),
                    "title": data.get("title", ""),
                    "word_count": data.get("word_count", 0),
                    "authors": "|".join(data.get("authors", [])),
                    "publish_date": data.get("publish_date", ""),
                    "score": score,
                    "label": label,
                    "reasoning": reasoning,
                }
            )

            if i % 100 == 0:
                print(f"  Processed {i}/{len(parsed_data)}...")

        except Exception as e:
            print(f"  Error processing entry {i}: {e}")

    # Write to CSV
    with open(weak_labels_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id",
            "url",
            "title",
            "word_count",
            "authors",
            "publish_date",
            "score",
            "label",
            "reasoning",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print statistics
    if len(results) == 0:
        print(f"\n✗ No files processed. Make sure data/parsed/ contains JSON files.")
        return

    blog_count = sum(1 for r in results if r["label"] == "blog")
    not_blog_count = sum(1 for r in results if r["label"] == "not-blog")

    print(f"\n✓ Weak labels generated: {weak_labels_file}")
    print(f"  Total: {len(results)}")
    print(f"  Blog: {blog_count} ({100*blog_count/len(results):.1f}%)")
    print(f"  Not-blog: {not_blog_count} ({100*not_blog_count/len(results):.1f}%)")


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        heuristic_params = params['ETL']['heuristic_labeling']

    project_root = Path(__file__).parent.parent.parent
    parsed_file = project_root / "data" / "parsed.json"
    weak_labels_file = project_root / "data" / "labels" / "weak.csv"

    generate_weak_labels(
        parsed_file,
        weak_labels_file,
        score_threshold=heuristic_params['score_threshold']
    )