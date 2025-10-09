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

import ollama


def score_url_with_ollama(url: str, model: str) -> tuple[int, str]:
    """
    Use Ollama Python API for URL pattern recognition.
    Returns (delta_score, reason).
    """

    parsed_url = urlparse(url)
    path_components = parsed_url.path.strip("/").split("/")
    if path_components == [""]:
        path_components = ["not a blog, it is a base URL"]

    prompt = (
        "Your duty is to analyze the provided URL's structure and patterns to determine if it points to a blog post or a non-blog page (like documentation, product page, etc.)."
        "If the URL contains Machine Learning, Data Science, AI, or Programming related terms, It is a blog."
        "Based on your analysis of the URL pattern, rate the likelihood of it being a blog post on a scale from -5 (definitely not a blog) to +20 (definitely a blog). "
        'Respond strictly in JSON format: {"reason": "Your reasoning here", "score": INT}.\n'
        f"URL components: {path_components}"
    )
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        text = response["message"]["content"]
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return 0, "invalid ollama output"
        data = json.loads(match.group(0))
        print(f"  Scoring URL with Ollama: {url}: got score {data.get('score', 0)}")
        return int(data.get("score")), f"ollama: {data.get('reason', '')}"
    except Exception as e:
        return 0, f"ollama error: {e}"


def calculate_blog_score(data: dict, ollama_model: str) -> Tuple[int, str]:
    """
    Calculate blog score based on heuristics.

    Returns:
        (score, reasoning) tuple
    """
    score = 0
    reasons = []

    # 1. URL Pattern Analysis with Ollama
    url = data.get("url", "")

    # Use Ollama to intelligently score the URL
    ollama_score, ollama_reason = score_url_with_ollama(url, ollama_model)
    if ollama_reason and ollama_reason.strip():
        reasons.append(f"blog URL ({ollama_reason.strip()})")
    score += ollama_score

    # Date patterns in URL (still use regex for this)
    if re.search(r"/\d{4}/\d{2}/", url) or re.search(r"/\d{4}-\d{2}-\d{2}", url):
        score += 2
        reasons.append("+2 date in URL")

    # 2. Author Information
    if data.get("has_author_bio"):
        score += 2
        reasons.append("+2 has author bio")

    if len(data.get("authors", [])) > 0:
        score += 2
        reasons.append(f"+2 has authors ({len(data['authors'])})")

    # 3. Publication Date
    if data.get("publish_date"):
        score += 3
        reasons.append("+3 has publish date")

    # 4. Word Count (blogs are typically 500-5000 words)
    word_count = data.get("word_count", 0)
    if 500 <= word_count <= 5000:
        score += 2
        reasons.append(f"+2 ideal blog length ({word_count} words)")
    elif 300 <= word_count < 500:
        score += 1
        reasons.append(f"+1 acceptable length ({word_count} words)")
    elif word_count < 200:
        score -= 3
        reasons.append(f"-3 too short ({word_count} words)")
    elif word_count > 8000:
        score -= 1
        reasons.append(f"-1 very long ({word_count} words)")

    # 5. Code Blocks (blogs have some code, but not excessive)
    code_blocks = data.get("code_blocks_count", 0)
    if 1 <= code_blocks <= 15:
        score += 1
        reasons.append(f"+1 moderate code blocks ({code_blocks})")
    elif code_blocks > 30:
        score -= 2
        reasons.append(f"-2 excessive code blocks ({code_blocks})")

    # 7. Academic Indicators (negative for blogs)
    if data.get("has_arxiv_citation"):
        score -= 2
        reasons.append("-2 has arXiv citations")

    if data.get("has_doi_citation"):
        score -= 2
        reasons.append("-2 has DOI citations")

    if data.get("has_references_section"):
        score -= 1
        reasons.append("-1 has references section")

    # 8. Header Structure (blogs typically have clear hierarchy)
    headers = data.get("headers", [])
    h1_count = sum(1 for h in headers if h.get("level") == 1)
    if h1_count == 1:
        score += 1
        reasons.append("+1 single H1 (good structure)")
    elif h1_count > 3:
        score -= 1
        reasons.append(f"-1 multiple H1s ({h1_count})")

    # 9. Title Analysis
    title = data.get("title", "").lower()
    if any(word in title for word in ["tutorial", "guide", "how to", "introduction"]):
        score += 1
        reasons.append("+1 educational title")

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


def generate_weak_labels(parsed_dir: Path, weak_labels_file: Path, score_threshold: int, ollama_model: str):
    """Generate weak labels from parsed data."""
    if not parsed_dir.exists():
        print(f"✗ Error: {parsed_dir} does not exist")
        print(f"  Please run the parser first: dvc repro parse")
        return

    weak_labels_file.parent.mkdir(parents=True, exist_ok=True)

    json_files = list(parsed_dir.glob("*.json"))
    print(f"Processing {len(json_files)} parsed files...")

    results = []

    for i, json_file in enumerate(json_files, 1):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            score, reasoning = calculate_blog_score(data, ollama_model)
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
                print(f"  Processed {i}/{len(json_files)}...")

        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")

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
    parsed_dir = project_root / "data" / "parsed"
    weak_labels_file = project_root / "data" / "labels" / "weak.csv"

    generate_weak_labels(
        parsed_dir,
        weak_labels_file,
        score_threshold=heuristic_params['score_threshold'],
        ollama_model=heuristic_params['ollama_model']
    )