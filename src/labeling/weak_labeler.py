#!/usr/bin/env python3
"""
Weak Labeling Script - Generate heuristic-based labels for blog classification.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, Tuple


def calculate_blog_score(data: dict) -> Tuple[int, str]:
    """
    Calculate blog score based on heuristics.

    Returns:
        (score, reasoning) tuple
    """
    score = 0
    reasons = []

    # 1. URL Pattern Analysis
    url = data.get('url', '').lower()

    # Positive URL patterns
    blog_url_patterns = ['/blog/', '/post/', '/posts/', '/article/', '/articles/', '/news/']
    if any(pattern in url for pattern in blog_url_patterns):
        score += 3
        reasons.append("+3 blog URL pattern")

    # Date patterns in URL (YYYY/MM or YYYY-MM-DD)
    if re.search(r'/\d{4}/\d{2}/', url) or re.search(r'/\d{4}-\d{2}-\d{2}', url):
        score += 2
        reasons.append("+2 date in URL")

    # Negative URL patterns
    non_blog_patterns = ['/docs/', '/documentation/', '/api/', '/papers/', '/spec/', '/reference/']
    if any(pattern in url for pattern in non_blog_patterns):
        score -= 3
        reasons.append("-3 non-blog URL pattern")

    # 2. Author Information
    if data.get('has_author_bio'):
        score += 2
        reasons.append("+2 has author bio")

    if len(data.get('authors', [])) > 0:
        score += 2
        reasons.append(f"+2 has authors ({len(data['authors'])})")

    # 3. Publication Date
    if data.get('publish_date'):
        score += 3
        reasons.append("+3 has publish date")

    # 4. First-Person Writing Style
    first_person_ratio = data.get('first_person_ratio', 0)
    if first_person_ratio > 0.015:
        score += 2
        reasons.append(f"+2 high first-person ratio ({first_person_ratio:.4f})")
    elif first_person_ratio > 0.005:
        score += 1
        reasons.append(f"+1 moderate first-person ratio ({first_person_ratio:.4f})")

    # 5. Word Count (blogs are typically 500-5000 words)
    word_count = data.get('word_count', 0)
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

    # 6. Code Blocks (blogs have some code, but not excessive)
    code_blocks = data.get('code_blocks_count', 0)
    if 1 <= code_blocks <= 15:
        score += 1
        reasons.append(f"+1 moderate code blocks ({code_blocks})")
    elif code_blocks > 30:
        score -= 2
        reasons.append(f"-2 excessive code blocks ({code_blocks})")

    # 7. Academic Indicators (negative for blogs)
    if data.get('has_arxiv_citation'):
        score -= 2
        reasons.append("-2 has arXiv citations")

    if data.get('has_doi_citation'):
        score -= 2
        reasons.append("-2 has DOI citations")

    if data.get('has_references_section'):
        score -= 1
        reasons.append("-1 has references section")

    # 8. Header Structure (blogs typically have clear hierarchy)
    headers = data.get('headers', [])
    h1_count = sum(1 for h in headers if h.get('level') == 1)
    if h1_count == 1:
        score += 1
        reasons.append("+1 single H1 (good structure)")
    elif h1_count > 3:
        score -= 1
        reasons.append(f"-1 multiple H1s ({h1_count})")

    # 9. Title Analysis
    title = data.get('title', '').lower()
    if any(word in title for word in ['tutorial', 'guide', 'how to', 'introduction']):
        score += 1
        reasons.append("+1 educational title")

    return score, "; ".join(reasons)


def classify_blog(score: int) -> Tuple[str, float]:
    """
    Classify as blog/not-blog based on score.

    Returns:
        (label, confidence) tuple
    """
    if score >= 7:
        return "blog", 0.9
    elif score >= 5:
        return "blog", 0.7
    elif score >= 3:
        return "blog", 0.5
    elif score >= 0:
        return "uncertain", 0.5
    elif score >= -3:
        return "not-blog", 0.6
    else:
        return "not-blog", 0.8


def generate_weak_labels():
    """Generate weak labels from parsed data."""
    parsed_dir = Path("data/parsed")
    output_path = Path("data/labels/weak.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = list(parsed_dir.glob("*.json"))
    print(f"Processing {len(json_files)} parsed files...")

    results = []

    for i, json_file in enumerate(json_files, 1):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            score, reasoning = calculate_blog_score(data)
            label, confidence = classify_blog(score)

            results.append({
                'id': data.get('id', ''),
                'url': data.get('url', ''),
                'title': data.get('title', ''),
                'word_count': data.get('word_count', 0),
                'authors': '|'.join(data.get('authors', [])),
                'publish_date': data.get('publish_date', ''),
                'score': score,
                'label': label,
                'confidence': confidence,
                'reasoning': reasoning
            })

            if i % 100 == 0:
                print(f"  Processed {i}/{len(json_files)}...")

        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")

    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'url', 'title', 'word_count', 'authors', 'publish_date',
                      'score', 'label', 'confidence', 'reasoning']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print statistics
    blog_count = sum(1 for r in results if r['label'] == 'blog')
    not_blog_count = sum(1 for r in results if r['label'] == 'not-blog')
    uncertain_count = sum(1 for r in results if r['label'] == 'uncertain')

    print(f"\n✓ Weak labels generated: {output_path}")
    print(f"  Total: {len(results)}")
    print(f"  Blog: {blog_count} ({100*blog_count/len(results):.1f}%)")
    print(f"  Not-blog: {not_blog_count} ({100*not_blog_count/len(results):.1f}%)")
    print(f"  Uncertain: {uncertain_count} ({100*uncertain_count/len(results):.1f}%)")


if __name__ == "__main__":
    generate_weak_labels()
