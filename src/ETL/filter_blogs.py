#!/usr/bin/env python3
"""
Filter clean blog dataset from parsed data using weak labels.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


def load_blog_labels(labels_path: Path) -> set:
    """Load blog IDs from strong labels CSV."""
    df = pd.read_csv(labels_path)
    blog_ids = set(df[df['label'] == 'blog']['id'])
    return blog_ids


def load_parsed_blogs(parsed_file: Path, blog_ids: set) -> List[Dict]:
    """Load and filter blog posts from parsed data."""
    print(f"Loading parsed data from {parsed_file}...")

    try:
        with open(parsed_file, 'r', encoding='utf-8') as f:
            all_docs = json.load(f)

        # Filter only the blogs and keep only essential fields
        blogs = []
        for doc in all_docs:
            if doc.get('id') in blog_ids:
                blogs.append({
                    'id': doc.get('id', ''),
                    'url': doc.get('url', ''),
                    'title': doc.get('title', ''),
                    'body_text': doc.get('body_text', '')
                })

        return blogs

    except Exception as e:
        print(f"  Error loading parsed data: {e}")
        return []


def calculate_stats(blogs: List[Dict]) -> Dict:
    """Calculate dataset statistics."""
    if not blogs:
        return {}

    word_counts = [len(b.get('body_text', '').split()) for b in blogs]

    return {
        'total_blogs': len(blogs),
        'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
        'min_word_count': min(word_counts) if word_counts else 0,
        'max_word_count': max(word_counts) if word_counts else 0,
    }


def save_json(blogs: List[Dict], output_path: Path):
    """Save blogs as a single JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(blogs, f, indent=2, ensure_ascii=False)


def filter_blogs():
    """Main function to filter and save clean blog dataset."""
    # Get script directory and project root
    project_root = Path(__file__).parent.parent.parent

    labels_path = project_root / 'data' / 'labels' / 'strong.csv'
    parsed_file = project_root / 'data' / 'parsed.json'
    output_dir = project_root / 'data' / 'clean'

    # Validate inputs
    if not labels_path.exists():
        print(f"✗ Error: {labels_path} not found")
        print(f"  Run the weak labeler first: dvc repro label")
        return

    if not parsed_file.exists():
        print(f"✗ Error: {parsed_file} not found")
        print(f"  Run the parser first: dvc repro parse")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing labels from {labels_path}...")
    blog_ids = load_blog_labels(labels_path)
    print(f"  Found {len(blog_ids)} blogs")

    print(f"\nFiltering blog posts from {parsed_file}...")
    blogs = load_parsed_blogs(parsed_file, blog_ids)
    print(f"  Loaded {len(blogs)}/{len(blog_ids)} blog posts")

    if len(blogs) == 0:
        print(f"\n✗ No blogs found!")
        return

    # Save as JSON
    output_path = output_dir / 'blogs.json'
    save_json(blogs, output_path)

    # Calculate and save statistics
    stats = calculate_stats(blogs)

    # Print summary
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n✓ Clean dataset created: {output_path}")
    print(f"  Total blogs: {stats['total_blogs']}")
    print(f"  Avg word count: {stats['avg_word_count']:.0f}")
    print(f"  Range: {stats['min_word_count']} - {stats['max_word_count']} words")
    print(f"  File size: {file_size_mb:.2f} MB")


if __name__ == '__main__':
    filter_blogs()