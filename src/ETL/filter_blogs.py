#!/usr/bin/env python3
"""
Filter clean blog dataset from parsed data using weak labels.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


def load_blog_labels(labels_path: Path) -> set:
    """Load blog IDs from weak labels CSV."""
    df = pd.read_csv(labels_path)
    blog_ids = set(df[df['label'] == 'blog']['id'])
    return blog_ids


def load_parsed_blogs(parsed_dir: Path, blog_ids: set) -> List[Dict]:
    """Load and filter blog posts from parsed data."""
    blogs = []

    for json_file in parsed_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                doc = json.load(f)

            if doc.get('id') in blog_ids:
                blogs.append(doc)

        except Exception as e:
            print(f"  Warning: Failed to load {json_file.name}: {e}")

    return blogs


def calculate_stats(blogs: List[Dict]) -> Dict:
    """Calculate dataset statistics."""
    if not blogs:
        return {}

    word_counts = [b.get('word_count', 0) for b in blogs]

    return {
        'total_blogs': len(blogs),
        'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
        'min_word_count': min(word_counts) if word_counts else 0,
        'max_word_count': max(word_counts) if word_counts else 0,
        'with_authors': sum(1 for b in blogs if b.get('authors')),
        'with_publish_date': sum(1 for b in blogs if b.get('publish_date')),
        'avg_code_blocks': sum(b.get('code_blocks_count', 0) for b in blogs) / len(blogs),
    }


def save_jsonl(blogs: List[Dict], output_path: Path):
    """Save blogs as JSONL (one JSON per line)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for blog in blogs:
            f.write(json.dumps(blog, ensure_ascii=False) + '\n')


def filter_blogs():
    """Main function to filter and save clean blog dataset."""
    # Get script directory and project root
    project_root = Path(__file__).parent.parent.parent

    labels_path = project_root / 'data' / 'labels' / 'weak.csv'
    parsed_dir = project_root / 'data' / 'parsed'
    output_dir = project_root / 'data' / 'clean'

    # Validate inputs
    if not labels_path.exists():
        print(f"✗ Error: {labels_path} not found")
        print(f"  Run the weak labeler first: dvc repro label")
        return

    if not parsed_dir.exists():
        print(f"✗ Error: {parsed_dir} not found")
        print(f"  Run the parser first: dvc repro parse")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing labels from {labels_path}...")
    blog_ids = load_blog_labels(labels_path)
    print(f"  Found {len(blog_ids)} blogs")

    print(f"\nFiltering blog posts from {parsed_dir}...")
    blogs = load_parsed_blogs(parsed_dir, blog_ids)
    print(f"  Loaded {len(blogs)}/{len(blog_ids)} blog posts")

    if len(blogs) == 0:
        print(f"\n✗ No blogs found!")
        return

    # Sort by publish date (newest first, None last)
    blogs.sort(
        key=lambda x: x.get('publish_date') or '',
        reverse=True
    )

    # Save as JSONL
    output_path = output_dir / 'blogs.jsonl'
    save_jsonl(blogs, output_path)

    # Calculate and save statistics
    stats = calculate_stats(blogs)
    stats_path = output_dir / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n✓ Clean dataset created: {output_path}")
    print(f"  Total blogs: {stats['total_blogs']}")
    print(f"  Avg word count: {stats['avg_word_count']:.0f}")
    print(f"  Range: {stats['min_word_count']} - {stats['max_word_count']} words")
    print(f"  With authors: {stats['with_authors']} ({100*stats['with_authors']/stats['total_blogs']:.1f}%)")
    print(f"  With dates: {stats['with_publish_date']} ({100*stats['with_publish_date']/stats['total_blogs']:.1f}%)")
    print(f"  Avg code blocks: {stats['avg_code_blocks']:.1f}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"\n✓ Statistics saved: {stats_path}")


if __name__ == '__main__':
    filter_blogs()