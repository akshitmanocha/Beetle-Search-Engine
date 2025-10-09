#!/usr/bin/env python3
"""
Enhanced HTML Parser - Extracts comprehensive content and features from crawled HTML.
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import yaml

from bs4 import BeautifulSoup
from newspaper import Article

PARSER_VERSION = "2.0.0"


def sha_name(url: str) -> str:
    """Generate filename from URL hash."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def extract_with_newspaper(html: str, url: str):
    """Use newspaper3k to extract article metadata."""
    try:
        article = Article(url)
        article.set_html(html)
        article.parse()
        return {
            "title": article.title,
            "authors": article.authors,
            "publish_date": (
                article.publish_date.isoformat() if article.publish_date else None
            ),
            "text": article.text,
        }
    except:
        print("parsing with newspaper failed")
        return None


def extract_headers(soup: BeautifulSoup) -> list:
    """Extract all headers (h1-h6)."""
    headers = []
    for i in range(1, 7):
        for tag in soup.find_all(f"h{i}"):
            headers.append({"level": i, "text": tag.get_text(strip=True)})
    return headers


def count_code_blocks(soup: BeautifulSoup) -> int:
    """Count code blocks (<pre>, <code>, <pre><code>)."""
    pre_tags = soup.find_all("pre")
    code_tags = soup.find_all("code")

    # Count pre>code as one block
    pre_code = len([p for p in pre_tags if p.find("code")])

    # Count standalone pre and code
    standalone_pre = len(pre_tags) - pre_code
    standalone_code = len([c for c in code_tags if not c.find_parent("pre")])

    return pre_code + standalone_pre + standalone_code


def detect_citations(soup: BeautifulSoup, body_text: str) -> dict:
    """Detect presence of citations.""" 
    html_str = str(soup)

    has_arxiv = "arxiv.org" in html_str or "arxiv" in body_text.lower()
    has_doi = (
        "doi.org" in html_str
        or re.search(r"\bdoi:\s*10\.\d+", body_text, re.I) is not None
    )

    # Check for references section
    has_references = False
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "div", "section"]):
        text = tag.get_text(strip=True).lower()
        if text in ["references", "bibliography", "citations", "works cited"]:
            has_references = True
            break

    return {
        "has_arxiv": has_arxiv,
        "has_doi": has_doi,
        "has_references_section": has_references,
    }


def detect_author_bio(soup: BeautifulSoup) -> bool:
    """Detect if author bio exists."""
    # Look for common author bio patterns
    bio_indicators = [
        "author-bio",
        "author-info",
        "about-author",
        "author-description",
    ]

    for indicator in bio_indicators:
        if soup.find(class_=re.compile(indicator, re.I)) or soup.find(
            id=re.compile(indicator, re.I)
        ):
            return True

    # Check for meta tags
    for meta in soup.find_all("meta"):
        if "author" in str(meta.get("name", "")).lower() or "author" in str(
            meta.get("property", "")
        ).lower():
            return True

    return False


def extract_content_fallback(soup: BeautifulSoup) -> str:
    """Fallback content extraction."""
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.find("body")
    if main:
        return main.get_text(separator=" ", strip=True)
    return soup.get_text(separator=" ", strip=True)


def parse_html_file(html_path: str, metadata_path: str, min_word_count: int) -> dict:
    """Parse HTML file and extract comprehensive content and features."""
    # Load HTML
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    # Load metadata
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except:
            pass

    url = metadata.get("url", "unknown")
    sha = metadata.get("sha", sha_name(url))

    # Parse with newspaper3k
    newspaper_data = extract_with_newspaper(html, url)

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Extract content - only use newspaper if it extracted substantial content

    if newspaper_data and newspaper_data["text"]:
        word_count = len(newspaper_data["text"].split())
        if word_count >= min_word_count:
            # Newspaper extracted good content
            title = newspaper_data["title"] or metadata.get("title", "Untitled")
            body_text = newspaper_data["text"]
            authors = newspaper_data["authors"]
            publish_date = newspaper_data["publish_date"]
        else:
            # Newspaper extraction too short, use fallback
            title = newspaper_data["title"] or metadata.get("title", "Untitled")
            body_text = extract_content_fallback(soup)
            authors = newspaper_data["authors"] if newspaper_data["authors"] else []
            publish_date = newspaper_data["publish_date"]
    else:
        # Newspaper failed completely, use fallback
        title = metadata.get("title", "Untitled")
        body_text = extract_content_fallback(soup)
        authors = []
        publish_date = None

    # Extract features
    headers = extract_headers(soup)
    code_blocks_count = count_code_blocks(soup)
    citations = detect_citations(soup, body_text)
    has_author_bio = detect_author_bio(soup)

    return {
        "id": sha,
        "url": url,
        "canonical_url": metadata.get("canonical_url", url),
        "title": title,
        "authors": authors,
        "publish_date": publish_date,
        "body_text": body_text,
        "headers": headers,
        "word_count": len(body_text.split()),
        "char_count": len(body_text),
        "code_blocks_count": code_blocks_count,
        "has_arxiv_citation": citations["has_arxiv"],
        "has_doi_citation": citations["has_doi"],
        "has_references_section": citations["has_references_section"],
        "has_author_bio": has_author_bio,
        "fetched_at": metadata.get("fetched_at"),
        "parse_date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def parse_directory(raw_dir: Path, meta_dir: Path, parsed_dir: Path, min_word_count: int):
    """Parse all HTML files from data/raw using metadata from data/metadata."""
    parsed_dir.mkdir(parents=True, exist_ok=True)

    html_files = list(raw_dir.glob("*.html"))
    print(f"Parsing {len(html_files)} files...")

    succeeded = 0
    failed = 0

    for html_file in html_files:
        try:
            meta_file = meta_dir / html_file.with_suffix(".json").name
            data = parse_html_file(str(html_file), str(meta_file), min_word_count)

            output = parsed_dir / f"{data['id']}.json"
            with open(output, "w") as f:
                json.dump(data, f, indent=2)

            succeeded += 1

            if succeeded % 10 == 0:
                print(f"  {succeeded}/{len(html_files)}...")

        except Exception as e:
            failed += 1
            print(f"  Error: {html_file.name} - {e}")

    print(f"\nDone: {succeeded} succeeded, {failed} failed")


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        parser_params = params['ETL']['parser']

    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    meta_dir = project_root / "data" / "metadata"
    parsed_dir = project_root / "data" / "parsed"

    parse_directory(
        raw_dir,
        meta_dir,
        parsed_dir,
        min_word_count=parser_params['min_word_count']
    )


if __name__ == "__main__":
    main()