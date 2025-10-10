#!/usr/bin/env python3
"""
Enhanced HTML Parser - Extracts comprehensive content and features from crawled HTML.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import yaml

from bs4 import BeautifulSoup
from trafilatura import extract
from trafilatura.metadata import extract_metadata as meta_extract
from readability import Document

PARSER_VERSION = "2.0.0"


def sha_name(url: str) -> str:
    """Generate filename from URL hash."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _normalize_metadata(meta):
    if not meta:
        return {}
    if isinstance(meta, dict):
        return meta
    # meta is an object (Document). copy relevant fields to a dict.
    return {
        "title": getattr(meta, "title", None),
        "author": getattr(meta, "author", None) or getattr(meta, "authors", None),
        "date": getattr(meta, "date", None),
    }

def extract_with_trafilatura(html: str, url: str):
    """Use Trafilatura to extract article metadata (robust to return type)."""
    try:
        text = extract(html, include_comments=False, include_tables=False, url=url)
        raw_meta = meta_extract(html)
        metadata = _normalize_metadata(raw_meta)

        author_raw = metadata.get("author") or ""
        if isinstance(author_raw, (list, tuple)):
            authors = [str(a).strip() for a in author_raw if str(a).strip()]
        else:
            authors = [a.strip() for a in re.split(r'[,;]|\band\b', str(author_raw)) if a.strip()]

        date_raw = metadata.get("date")
        if date_raw:
            publish_date = date_raw.isoformat() if hasattr(date_raw, "isoformat") else str(date_raw)
        else:
            publish_date = None

        return {
            "title": metadata.get("title") or None,
            "authors": authors,
            "publish_date": publish_date,
            "text": text,
        }
    except Exception as e:
        print(f"  Trafilatura parsing failed for {url}: {e}")
        return None

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


def extract_with_readability(html: str) -> str:
    """Fallback content extraction using readability-lxml."""
    try:
        doc = Document(html)
        # Extract the main content
        content_html = doc.summary()
        # Convert HTML to text using BeautifulSoup
        soup = BeautifulSoup(content_html, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"  Readability extraction failed: {e}")
        return ""


def parse_html_file(html_path: str, url: str, min_word_count: int) -> dict:
    """Parse HTML file and extract comprehensive content and features."""
    # Load HTML
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    sha = sha_name(url)

    # Parse with Trafilatura
    trafilatura_data = extract_with_trafilatura(html, url)

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Extract content - only use trafilatura if it extracted substantial content
    if trafilatura_data and trafilatura_data["text"]:
        word_count = len(trafilatura_data["text"].split())
        if word_count >= 50:
            # Trafilatura extracted good content
            title = trafilatura_data["title"] or "Untitled"
            body_text = trafilatura_data["text"]
            authors = trafilatura_data["authors"]
            publish_date = trafilatura_data["publish_date"]
        else:
            # Trafilatura extraction too short, use readability fallback
            print("        [INFO] Trafilatura extraction too short, using readability fallback")
            title = trafilatura_data["title"] or "Untitled"
            body_text = extract_with_readability(html)
            authors = trafilatura_data["authors"] if trafilatura_data["authors"] else []
            publish_date = trafilatura_data["publish_date"]
    else:
        # Trafilatura failed completely, use readability fallback
        print("        [INFO] Trafilatura failed, using readability fallback")
        title = "Untitled"
        body_text = extract_with_readability(html)
        authors = []
        publish_date = None

    # Extract features
    code_blocks_count = count_code_blocks(soup)
    citations = detect_citations(soup, body_text)
    has_author_bio = detect_author_bio(soup)

    return {
        "id": sha,
        "url": url,
        "title": title,
        "authors": authors,
        "publish_date": publish_date,
        "body_text": body_text,
        "word_count": len(body_text.split()),
        "char_count": len(body_text),
        "code_blocks_count": code_blocks_count,
        "has_arxiv_citation": citations["has_arxiv"],
        "has_doi_citation": citations["has_doi"],
        "has_references_section": citations["has_references_section"],
        "has_author_bio": has_author_bio,
        "parse_date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def parse_directory(urls_file: Path, raw_dir: Path, parsed_file: Path, min_word_count: int):
    """Parse all HTML files from data/raw using URLs from crawled_websites.txt."""
    parsed_file.parent.mkdir(parents=True, exist_ok=True)

    # Read URLs
    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Parsing {len(urls)} URLs...")

    all_data = []
    succeeded = 0
    failed = 0
    skipped = 0

    for url in urls:
        try:
            # Get the HTML file path
            sha = sha_name(url)
            html_file = raw_dir / f"{sha}.html"

            # Skip if HTML file doesn't exist
            if not html_file.exists():
                skipped += 1
                continue

            # Parse the HTML file
            data = parse_html_file(str(html_file), url, min_word_count)
            all_data.append(data)

            succeeded += 1

            if succeeded % 10 == 0:
                print(f"  {succeeded}/{len(urls)}...")

        except Exception as e:
            failed += 1
            print(f"  Error parsing {url}: {e}")

    # Save all parsed data to a single JSON file
    with open(parsed_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {succeeded} succeeded, {failed} failed, {skipped} skipped (no HTML file)")
    print(f"✓ Saved all parsed data to: {parsed_file}")


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        parser_params = params['ETL']['parser']

    project_root = Path(__file__).parent.parent.parent
    urls_file = project_root / "data" / "crawled_websites.txt"
    raw_dir = project_root / "data" / "raw"
    parsed_file = project_root / "data" / "parsed.json"

    if not urls_file.exists():
        print(f"✗ Error: {urls_file} not found")
        print(f"  Run the crawler first: dvc repro crawl")
        return

    parse_directory(
        urls_file,
        raw_dir,
        parsed_file,
        min_word_count=parser_params['min_word_count']
    )


if __name__ == "__main__":
    main()