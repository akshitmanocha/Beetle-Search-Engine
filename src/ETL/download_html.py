#!/usr/bin/env python3
"""
Downloads HTML content from a list of URLs and saves it to the data/raw directory.
"""

import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import yaml

import requests

HEADERS = {"User-Agent": "Educational-Crawler/1.0"}


def sha_name(url: str) -> str:
    """Generate filename from URL hash."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def download_url(url: str, raw_dir: Path, meta_dir: Path) -> bool:
    """Download a single URL and save it."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
            print(
                f"  Skipping {url} (status: {r.status_code}, content-type: {r.headers.get('Content-Type')})"
            )
            return False

        sha = sha_name(url)
        html_path = raw_dir / f"{sha}.html"
        meta_path = meta_dir / f"{sha}.json"

        # Save HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(r.text)

        # Save metadata
        metadata = {
            "url": url,
            "sha": sha,
            "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return True

    except requests.RequestException as e:
        print(f"  Error downloading {url}: {e}")
        return False


def download_pages(urls_file: Path, raw_dir: Path, meta_dir: Path, max_workers: int):
    """Download HTML from a list of URLs concurrently."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    succeeded = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_url = {
            executor.submit(download_url, url, raw_dir, meta_dir): url for url in urls
        }
        for future in concurrent.futures.as_completed(future_to_url):
            if future.result():
                succeeded += 1
            if (succeeded % 10 == 0) and succeeded > 0:
                print(f"  Downloaded {succeeded}/{len(urls)}...")

    failed = len(urls) - succeeded
    print(f"\nDone: {succeeded} succeeded, {failed} failed")


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        downloader_params = params['scraper']['downloader']

    project_root = Path(__file__).parent.parent.parent
    crawled_urls_file = project_root / "data" / "crawled_websites.txt"
    raw_dir = project_root / "data" / "raw"
    meta_dir = project_root / "data" / "metadata"

    if not crawled_urls_file.exists():
        print(f"✗ Error: {crawled_urls_file} not found")
        print(f"  Run the crawler first: dvc repro crawl")
    else:
        download_pages(
            crawled_urls_file,
            raw_dir,
            meta_dir,
            max_workers=downloader_params['max_workers']
        )