#!/usr/bin/env python3
"""
Downloads HTML content from a list of URLs and saves it to the data/raw directory.
"""

import concurrent.futures
import hashlib
import argparse
from pathlib import Path
import yaml

import requests

HEADERS = {"User-Agent": "Educational-Crawler/1.0"}


def sha_name(url: str) -> str:
    """Generate filename from URL hash."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def download_url(url: str, raw_dir: Path) -> bool:
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

        # Save HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(r.text)

        return True

    except Exception as e:
        # Skip this URL on any error and continue with the next one
        print(f"  Failed to download {url}: {e}")
        return False


def download_pages(urls_file: Path, raw_dir: Path, max_workers: int):
    """Download HTML from a list of URLs concurrently."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    succeeded = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_url = {
            executor.submit(download_url, url, raw_dir): url for url in urls
        }
        for future in concurrent.futures.as_completed(future_to_url):
            if future.result():
                succeeded += 1
            if (succeeded % 10 == 0) and succeeded > 0:
                print(f"  Downloaded {succeeded}/{len(urls)}...")

    failed = len(urls) - succeeded
    print(f"\nDone: {succeeded} succeeded, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HTML content from a list of URLs.")
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of concurrent workers for downloading."
    )
    args = parser.parse_args()

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        downloader_params = params['ETL']['downloader']

    project_root = Path(__file__).parent.parent.parent
    crawled_urls_file = project_root / "data" / "crawled_websites.txt"
    raw_dir = project_root / "data" / "raw"

    if not crawled_urls_file.exists():
        print(f"✗ Error: {crawled_urls_file} not found")
        print(f"  Run the crawler first: dvc repro crawl")
    else:
        download_pages(
            crawled_urls_file,
            raw_dir,
            max_workers=args.max_workers or downloader_params['max_workers']
        )