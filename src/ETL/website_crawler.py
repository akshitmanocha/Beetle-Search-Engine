#!/usr/bin/env python3
"""Crawl seed domains breadth-first and save each page's HTML in one pass.

Single-fetch: the HTML fetched to extract links is the HTML we keep, so there is
no separate download stage. Outputs match what the parser expects:
  data/crawled_websites.txt   one URL per line
  data/raw/<sha256(url)>.html  the page HTML
"""

import argparse
import hashlib
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Educational-Crawler/1.0"}


def sha_name(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def crawl(seed, urls_file, raw_dir, max_urls, max_depth, sleep_time):
    seed = seed if seed.startswith("http") else f"https://{seed}"
    domain = urlparse(seed).netloc.lower()
    visited, queue, count = set(), deque([(seed, 0)]), 0

    with open(urls_file, "a", encoding="utf-8") as urls_out:
        while queue and (not max_urls or count < max_urls):
            url, depth = queue.popleft()
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            try:
                r = requests.get(url, headers=HEADERS, timeout=10)
            except Exception:
                print(f"Failed to crawl {url}")
                continue
            if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                continue

            # Save the HTML we just fetched (no second download pass).
            (raw_dir / f"{sha_name(url)}.html").write_text(r.text, encoding="utf-8")
            urls_out.write(url + "\n")
            urls_out.flush()
            count += 1
            print(f"Crawled {count}: {url}")

            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                link, _ = urldefrag(urljoin(url, a["href"]))
                if urlparse(link).netloc.lower() == domain and link not in visited:
                    queue.append((link, depth + 1))

            time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description="Crawl websites from seed domains")
    parser.add_argument("--limit", type=int, help="Max URLs to crawl per seed")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    with open(project_root / "params.yaml") as f:
        cfg = yaml.safe_load(f)["ETL"]["crawler"]

    seeds_file = project_root / "data" / "seeds" / "seed_domains.txt"
    urls_file = project_root / "data" / "crawled_websites.txt"
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    seeds = [s.strip() for s in seeds_file.read_text().splitlines() if s.strip()]
    urls_file.write_text("")  # reset before a fresh crawl

    for seed in seeds:
        print(f"\nCrawling {seed}...")
        crawl(
            seed, urls_file, raw_dir,
            max_urls=args.limit or cfg["max_urls_per_seed"],
            max_depth=cfg["max_depth"],
            sleep_time=cfg["sleep_time"],
        )


if __name__ == "__main__":
    main()
