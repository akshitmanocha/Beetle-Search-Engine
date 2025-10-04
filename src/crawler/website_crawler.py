import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from w3lib.url import canonicalize_url
import time
from collections import deque
import argparse

HEADERS = {"User-Agent": "Educational-Crawler/1.0"}
MAX_DEPTH = 2
SLEEP = 0

def same_domain(u1, u2):
    return urlparse(u1).netloc.lower() == urlparse(u2).netloc.lower()

def crawl(seed, out_path="data/crawled_websites.txt", max_urls=None):
    seed = seed if seed.startswith("http") else f"https://{seed}"
    base = urlparse(seed).netloc.lower()
    visited, q = set(), deque([(seed, 0)])
    crawled_count = 0

    with open(out_path, "a") as out:
        while q:
            if max_urls and crawled_count >= max_urls:
                break

            url, depth = q.popleft()
            if depth > MAX_DEPTH or url in visited:
                continue
            visited.add(url)
            try:
                r = requests.get(url, headers=HEADERS, timeout=10)
                if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                    continue
            except requests.RequestException:
                continue

            out.write(url + "\n")
            out.flush()
            crawled_count += 1
            print(f"Crawled {crawled_count}: {url}")

            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                abs_url = canonicalize_url(urljoin(url, a["href"]))
                if same_domain(abs_url, seed) and abs_url not in visited:
                    q.append((abs_url, depth + 1))
            time.sleep(SLEEP)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl websites from seed domains")
    parser.add_argument("--limit", type=int, help="Maximum number of URLs to crawl per seed")
    args = parser.parse_args()

    with open("data/seeds/seed_domains.txt", "r") as f:
        seeds = [line.strip() for line in f if line.strip()]

    for seed in seeds:
        print(f"\nCrawling {seed}...")
        crawl(seed, max_urls=args.limit)