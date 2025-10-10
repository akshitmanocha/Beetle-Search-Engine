import argparse
import time
from collections import deque
from urllib.parse import urljoin, urlparse
import yaml
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from w3lib.url import canonicalize_url

HEADERS = {"User-Agent": "Educational-Crawler/1.0"}


def same_domain(u1, u2):
    return urlparse(u1).netloc.lower() == urlparse(u2).netloc.lower()


def crawl(seed, out_path, max_urls, max_depth, sleep_time):
    seed = seed if seed.startswith("http") else f"https://{seed}"
    base = urlparse(seed).netloc.lower()
    visited, q = set(), deque([(seed, 0)])
    crawled_count = 0

    with open(out_path, "a") as out:
        while q:
            if max_urls and crawled_count >= max_urls:
                break

            url, depth = q.popleft()
            if depth > max_depth or url in visited:
                continue
            visited.add(url)

            try:
                r = requests.get(url, headers=HEADERS, timeout=10)
                if r.status_code != 200 or "text/html" not in r.headers.get(
                    "Content-Type", ""
                ):
                    continue

                out.write(url + "\n")
                out.flush()
                crawled_count += 1
                print(f"Crawled {crawled_count}: {url}")

                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    try:
                        abs_url = canonicalize_url(urljoin(url, a["href"]))
                        if same_domain(abs_url, seed) and abs_url not in visited:
                            q.append((abs_url, depth + 1))
                    except ValueError:
                        continue
                time.sleep(sleep_time)
            except Exception:
                print(f"Failed to crawl {url}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl websites from seed domains")
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of URLs to crawl per seed",
    )
    args = parser.parse_args()

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        crawler_params = params['ETL']['crawler']

    # Correctly define project_root and other paths
    project_root = Path(__file__).parent.parent.parent
    seed_domains_file = project_root / "data" / "seeds" / "seed_domains.txt"
    crawled_urls_file = project_root / "data" / "crawled_websites.txt"

    with open(seed_domains_file, "r") as f:
        seeds = [line.strip() for line in f if line.strip()]

    # Clear the crawled URLs file before starting
    with open(crawled_urls_file, "w") as f:
        pass

    for seed in seeds:
        print(f"\nCrawling {seed}...")
        crawl(
            seed,
            crawled_urls_file,
            max_urls=args.limit or crawler_params['max_urls_per_seed'],
            max_depth=crawler_params['max_depth'],
            sleep_time=crawler_params['sleep_time'],
        )
