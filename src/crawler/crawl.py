#!/usr/bin/env python3
"""
Polite seed crawler with comprehensive features for search engine building.

Features:
- robots.txt check per domain with crawl-delay respect
- URL canonicalization and deduplication
- concurrent workers with configurable parallelism
- content hashing for duplicate detection
- version snapshots for changed pages
- blog-like content heuristics and scoring
- JS detection and trap avoidance
- comprehensive logging and crawl manifest
- post-crawl reporting and candidate export
"""

import argparse
import hashlib
import json
import os
import queue
import time
import threading
import logging
import csv
import re
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin, urldefrag, parse_qs, urlencode
from pathlib import Path
import requests
from urllib import robotparser
from bs4 import BeautifulSoup
from groq import Groq

CRAWLER_VERSION = "1.0.0"
USER_AGENT = "DeepBlogSearch/1.0 (+mailto:your-email@example.com)"
DEFAULT_RATE = 1.5
DEFAULT_TIMEOUT = 10
DEFAULT_RETRIES = 2
DEFAULT_MAX_DEPTH = 2
DEFAULT_WORKERS = 2

# Initialize Groq client (API key from environment variable)
GROQ_CLIENT = None
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if GROQ_API_KEY:
        GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
except Exception:
    pass

# Trap detection patterns
TRAP_PATTERNS = [
    r'/\d{4}/\d{2}/\d{2}/',  # date-based URLs
    r'/page/\d+',             # pagination
    r'/tag/',                 # tag pages
    r'/category/',            # category pages
    r'/author/',              # author archives
    r'\?page=\d+',            # query pagination
]

def canonicalize_url(url: str) -> str:
    """Normalize URL for deduplication."""
    parsed = urlparse(url)

    # Normalize scheme to https
    scheme = parsed.scheme.lower()
    if scheme not in ('http', 'https'):
        return url

    # Normalize netloc (lowercase)
    netloc = parsed.netloc.lower()

    # Normalize path (remove trailing slash except for root)
    path = parsed.path
    if path != '/' and path.endswith('/'):
        path = path.rstrip('/')
    if not path:
        path = '/'

    # Sort query parameters for consistency
    query_params = parse_qs(parsed.query, keep_blank_values=True)
    sorted_query = urlencode(sorted(query_params.items()), doseq=True)

    # Remove fragment
    canonical = f"{scheme}://{netloc}{path}"
    if sorted_query:
        canonical += f"?{sorted_query}"

    return canonical

def sha_name(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return h

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def is_potential_trap(url: str) -> bool:
    """Detect URLs that might be infinite traps."""
    for pattern in TRAP_PATTERNS:
        if re.search(pattern, url):
            return True
    return False

def detect_js_requirement(html: str, url: str) -> bool:
    """Heuristically detect if page requires JS rendering."""
    if not html or len(html) < 500:
        return True

    soup = BeautifulSoup(html, "html.parser")

    # Check for common SPA frameworks
    js_indicators = [
        soup.find(id="root"),
        soup.find(id="app"),
        soup.find("div", {"data-reactroot": True}),
        soup.find("div", {"ng-app": True}),
    ]

    if any(js_indicators):
        # Check if there's actual content
        text = soup.get_text(strip=True)
        if len(text) < 200:
            return True

    return False

def extract_page_content(html: str, url: str, max_words=500):
    """Extract clean content from HTML for LLM analysis."""
    if not html:
        return {"title": "", "content": "", "url": url}

    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Try to get h1 if no title
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

    # Remove script, style, nav, footer, header
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Get main content (prefer article, main, or body)
    main_content = soup.find("article") or soup.find("main") or soup.find("body")

    if main_content:
        text = main_content.get_text(separator=" ", strip=True)
    else:
        text = soup.get_text(separator=" ", strip=True)

    # Clean and limit words
    words = text.split()[:max_words]
    content = " ".join(words)

    return {
        "title": title[:200],
        "content": content,
        "url": url
    }

def calculate_blog_heuristic_score(html: str, url: str, logger=None) -> float:
    """Score page using LLM (with fallback to heuristics)."""
    # Extract clean content
    page_data = extract_page_content(html, url, max_words=500)

    # Prepare prompt
    prompt = f"""Analyze this webpage and determine if it's a high-quality AI/ML blog post or technical article.
                URL: {page_data['url']}
                Title: {page_data['title']}
                Content preview: {page_data['content'][:1500]}

                Score this page from 0-100 based on:
                - Educational/technical content about AI, ML, deep learning
                - Long-form blog post or article (not just news snippets)
                - Contains explanations, code examples, or research insights
                - Written by researchers, engineers, or technical authors

                Return ONLY a JSON object with this exact format:
                {{"score": <number 0-100>, "reasoning": "<brief 1-sentence explanation>"}}"""

    # Call Groq API
    response = GROQ_CLIENT.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at identifying high-quality AI/ML technical blog posts and articles. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=150,
    )

    # Parse response
    result_text = response.choices[0].message.content.strip()

    # Extract JSON (handle markdown code blocks)
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0].strip()

    result = json.loads(result_text)
    score = float(result.get("score", 0))

    if logger:
        logger.debug(f"LLM score for {url}: {score} - {result.get('reasoning', '')}")

    return min(100.0, max(0.0, score))

def setup_logging(out_dir: str):
    """Setup logging to file and console."""
    log_dir = os.path.join(out_dir, "logs")
    safe_makedirs(log_dir)

    log_file = os.path.join(log_dir, f"crawl_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

def read_seeds(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    seeds = []
    for u in lines:
        if not u.startswith("http://") and not u.startswith("https://"):
            u = "https://" + u
        seeds.append(u)
    
    return seeds

def fetch_robots(session: requests.Session, base_url: str):
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        resp = session.get(robots_url, timeout=DEFAULT_TIMEOUT, headers={"User-Agent": USER_AGENT})
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            rp = None
    except Exception:
        rp = None
    return rp

def get_crawl_delay(rp: robotparser.RobotFileParser):
    if not rp:
        return None
    try:
        return rp.crawl_delay(USER_AGENT)
    except Exception:
        return None
    
def is_allowed(rp: robotparser.RobotFileParser, url: str):
    if not rp:
        return True
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True
    
def extract_same_origin_links(base_url: str, html: str):
    if not html:
        return []
    base_parsed = urlparse(base_url)
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        href_full = urljoin(base_url, href)
        href_full, _ = urldefrag(href_full)
        parsed = urlparse(href_full)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.netloc == base_parsed.netloc:
            links.add(href_full)
    return list(links)

def load_seen_urls(out_dir: str) -> set:
    """Load previously crawled URLs."""
    seen_file = os.path.join(out_dir, "seen_urls.json")
    if os.path.exists(seen_file):
        with open(seen_file, "r") as f:
            return set(json.load(f))
    return set()

def save_seen_urls(out_dir: str, seen_urls: set):
    """Save crawled URLs to prevent re-crawling."""
    seen_file = os.path.join(out_dir, "seen_urls.json")
    with open(seen_file, "w") as f:
        json.dump(sorted(list(seen_urls)), f, indent=2)

def save_page(out_dir: str, url: str, html: str, resp: requests.Response, seed: str,
              canonical_url: str, referer: str, seed_id: int, logger):
    """Save page with metadata (overwrites if exists)."""
    name = sha_name(canonical_url)
    html_path = os.path.join(out_dir, f"{name}.html")
    meta_path = os.path.join(out_dir, f"{name}.json")

    # Save HTML
    with open(html_path, "wb") as f:
        if isinstance(html, str):
            f.write(html.encode("utf-8", errors="replace"))
        else:
            f.write(html or b"")

    # Calculate heuristic score using LLM
    heuristic_score = calculate_blog_heuristic_score(html, url, logger)
    requires_js = detect_js_requirement(html, url)

    # Metadata
    meta = {
        "url": url,
        "canonical_url": canonical_url,
        "seed": seed,
        "seed_id": seed_id,
        "fetched_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "status_code": resp.status_code if resp is not None else None,
        "content_type": resp.headers.get("Content-Type") if resp is not None else None,
        "content_length": len(html) if html else 0,
        "final_url": resp.url if resp is not None else None,
        "referer": referer,
        "sha": name,
        "crawler_version": CRAWLER_VERSION,
        "heuristic_score": heuristic_score,
        "requires_js": requires_js,
        "headers": dict(resp.headers) if resp is not None else {},
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return True


def polite_fetch(session: requests.Session, url: str, host_last_access: dict, default_rate: float,
                 timeout: int, retries: int, rp_map: dict, logger):
    """Fetch URL with politeness, rate limiting, and smart retry logic."""
    parsed = urlparse(url)
    host = parsed.netloc

    # Check robots
    rp = rp_map.get(host)
    if rp and not is_allowed(rp, url):
        logger.debug(f"Disallowed by robots.txt: {url}")
        return None, "disallowed-by-robots", None

    # Compute delay
    now = time.time()
    last = host_last_access.get(host, 0)
    wait = max(0, default_rate - (now - last))
    if wait > 0:
        time.sleep(wait)

    attempt = 0
    backoff = 1.0

    while attempt <= retries:
        try:
            resp = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            host_last_access[host] = time.time()

            # Handle rate limiting specifically
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                    except ValueError:
                        wait_time = 60
                else:
                    wait_time = min(300, backoff * 30)  # Cap at 5 minutes

                logger.warning(f"Rate limited (429) for {url}, waiting {wait_time}s")
                time.sleep(wait_time)
                attempt += 1
                backoff *= 2
                continue

            # Handle server errors with backoff
            if resp.status_code == 503:
                logger.warning(f"Service unavailable (503) for {url}, backing off")
                time.sleep(backoff * 5)
                attempt += 1
                backoff *= 2
                continue

            return resp, None, None

        except requests.Timeout:
            logger.warning(f"Timeout for {url} (attempt {attempt + 1}/{retries + 1})")
            attempt += 1
            if attempt > retries:
                return None, "timeout", None
            time.sleep(backoff)
            backoff *= 2

        except requests.RequestException as e:
            logger.warning(f"Request error for {url}: {repr(e)} (attempt {attempt + 1}/{retries + 1})")
            attempt += 1
            if attempt > retries:
                return None, f"error:{repr(e)}", None
            time.sleep(backoff)
            backoff *= 2

    return None, "max-retries-exceeded", None

class CrawlStats:
    """Thread-safe statistics tracker."""
    def __init__(self):
        self.lock = threading.Lock()
        self.pages_fetched = 0
        self.pages_saved = 0
        self.pages_failed = 0
        self.pages_skipped = 0
        self.bytes_downloaded = 0
        self.errors = defaultdict(int)
        self.domains_contacted = set()
        self.start_time = time.time()

    def increment(self, metric, value=1):
        with self.lock:
            if metric == "fetched":
                self.pages_fetched += value
            elif metric == "saved":
                self.pages_saved += value
            elif metric == "failed":
                self.pages_failed += value
            elif metric == "skipped":
                self.pages_skipped += value
            elif metric == "bytes":
                self.bytes_downloaded += value

    def add_error(self, error_type):
        with self.lock:
            self.errors[error_type] += 1

    def add_domain(self, domain):
        with self.lock:
            self.domains_contacted.add(domain)

    def get_summary(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                "pages_fetched": self.pages_fetched,
                "pages_saved": self.pages_saved,
                "pages_failed": self.pages_failed,
                "pages_skipped": self.pages_skipped,
                "bytes_downloaded": self.bytes_downloaded,
                "domains_contacted": len(self.domains_contacted),
                "errors": dict(self.errors),
                "elapsed_seconds": elapsed,
                "pages_per_second": self.pages_fetched / elapsed if elapsed > 0 else 0,
            }

def worker_crawl(worker_id, url_queue, seen_lock, seen_set, host_last_access,
                 rp_map, out_dir, rate, timeout, retries, max_depth, stats, seen_urls, logger, stop_event):
    """Worker thread for concurrent crawling."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    logger.info(f"Worker {worker_id} started")

    while not stop_event.is_set():
        try:
            url, depth, seed, seed_id, referer = url_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Check if already seen (URL-based deduplication)
        with seen_lock:
            canonical_url = canonicalize_url(url)
            if canonical_url in seen_set or canonical_url in seen_urls:
                stats.increment("skipped")
                url_queue.task_done()
                continue
            seen_set.add(canonical_url)
            seen_urls.add(canonical_url)

        # Check for traps
        if is_potential_trap(canonical_url):
            logger.debug(f"Skipping potential trap: {canonical_url}")
            stats.increment("skipped")
            stats.add_error("potential-trap")
            url_queue.task_done()
            continue

        # Fetch the page
        resp, err, _ = polite_fetch(session, url, host_last_access, rate, timeout,
                                     retries, rp_map, logger)

        parsed = urlparse(url)
        stats.add_domain(parsed.netloc)

        if err:
            stats.increment("failed")
            stats.add_error(err)
            logger.debug(f"Failed to fetch {url}: {err}")
            url_queue.task_done()
            continue

        if resp is None:
            stats.increment("failed")
            stats.add_error("no-response")
            url_queue.task_done()
            continue

        stats.increment("fetched")

        # Only save HTML
        ctype = resp.headers.get("Content-Type", "")
        if resp.status_code == 200 and "html" in ctype.lower():
            html = resp.text
            stats.increment("bytes", len(html))

            try:
                save_page(out_dir, url, html, resp, seed, canonical_url,
                          referer, seed_id, logger)
                stats.increment("saved")

                # Enqueue same-origin links if depth < max_depth
                if depth < max_depth:
                    links = extract_same_origin_links(url, html)
                    for link in links:
                        url_queue.put((link, depth + 1, seed, seed_id, url))

            except Exception as e:
                logger.error(f"Error saving page {url}: {repr(e)}")
                stats.increment("failed")
                stats.add_error("save-error")
        else:
            stats.increment("skipped")
            if resp.status_code != 200:
                stats.add_error(f"status-{resp.status_code}")

        url_queue.task_done()

    logger.info(f"Worker {worker_id} stopped")

def crawl(seeds, out_dir, limit, rate, timeout, retries, max_depth, workers):
    """Main crawl orchestration with concurrent workers."""
    safe_makedirs(out_dir)
    logger = setup_logging(out_dir)

    logger.info(f"Starting crawl with {workers} workers, max depth {max_depth}, limit {limit}")
    logger.info(f"Seeds: {len(seeds)}")

    # Setup
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Prepare robots per host
    rp_map = {}
    for s in seeds:
        parsed = urlparse(s)
        host = parsed.netloc
        if host not in rp_map:
            rp = fetch_robots(session, s)
            rp_map[host] = rp
            cd = get_crawl_delay(rp)
            if isinstance(cd, (int, float)) and cd > 0:
                rate = max(rate, float(cd))
                logger.info(f"Using crawl-delay {cd}s for {host}")

    # Shared state
    url_queue = queue.Queue()
    seen_set = set()
    seen_lock = threading.Lock()
    host_last_access = defaultdict(float)
    seen_urls = load_seen_urls(out_dir)
    stats = CrawlStats()
    stop_event = threading.Event()

    # Seed the queue
    for idx, s in enumerate(seeds):
        url_queue.put((s, 0, s, idx, None))

    # Start workers
    threads = []
    for i in range(workers):
        t = threading.Thread(
            target=worker_crawl,
            args=(i, url_queue, seen_lock, seen_set, host_last_access,
                  rp_map, out_dir, rate, timeout, retries, max_depth, stats, seen_urls, logger, stop_event),
            daemon=True
        )
        t.start()
        threads.append(t)

    # Monitor progress
    try:
        while stats.pages_saved < limit:
            time.sleep(5)
            summary = stats.get_summary()
            logger.info(f"Progress: {summary['pages_saved']} saved, {summary['pages_fetched']} fetched, "
                        f"{summary['pages_failed']} failed, {summary['domains_contacted']} domains")

            if url_queue.empty() and url_queue.unfinished_tasks == 0:
                logger.info("Queue empty, crawl complete")
                break

            if stats.pages_saved >= limit:
                logger.info("Limit reached")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    # Shutdown
    logger.info("Stopping workers...")
    stop_event.set()

    for t in threads:
        t.join(timeout=5)

    # Save seen URLs for next crawl
    save_seen_urls(out_dir, seen_urls)

    summary = stats.get_summary()
    logger.info(f"Crawl complete: {summary}")

    return summary

def generate_crawl_manifest(out_dir, summary, args):
    """Generate crawl manifest with run details."""
    manifest = {
        "crawl_date": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "crawler_version": CRAWLER_VERSION,
        "seeds_file": args.seeds,
        "limit": args.limit,
        "rate": args.rate,
        "timeout": args.timeout,
        "retries": args.retries,
        "max_depth": args.max_depth,
        "workers": args.workers,
        "summary": summary,
    }

    manifest_path = os.path.join(out_dir, "crawl_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path

def generate_report(out_dir, summary):
    """Generate human-readable crawl report."""
    report_path = os.path.join(out_dir, "crawl_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("CRAWL REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Pages fetched: {summary['pages_fetched']}\n")
        f.write(f"Pages saved: {summary['pages_saved']}\n")
        f.write(f"Pages failed: {summary['pages_failed']}\n")
        f.write(f"Pages skipped: {summary['pages_skipped']}\n")
        f.write(f"Unique pages: {summary['pages_saved']}\n")
        f.write(f"Domains contacted: {summary['domains_contacted']}\n")
        f.write(f"Total bytes: {summary['bytes_downloaded']:,} ({summary['bytes_downloaded'] / 1024 / 1024:.2f} MB)\n")
        f.write(f"Elapsed time: {summary['elapsed_seconds']:.1f}s\n")
        f.write(f"Pages per second: {summary['pages_per_second']:.2f}\n\n")

        if summary['errors']:
            f.write("Errors:\n")
            for error, count in sorted(summary['errors'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {error}: {count}\n")

    return report_path

def export_candidates(out_dir, min_score=40, max_candidates=800):
    """Export candidate pages for labeling based on heuristic scores."""
    candidates = []

    # Read all metadata files
    for filename in os.listdir(out_dir):
        if filename.endswith(".json") and not filename.startswith("crawl"):
            meta_path = os.path.join(out_dir, filename)
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                score = meta.get("heuristic_score", 0)
                if score >= min_score:
                    candidates.append({
                        "url": meta.get("canonical_url", meta.get("url")),
                        "score": score,
                        "requires_js": meta.get("requires_js", False),
                        "word_count": meta.get("content_length", 0) // 5,
                        "domain": urlparse(meta.get("url", "")).netloc,
                    })
            except Exception:
                continue

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Limit to max_candidates
    candidates = candidates[:max_candidates]

    # Export to CSV
    labels_dir = os.path.join(Path(out_dir).parent.parent, "labels")
    safe_makedirs(labels_dir)
    csv_path = os.path.join(labels_dir, "for_labeling.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "score", "requires_js", "word_count", "domain"])
        writer.writeheader()
        writer.writerows(candidates)

    return csv_path, len(candidates)

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive polite crawler for search engine building"
    )
    parser.add_argument("--seeds", required=True, help="Seeds file (one URL per line)")
    parser.add_argument("--out", required=True, help="Output directory for raw HTML and metadata")
    parser.add_argument("--limit", type=int, default=500, help="Max pages to save")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="Seconds between requests to same host")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries on transient errors")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="Max same-origin link depth")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of concurrent workers")
    parser.add_argument("--export-candidates", action="store_true", help="Export candidate pages for labeling")
    parser.add_argument("--min-score", type=int, default=40, help="Minimum heuristic score for candidates")
    args = parser.parse_args()

    seeds = read_seeds(args.seeds)
    if not seeds:
        print("No seeds found. Exiting.")
        return

    print(f"Starting crawl with {len(seeds)} seeds...")
    summary = crawl(seeds, args.out, args.limit, args.rate, args.timeout, args.retries,
                    args.max_depth, args.workers)

    # Generate manifest and report
    manifest_path = generate_crawl_manifest(args.out, summary, args)
    report_path = generate_report(args.out, summary)

    print(f"\n{'=' * 60}")
    print("CRAWL COMPLETE")
    print(f"{'=' * 60}")
    print(f"Pages saved: {summary['pages_saved']}")
    print(f"Pages fetched: {summary['pages_fetched']}")
    print(f"Domains: {summary['domains_contacted']}")
    print(f"Data: {summary['bytes_downloaded'] / 1024 / 1024:.2f} MB")
    print(f"Time: {summary['elapsed_seconds']:.1f}s")
    print(f"\nManifest: {manifest_path}")
    print(f"Report: {report_path}")

    # Export candidates if requested
    if args.export_candidates:
        print("\nExporting candidates for labeling...")
        csv_path, count = export_candidates(args.out, args.min_score)
        print(f"Exported {count} candidates to {csv_path}")


if __name__ == "__main__":
    main()

