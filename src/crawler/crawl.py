#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

def canonicalize_url(url):
    parsed = urlparse(url)
    path = parsed.path.rstrip('/') if parsed.path != '/' else '/'
    return f"{parsed.scheme}://{parsed.netloc}{path}"

def sha_name(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()

def extract_title(html):
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""

def calculate_score(html, url, title):
    """Use Groq LLM to score page quality"""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        content = " ".join(text.split()[:500])

        prompt = f"""Analyze if this is a high-quality AI/ML blog post or technical article or a research.
URL: {url}
Title: {title}
Content: {content}

Score 0-100 based on:
- Educational AI/ML/deep learning content
- Long-form blog post or article
- Explanations, code examples, or research insights

Return ONLY JSON: {{"score": <number>, "reasoning": "<1 sentence>"}}"""

        response = GROQ_CLIENT.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
            messages=[
                {"role": "system", "content": "You are an expert at identifying high-quality AI/ML technical content. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )

        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)
        return min(100.0, max(0.0, float(result.get("score", 0))))
    except:
        print(f"Failed to score: {url}")
        return 0.0

def save_page(url, html, final_url, title, score):
    canonical = canonicalize_url(url)
    sha = sha_name(canonical)

    html_dir = "data/raw"
    meta_dir = "data/metadata"
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # Save HTML
    with open(os.path.join(html_dir, f"{sha}.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # Save metadata
    meta = {
        "url": url,
        "canonical_url": canonical,
        "final_url": final_url,
        "title": title,
        "heuristic_score": score,
        "sha": sha,
        "fetched_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    }

    with open(os.path.join(meta_dir, f"{sha}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def crawl(urls, limit):
    visited = set()
    saved = 0

    for url in urls:
        if saved >= limit:
            break

        canonical = canonicalize_url(url)
        if canonical in visited:
            continue
        visited.add(canonical)

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                continue

            html = resp.text
            title = extract_title(html)
            score = 0  # Skip scoring for speed

            save_page(url, html, resp.url, title, score)
            saved += 1
            print(f"[{saved}/{limit}] {url}")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error: {url} - {e}")
            continue

    print(f"\nCrawl complete: {saved} pages saved")

def main():
    parser = argparse.ArgumentParser(description="Crawl and score URLs")
    parser.add_argument("--limit", type=int, default=500, help="Max pages")
    args = parser.parse_args()

    with open("data/crawled_websites.txt", "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    crawl(urls, args.limit)

if __name__ == "__main__":
    main()
