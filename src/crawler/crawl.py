#!/usr/bin/env python3
"""
Polite seed crawler.
Saves raw HTML per URL as JSON files under OUT_DIR/raw/<domain>/<safe_filename>.json

Features:
- robots.txt check per domain
- requests with retry and backoff
- configurable concurrency and per-domain delay
- simple domain-limited queue seeded from seed_domains.txt
- optional depth=0 (only seed pages) or follow internal links up to depth 1
- outputs metadata + html, suitable for DVC tracking
"""

