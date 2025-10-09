#!/bin/bash
set -e

python src/ETL/website_crawler.py --limit 2000
python src/ETL/parse.py
python src/ETL/download_html.py --max-workers 32
python src/ETL/heuristic_label.py
