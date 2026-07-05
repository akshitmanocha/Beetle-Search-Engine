# Beetle Search Engine

Beetle is a hybrid search engine over high-quality AI/ML content. It composes
retrieval families — lexical (BM25), dense (FAISS, cosine), and learned-sparse
(SPLADE) — fused with weighted Reciprocal
Rank Fusion, with an optional cross-encoder reranker. Alongside the serving path it
ships a **reproducible evaluation harness** that measures every component on small
public BEIR benchmarks (SciFact and NFCorpus) — so retrieval-quality claims are
backed by numbers, not folklore.

It runs end-to-end on a single laptop (Apple M4 Pro, MPS/CPU; no external GPU).

## Features

- **Hybrid search:** weighted RRF over BM25 + dense (cosine) + SPLADE.
- **Cross-encoder reranking:** loaded once at start-up, with input truncation.
- **Evaluation harness:** nDCG@10 / MRR@10 / Recall@100 over BEIR datasets, a
  five-system ablation, a fusion-weight sweep, and a cosine-vs-L2 comparison;
  emits CSV tables and Pareto / bar-chart figures.
- **Production-shaped FastAPI backend:** start-up model/index registry,
  `/healthz` + `/readyz`, tightened CORS, structured logging, a typed result
  contract.
- **Tested:** pytest unit + integration tests and Hypothesis property tests for
  the eight core correctness properties (RRF, cosine, metrics, determinism).
- **Reproducible:** pinned dependencies + a DVC pipeline; containerized demo.

> **Research write-up.** A short LaTeX paper (`paper/beetle.tex`) presents the
> system and the ablation, with tables and figures sourced directly from the
> evaluation outputs. See [plan.md](plan.md) for the sprint scope and the
> demoted research roadmap (QARR / CR-CEE / LDDE).

## Architecture

The project is composed of the following components:

- **ETL Pipeline:** A DVC-managed pipeline that crawls websites (saving HTML as it goes), parses the content, and generates labels for training.
- **Indexing:** Builds BM25, FAISS, and SPLADE indexes for fast retrieval.
- **Models:** Includes models for embedding, reranking, and classification.
- **Serving:** A FastAPI application that exposes a search API.
- **Frontend:** A simple HTML, CSS and JavaScript frontend for interacting with the search engine.

## Core Concepts

### ETL and Content Extraction

The ETL (Extract, Transform, Load) pipeline is responsible for collecting and processing the blog posts. It uses a combination of `trafilatura` and `readability-lxml` for robust content extraction from HTML. This process extracts the main text, title, author, and publication date, while also identifying features like code blocks, citations, and author bios.

### Semi-Supervised Labeling

To filter out low-quality content, the project uses a semi-supervised labeling approach. Initially, a set of "weak" labels are generated using a heuristic-based method (`heuristic_label.py`). These labels are then used to train a TF-IDF based Logistic Regression model (`train_tfidf.py`), which in turn generates a set of "strong" labels for the entire dataset. This allows for a more accurate classification of blog posts without requiring a large manually labeled dataset.

### Search and Indexing

Beetle employs a hybrid search strategy, combining several indexing and retrieval techniques:

- **BM25:** A classical keyword-based search algorithm that ranks documents based on the frequency and inverse document frequency of the query terms. It's highly effective for matching keywords and phrases.
- **FAISS (Facebook AI Similarity Search):** A library for efficient similarity search on dense vector embeddings. The blog posts are converted into high-dimensional vectors using a SentenceTransformer model, and FAISS is used to quickly find the most similar documents to a query vector.
- **SPLADE:** A model that learns sparse representations for documents and queries. Unlike dense embeddings, SPLADE vectors are sparse and interpretable, and can be indexed with inverted indexes, making them very efficient for retrieval.

### Hybrid Search and Reciprocal Rank Fusion

Hybrid search combines the strengths of keyword-based and vector-based search. In this project, the results from BM25 and FAISS are combined using **Reciprocal Rank Fusion (RRF)**. RRF is a simple yet powerful technique that merges multiple ranked lists by giving more weight to documents that appear higher in each list. This results in a more robust and accurate ranking than either method could achieve alone.

### Reranking

After the initial retrieval, a more powerful Transformer-based model can be used to rerank the top results. This reranker takes the query and the retrieved documents as input and re-orders them based on a more fine-grained understanding of their semantic relationship. This two-stage process allows for a fast initial retrieval followed by a more accurate but slower reranking of a small number of candidates.

## Getting Started

### Prerequisites

- Python 3.9+ (tested on 3.9; the container uses 3.11). All pinned dependencies
  install with prebuilt wheels on Apple Silicon / CPU — no external GPU needed.
- A Hugging Face token with access to `google/embeddinggemma-300m` (the dense
  embedder is gated). Set it via `huggingface-cli login` or `export HF_TOKEN=...`.
- [DVC](https://dvc.org/) (optional) for the data pipeline.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Beetle-Search-Engine.git
   cd Beetle-Search-Engine
   ```

2. **Create a virtual environment and install dependencies (pinned):**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

### Running the evaluation harness

```bash
# Five-system ablation on the default BEIR datasets (downloads them).
python -c "from eval.ablation import run_ablation; run_ablation(['scifact','nfcorpus'])"
# Generate figures + LaTeX tables from the results.
python eval/figures.py && python eval/make_tables.py
```

Results land in `eval/results/*.csv`, figures in `eval/figures/`, and LaTeX
fragments in `paper/tables/`.

### Running the demo (local)

1. **Build the demo index** (BEIR corpus → on-disk artifacts; needs the HF token):
   ```bash
   HF_TOKEN=... python scripts/prepare_demo_index.py --dataset scifact
   ```

2. **Start the server:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. **Open `http://localhost:8000`** and try the method selector (BM25 / dense /
   SPLADE / hybrid) and the reranker toggle.

### Running the tests

```bash
pytest -m "not integration"   # fast unit + property tests (no model downloads)
pytest -m integration         # slow: loads real models, builds real indexes
```

## Usage

The main entry point for the application is `app.py`, which starts a FastAPI server. The server exposes a `/search` endpoint that accepts a JSON object with a "query" field.

The frontend is located in the `static` directory and can be accessed by navigating to the root URL (`/`).

### Docker Usage

The container serves the UI **and** the API from one FastAPI app and bakes the
small demo index in, so it is self-contained. Build the index on the host first
(it needs your HF token, which is intentionally **not** baked into the image):

```bash
HF_TOKEN=... python scripts/prepare_demo_index.py --dataset scifact
docker compose up --build
```

Then open `http://localhost:8000`. The platform health probe is `/healthz`.
The recommended free-tier deploy target is **Hugging Face Spaces** (Docker);
set `BEETLE_CORS_ORIGINS` to the Space's origin.

## Project Structure

```
├── app.py                  # FastAPI application (registry, /healthz, typed /search)
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Parameters (models, search, ETL)
├── requirements.txt        # Pinned Python dependencies
├── src                     # Source code
│   ├── config.py           # Centralized config: paths, params, device (singleton)
│   ├── ETL                 # ETL pipeline scripts
│   ├── index               # Index builders (FAISS cosine + L2, SPLADE)
│   ├── models              # Embedding + reranker
│   ├── search              # Retrievers (bm25/faiss/splade) + weighted RRF hybrid
│   └── serving             # Registry + typed contracts
├── eval                    # Evaluation harness
│   ├── metrics.py          # nDCG@10 / MRR@10 / Recall@100
│   ├── datasets.py         # BEIR loaders
│   ├── ablation.py         # 5-system ablation, weight sweep, cosine-vs-L2
│   ├── figures.py          # Pareto + per-dataset bar charts
│   └── make_tables.py      # LaTeX table fragments from results CSVs
├── paper                   # LaTeX research paper (beetle.tex, tables/, figures/)
├── scripts                 # prepare_demo_index.py (BEIR → demo artifacts)
├── tests                   # pytest unit + property + integration tests
├── static                  # Frontend (method selector + reranker toggle)
└── data                    # Indexes + corpus (BEIR cache, demo index)
```

## DVC Pipeline

The `dvc.yaml` file defines the data pipeline. The main stages are:

- `crawl`: Crawls websites from a seed list, saving each page's HTML in one pass.
- `parse`: Parses the saved HTML to extract the main content.
- `label`: Generates weak labels for the parsed content.
- `train_tfidf`: Trains a TF-IDF model to generate strong labels.
- `filter`: Filters the blogs based on the generated labels.
- `embed`: Generates embeddings for the filtered blogs.
- `build_faiss`: Builds a FAISS index for similarity search.
- `build_bm25`: Builds a BM25 index for keyword search.
- `build_splade`: Builds a SPLADE index.

To run the full pipeline, use the following command:

```bash
dvc repro
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
