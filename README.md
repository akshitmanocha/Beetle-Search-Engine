# Beetle Search Engine

Beetle is a search engine for high-quality AI research blog posts, designed to filter out low-quality, SEO-farmed content. It uses a combination of classic and modern retrieval techniques to provide relevant and technical blog posts for AI researchers and engineers.

## Features

- **Hybrid Search:** Combines BM25 and FAISS for efficient and accurate retrieval.
- **Two-Stage Filtering:** Uses a TF-IDF + Logistic Regression model for initial filtering and a Transformer-based model for fine-grained classification.
- **FastAPI Backend:** A modern, fast (high-performance) web framework for building APIs.
- **DVC Pipeline:** A DVC-managed pipeline for crawling, parsing, and extracting features from web pages.
- **Containerized:** Can be deployed using Docker and Kubernetes.

## Architecture

The project is composed of the following components:

- **ETL Pipeline:** A DVC-managed pipeline that crawls websites, downloads HTML, parses the content, and generates labels for training.
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

- Python 3.8+
- [Poetry](https://python-poetry.org/) for dependency management
- [DVC](https://dvc.org/) for data versioning

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Deep-Blog-Search.git
   cd Deep-Blog-Search
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull the data and models:**
   ```bash
   dvc pull
   ```

### Running the Application

1. **Start the FastAPI server:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. **Open your browser and navigate to `http://localhost:8000`**

## Usage

The main entry point for the application is `app.py`, which starts a FastAPI server. The server exposes a `/search` endpoint that accepts a JSON object with a "query" field.

The frontend is located in the `static` directory and can be accessed by navigating to the root URL (`/`).

### Docker Usage

To run the application with Docker, you can use Docker Compose.

1. **Pull DVC data:**
   ```bash
   dvc pull
   ```

2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

This will build the Docker image and start the service. You can then access the application at `http://localhost:8000`.

## Project Structure

```
├── app.py                  # FastAPI application
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Parameters for the DVC pipeline
├── requirements.txt        # Python dependencies
├── src                     # Source code
│   ├── ETL                 # ETL pipeline scripts
│   ├── index               # Indexing scripts
│   ├── models              # Model training and embedding scripts
│   ├── search              # Search and retrieval scripts
│   └── serving             # Serving scripts
├── static                  # Frontend files
└── data                    # Data (managed by DVC)
```

## DVC Pipeline

The `dvc.yaml` file defines the data pipeline. The main stages are:

- `crawl`: Crawls websites from a seed list.
- `download`: Downloads the HTML content of the crawled websites.
- `parse`: Parses the HTML to extract the main content.
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
