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