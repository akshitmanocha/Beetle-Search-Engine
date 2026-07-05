# Beetle Search Engine — self-contained demo image.
#
# Serves the static frontend AND the search API from a single FastAPI app, with
# the small BEIR-derived demo index baked in so the container is self-contained
# and starts without any network access. Uses /healthz as the platform health
# probe (Requirement 10).
#
# IMPORTANT: build the demo index on the host FIRST (it needs the gated
# embedder + your HF token, which must NOT be baked into the image):
#
#   HF_TOKEN=... python scripts/prepare_demo_index.py --dataset scifact
#   docker build -t beetle-search .
#
# The COPY data/ step then bakes the prepared index into the image.

FROM python:3.11-slim

WORKDIR /app

# System deps for the ML/scientific wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY src/ ./src/
COPY static/ ./static/
COPY scripts/ ./scripts/
COPY app.py params.yaml ./

# Bake the pre-built demo corpus + indexes (built on the host, see header).
# These are the artifacts build_registry loads: data/clean/blogs.json,
# data/bm25_index, data/faiss_index, data/splade_index.
COPY data/ ./data/

# Cache the model weights inside the image so startup needs no network.
# (Model downloads still require a token at build time for the gated embedder;
#  pass it as a build secret rather than a layer to avoid leaking it.)
ENV HF_HOME=/app/.hf_cache
ENV TOKENIZERS_PARALLELISM=false
# faiss-cpu and torch each ship an OpenMP runtime; pin OpenMP to one thread so
# they don't race and segfault when both run in the serving process.
ENV OMP_NUM_THREADS=1
# Restrict CORS to same-origin by default; override at deploy time.
ENV BEETLE_CORS_ORIGINS=""

EXPOSE 8000

# Health probe used by the platform (HF Spaces / Fly.io / compose).
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["python", "app.py"]
