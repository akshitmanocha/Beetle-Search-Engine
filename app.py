"""Beetle Search Engine — production-shaped FastAPI app.

  * Loads the model/index registry ONCE at startup via a FastAPI ``lifespan``
    (no per-request model reloads or corpus rescans).
  * ``/healthz`` returns 200 once the process is up; ``/readyz`` returns 200 only
    when the registry loaded cleanly, else 503 with the missing artifact list.
  * ``/search`` is a typed contract that routes through the registry-backed
    retrievers and weighted hybrid fusion, with optional reranking.
  * CORS is restricted to configured origins; queries are validated; logs are
    structured.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CONFIG
from src.serving.contracts import (
    HealthResponse,
    ReadyResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

# ---------------------------------------------------------------------------
# Structured logging (JSON lines with a request id) — fixes S10.
# ---------------------------------------------------------------------------

class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        return json.dumps(payload)


def _configure_logging() -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    return logging.getLogger("beetle")


logger = _configure_logging()

# Module-level registry handle, populated at startup.
REGISTRY = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the registry once at startup (Requirement 2.1)."""
    global REGISTRY
    logger.info("Building registry at startup...")
    try:
        from src.serving.registry import build_registry

        REGISTRY = build_registry(CONFIG)
        if REGISTRY.is_ready():
            logger.info("Registry ready: %d docs loaded", len(REGISTRY.corpus))
        else:
            logger.error("Registry NOT ready; missing: %s", REGISTRY.missing_artifacts())
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Registry build raised: %s", exc)
        REGISTRY = None
    yield
    REGISTRY = None


app = FastAPI(title="Beetle Search Engine", lifespan=lifespan)

# CORS restricted to configured origins (fixes S1). Set BEETLE_CORS_ORIGINS to a
# comma-separated list (e.g. the deployed HF Space URL). Defaults to localhost.
_origins_env = os.environ.get("BEETLE_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # credentials + wildcard is the invalid combo we avoid
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Map the public ``search_method`` names to the internal retriever names.
_METHOD_ALIASES = {"faiss": "dense"}


def _retrieve(method: str, query: str, top_k: int):
    """Run a single retriever or the hybrid path; returns (doc_id, score) list."""
    from src.search.search_bm25 import search_bm25_registry
    from src.search.search_faiss import search_faiss_registry
    from src.search.search_splade import search_splade_registry
    from src.search.hybrid_search import hybrid_search

    method = _METHOD_ALIASES.get(method, method)
    if method == "bm25":
        return search_bm25_registry(query, REGISTRY, top_k=top_k)
    if method == "dense":
        return search_faiss_registry(query, REGISTRY, top_k=top_k)
    if method == "splade":
        return search_splade_registry(query, REGISTRY, top_k=top_k)
    # default: hybrid
    return hybrid_search(query, REGISTRY, top_k=top_k)


def _build_items(scored, query: str) -> list:
    """Assemble typed SearchResultItem rows from (doc_id, score) pairs."""
    items = []
    for rank, (doc_id, score) in enumerate(scored):
        doc = REGISTRY.corpus.get(doc_id, {})
        text = doc.get("text", doc.get("body_text", "")) or ""
        items.append(
            SearchResultItem(
                doc_id=str(doc_id),
                title=doc.get("title", "") or "",
                url=doc.get("url", "") or "",
                snippet=text[:350],
                score=float(score),
                source_ranks={},
            )
        )
    return items


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    """Liveness: 200 once the process is up."""
    return HealthResponse(status="ok")


@app.get("/readyz")
async def readyz():
    """Readiness: 200 only when the registry loaded cleanly, else 503."""
    if REGISTRY is not None and REGISTRY.is_ready():
        return ReadyResponse(status="ready")
    missing = REGISTRY.missing_artifacts() if REGISTRY is not None else ["registry"]
    return JSONResponse(
        status_code=503,
        content=ReadyResponse(status="not_ready", missing=missing).model_dump(),
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if REGISTRY is None or not REGISTRY.is_ready():
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        scored = _retrieve(request.search_method, request.query, request.top_k)

        if request.reranker_enabled and scored:
            from src.models.reranker import rerank

            scored = rerank(request.query, scored[: request.top_k], REGISTRY)

        scored = scored[: request.rerank_k]
        items = _build_items(scored, request.query)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "search ok",
            extra={"request_id": request_id},
        )
        logging.getLogger("beetle.access").info(
            json.dumps({
                "request_id": request_id,
                "method": request.search_method,
                "reranker": request.reranker_enabled,
                "n_results": len(items),
                "latency_ms": round(latency_ms, 2),
            })
        )

        return SearchResponse(
            query=request.query,
            results=items,
            total_results=len(items),
            search_method=request.search_method,
            reranker_enabled=request.reranker_enabled,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("search failed: %s", exc, extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/config")
async def get_config():
    """Default search configuration for the frontend."""
    methods = ["hybrid", "bm25", "dense", "splade"]
    return {
        "search": CONFIG.params.get("search", {}),
        "available_methods": methods,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
