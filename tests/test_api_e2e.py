"""End-to-end API integration test (task 11.3).

Drives the FastAPI app via TestClient against a small fixture corpus injected
into the registry, exercising the hybrid + rerank /search path and the health
endpoints (Requirements 9.1, 9.2, 12.3).

Marked ``integration`` because it loads real models through the registry.
"""

import pytest
from fastapi.testclient import TestClient

import app as app_module
from src.config import CONFIG
from src.serving.registry import build_eval_registry

pytestmark = pytest.mark.integration

FIXTURE_CORPUS = {
    "transformers": {
        "title": "The Transformer Architecture",
        "text": "Attention is all you need. Self-attention powers language models.",
        "url": "https://example.com/transformers",
    },
    "gardening": {
        "title": "Growing Tomatoes",
        "text": "Plant seedlings after frost; water and give sunlight.",
        "url": "https://example.com/tomatoes",
    },
    "cooking": {
        "title": "Sourdough Bread",
        "text": "Ferment a flour-and-water starter, knead, proof, and bake.",
        "url": "https://example.com/sourdough",
    },
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("api_eval_indexes")
    registry = build_eval_registry(
        FIXTURE_CORPUS, CONFIG, index_dir, metric="cosine",
        with_splade=True, with_reranker=True,
    )
    # Inject the ready registry directly and DO NOT use TestClient as a context
    # manager: the `with` form would run the app's lifespan, which rebuilds the
    # registry from absent on-disk DVC artifacts and clobber this fixture.
    app_module.REGISTRY = registry
    c = TestClient(app_module.app)
    yield c
    app_module.REGISTRY = None


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_ready(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_search_hybrid(client):
    r = client.post("/search", json={
        "query": "attention in language models",
        "search_method": "hybrid", "top_k": 3, "rerank_k": 3,
        "reranker_enabled": False,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["search_method"] == "hybrid"
    assert body["results"]
    # Typed contract fields present.
    first = body["results"][0]
    assert {"doc_id", "title", "url", "snippet", "score", "source_ranks"} <= set(first.keys())
    assert first["doc_id"] == "transformers"


def test_search_hybrid_with_rerank(client):
    r = client.post("/search", json={
        "query": "transformer attention for NLP",
        "search_method": "hybrid", "top_k": 3, "rerank_k": 3,
        "reranker_enabled": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["reranker_enabled"] is True
    assert body["results"][0]["doc_id"] == "transformers"


def test_search_each_method(client):
    for method in ("bm25", "dense", "splade"):
        r = client.post("/search", json={
            "query": "attention language models", "search_method": method,
            "top_k": 3, "rerank_k": 3,
        })
        assert r.status_code == 200, method
        assert r.json()["results"], method


def test_search_empty_query_rejected(client):
    r = client.post("/search", json={"query": "   "})
    assert r.status_code == 400
