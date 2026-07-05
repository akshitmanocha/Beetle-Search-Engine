"""Typed API contracts for the search service (fixes S8).

Pydantic models defining the request/response shapes so clients get a stable,
documented contract instead of an untyped ``list``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """A search request. ``search_method`` selects the retriever(s)."""

    query: str
    top_k: int = Field(default=20, ge=1, le=100)
    rerank_k: int = Field(default=5, ge=1, le=100)
    search_method: str = Field(default="hybrid")  # hybrid | bm25 | dense | splade
    reranker_enabled: bool = False


class SearchResultItem(BaseModel):
    """One result row in the typed search response."""

    doc_id: str
    title: str
    url: str = ""
    snippet: str = ""
    score: float
    source_ranks: Dict[str, int] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """The typed search response."""

    query: str
    results: List[SearchResultItem]
    total_results: int
    search_method: str
    reranker_enabled: bool


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    missing: Optional[List[str]] = None
