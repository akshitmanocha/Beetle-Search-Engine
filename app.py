from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import yaml

from main import search_and_rerank

app = FastAPI(title="Blog Search Engine")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    rerank_k: int = 5
    search_method: str = "hybrid"
    reranker_enabled: bool = False


class SearchResponse(BaseModel):
    query: str
    results: list
    total_results: int
    search_method: str
    reranker_enabled: bool


@app.get("/")
async def root():
    return {"message": "Blog Search Engine API", "status": "running"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform a search query with optional reranking

    Parameters:
    - query: Search query string
    - top_k: Number of results to retrieve before reranking (default: 20)
    - rerank_k: Number of results to return after reranking (default: 5)
    - search_method: Search method to use (hybrid, bm25, faiss, splade)
    - reranker_enabled: Whether to enable reranking (default: False)
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        results = search_and_rerank(
            query=request.query,
            top_k=request.top_k,
            rerank_k=request.rerank_k,
            search_method=request.search_method,
            reranker_enabled=request.reranker_enabled
        )

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_method=request.search_method,
            reranker_enabled=request.reranker_enabled
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get default search configuration from params.yaml"""
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        return {
            "search": params.get('search', {}),
            "available_methods": ["hybrid", "bm25", "faiss", "splade"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading config: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
