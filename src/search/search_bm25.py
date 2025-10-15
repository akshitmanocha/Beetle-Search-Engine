from pathlib import Path
from whoosh.index import open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser

def get_schema():
    """Defines the schema for the Whoosh index."""
    return Schema(
        id=ID(stored=True, unique=True),
        title=TEXT(stored=True, analyzer=StemmingAnalyzer(), field_boost=2.0),
        body_text=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    )

def search_bm25(query_str: str, index_path: Path, top_k: int = 10):
    """
    Searches the BM25 index.

    Args:
        query_str: The query string.
        index_path: Path to the Whoosh index.
        top_k: The number of top results to return.

    Returns:
        A list of tuples, where each tuple contains the document ID and the score.
    """
    ix = open_dir(index_path)
    schema = get_schema()

    with ix.searcher() as searcher:
        # Use MultifieldParser to search in both title and body_text
        parser = MultifieldParser(["title", "body_text"], schema=schema)
        query = parser.parse(query_str)

        # Perform the search
        results = searcher.search(query, limit=top_k)

        # Extract results
        return [(hit['id'], hit.score) for hit in results]


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent
    index_path = project_root / "data" / "bm25_index"

    # Example search
    print("\nPerforming a sample search...")
    sample_query = "transformer models"
    results = search_bm25(sample_query, index_path)
    print(f"Top {len(results)} results for '{sample_query}':")
    for doc_id, score in results:
        print(f"  - Document ID: {doc_id}, Score: {score:.4f}")
