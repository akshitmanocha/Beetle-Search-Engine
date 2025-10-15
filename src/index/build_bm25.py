import json
from pathlib import Path
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer

def get_schema():
    """Defines the schema for the Whoosh index."""
    return Schema(
        id=ID(stored=True, unique=True),
        title=TEXT(stored=True, analyzer=StemmingAnalyzer(), field_boost=2.0),
        body_text=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    )

def build_bm25_index(data_path: Path, index_path: Path):
    """
    Builds and saves a Whoosh BM25 index.

    Args:
        data_path: Path to the parsed JSON data.
        index_path: Path to save the Whoosh index.
    """
    # Create schema and index directory
    schema = get_schema()
    index_path.mkdir(parents=True, exist_ok=True)

    # Create the index
    ix = create_in(index_path, schema)
    writer = ix.writer()

    # Load data and add documents to the index
    with open(data_path, "r") as f:
        documents = json.load(f)

    for doc in documents:
        writer.add_document(
            id=doc.get("id"),
            title=doc.get("title"),
            body_text=doc.get("body_text"),
        )
    
    writer.commit()
    print(f"✓ BM25 index built and saved to {index_path}")

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "parsed.json"
    index_path = project_root / "data" / "bm25_index"

    build_bm25_index(data_path, index_path)
