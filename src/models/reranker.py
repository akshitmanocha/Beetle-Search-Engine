from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self, model_name: str, device: str = 'cpu'):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        """
        Reranks a list of documents based on a query.

        Args:
            query: The query string.
            documents: A list of documents to be reranked. Each document is a dictionary
                       that should contain 'title' and 'body_text' keys.

        Returns:
            A list of reranked documents, sorted by their new scores.
        """
        # Create pairs of [query, document_text]
        pairs = []
        for doc in documents:
            text = doc.get('title', '') + " " + doc.get('body_text', '')
            pairs.append([query, text])

        # Predict the scores
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Add scores to documents and sort
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = score
        
        reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked_docs

if __name__ == '__main__':
    from pathlib import Path
    import json
    import torch

    # Proper device selection
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")

    # More realistic example usage
    reranker = Reranker('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

    project_root = Path(__file__).parent.parent.parent
    documents_path = project_root / "data" / "clean" / "blogs.json"

    with open(documents_path, 'r') as f:
        all_documents = json.load(f)
    
    sample_docs = all_documents[:5]

    sample_query = "transformer models for NLP"

    reranked_results = reranker.rerank(sample_query, sample_docs)

    print(f"Query: {sample_query}\n")
    print("Reranked documents:")
    for doc in reranked_results:
        print(f"  - ID: {doc['id']}, Score: {doc['rerank_score']:.4f}, Title: {doc['title']}")
