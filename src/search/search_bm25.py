from whoosh.qparser import MultifieldParser, OrGroup


def search_bm25_registry(query: str, registry, top_k: int = 10):
    """Search the registry's startup-loaded Whoosh index (no per-query open_dir).

    Uses OR-grouping so natural-language BEIR queries match documents containing
    ANY query term (the Whoosh default is conjunctive/AND, which cripples recall
    on multi-word queries). ``OrGroup.factory(0.9)`` additionally rewards docs
    matching more terms. Returns up to ``top_k`` ``(doc_id, score)`` pairs; an
    unparseable or empty query yields an empty list rather than raising.
    """
    ix = registry.bm25_index
    with ix.searcher() as searcher:
        parser = MultifieldParser(
            ["title", "body_text"], schema=ix.schema, group=OrGroup.factory(0.9)
        )
        try:
            query = parser.parse(query)
        except Exception:
            return []
        results = searcher.search(query, limit=top_k)
        return [(hit["id"], hit.score) for hit in results]
