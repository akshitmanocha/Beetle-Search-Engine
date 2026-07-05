# Beetle Search Engine — Project Notes & Improvement Plan

A working notebook for understanding the current system and planning targeted improvements.

> **Status (re-anchored for the Beetle Research Sprint).** This document has been
> re-scoped to the **8–24 hour research sprint**. The committed deliverable is a focused set
> of **Tier-1 engineering fixes plus an evaluation harness** (BEIR loaders, metrics, ablation
> runner, figures). Everything beyond that — the larger retrieval-quality work, ETL
> robustness, product-surface features, and stretch bets — is explicitly marked **post-sprint /
> future work**. The pain-points catalog below (R# / E# / S# / C#) is preserved in full and
> kept valuable; items the sprint closes are tagged **✅ in sprint scope**.

---

## 1. What Beetle is today

A niche search engine over high-quality AI/ML blog content. The pipeline is:

```
seed domains  →  crawl  →  download HTML  →  parse + extract features
                                                    │
                                                    ▼
                          weak heuristic labels  →  TF-IDF + LogReg  →  strong labels
                                                    │
                                                    ▼
                                         filter to "blog" docs only
                                                    │
                          ┌─────────────────────────┼─────────────────────────┐
                          ▼                         ▼                         ▼
                       BM25                       FAISS                     SPLADE
                    (Whoosh)              (SentenceTransformer)        (sparse MLM)
                          └─────────────────────────┴─────────────────────────┘
                                                    │
                                          Hybrid search (RRF)
                                                    │
                                            Optional reranker
                                       (CrossEncoder, gte-reranker-modernbert)
                                                    │
                                          FastAPI /search  +  static UI
```

**Stack:** Python, FastAPI, DVC, Whoosh (BM25), FAISS (dense), SPLADE
(`naver/splade-cocondenser-ensembledistil`), SentenceTransformers embedder
(`google/embeddinggemma-300m` — **gated on Hugging Face**; running dense retrieval requires HF
access + a token), CrossEncoder reranker (`Alibaba-NLP/gte-reranker-modernbert-base`),
BeautifulSoup / trafilatura / readability-lxml for ETL.

**Sprint evaluation surface:** small BEIR datasets — **SciFact**, **NFCorpus**, **ArguAna** —
for credible, comparable numbers (nDCG@10, MRR@10, Recall@100). The **live demo** searches a
BEIR dataset corpus; rebuilding the curated AI/ML blog corpus for the demo is **deferred to
post-sprint**.

---

## 2. Strengths worth preserving

- **DVC-backed reproducibility.** Every pipeline stage has clear deps/outs; rebuilds are cheap.
- **Three retrieval families covered.** Lexical (BM25), dense (FAISS), and learned-sparse (SPLADE) — a solid foundation for IR experimentation.
- **Hybrid via RRF.** Simple, robust, no tuning required.
- **Two-stage classifier (heuristic → TF-IDF self-training).** Pragmatic for a small project — gets reasonable labels without manual annotation.
- **Clean separation of concerns** (`ETL/`, `index/`, `search/`, `models/`, `serving/`).
- **Containerized + static frontend** — actually runnable end-to-end.

---

## 3. Pain points & limitations (observed in code)

This catalog is preserved as the project's running record of known issues. Items the sprint
closes are tagged **✅ in sprint scope**; the rest are **post-sprint**.

### 3.1 Retrieval / search quality

| # | Issue | Where | Why it matters | Sprint status |
|---|-------|-------|----------------|---------------|
| R1 | **FAISS uses L2 distance on embeddings that aren't normalized** | `src/index/build_faiss.py:30` | SentenceTransformer embeddings should use cosine similarity (`IndexFlatIP` on normalized vectors). L2 silently degrades dense recall. | ✅ **in sprint scope** — normalize + `IndexFlatIP` (cosine); cosine-vs-L2 delta reported in the ablation. |
| R2 | **SPLADE inverted index loaded from JSON on every query** | `src/search/search_splade.py:54` | Cold-start I/O on every request; not viable past a few thousand docs. | ✅ **in sprint scope** — loaded once into the startup registry. |
| R3 | **Hybrid only fuses BM25 + FAISS — SPLADE is excluded from the hybrid path** | `src/search/hybrid_search.py:60` | Hybrid mode never benefits from learned sparse retrieval despite the index existing. | ✅ **in sprint scope** — SPLADE included in weighted RRF fusion. |
| R4 | **Re-embeds the query for FAISS by calling `generate_embeddings` (which loads the full SentenceTransformer model from scratch)** | `src/search/search_faiss.py:37` | Every search reloads a 300M-param model. Latency is dominated by model init, not search. | ✅ **in sprint scope** — embed model loaded once at startup; reused per query. |
| R5 | **`search_faiss` returns L2 distances (lower=better) but `reciprocal_rank_fusion` only uses rank, not score** | OK for RRF, but if scores are ever surfaced they're inconsistent | Reranker workflow mixes "search_score" semantics across methods. | Partially addressed — cosine fix makes dense scores higher-is-better; full score-semantics cleanup is **post-sprint**. |
| R6 | **No deduplication** by URL canonicalization across BM25/FAISS/SPLADE results | `main.py:72` | Different surface URLs of the same doc could appear twice. | **Post-sprint.** |
| R7 | **Reranker concatenates `title + body_text`** without truncation | `src/models/reranker.py:23` | CrossEncoders have ~512 token context; long blog bodies silently get truncated to head only. Section-aware chunking would rerank more fairly. | ✅ **in sprint scope** (truncation only) — inputs truncated to `max_chars`; section-aware chunking remains post-sprint. |
| R8 | **No evaluation harness.** No held-out queries, no MRR/nDCG, no regression tests for retrieval. | — | "Did my change help or hurt?" is currently unanswerable. | ✅ **in sprint scope** — this is the core deliverable: BEIR loaders + metrics + ablation runner + figures. |

### 3.2 ETL / data quality

| # | Issue | Where | Why it matters | Sprint status |
|---|-------|-------|----------------|---------------|
| E1 | **Crawler ignores `robots.txt`** | `src/ETL/website_crawler.py` | Politeness + legal — also marks the project as not-for-production. | **Post-sprint.** |
| E2 | **No throttling per host** beyond a global `sleep_time` (defaulted to 0) | same | Easy way to get IP-blocked or hammer small sites. | **Post-sprint.** |
| E3 | **Crawler uses `BeautifulSoup` + `requests` only — no JS rendering** | same | Modern blogs (Substack, Medium, Notion-backed) often need rendering. Many pages will return shells with no body. | **Post-sprint.** |
| E4 | **Sequential crawl per seed**, parallel only on download | `website_crawler.py` vs `download_html.py` | Crawl phase dominates wall time. | **Post-sprint.** |
| E5 | **`parsed.json` is a single monolithic file** loaded in full by `train_tfidf`, `filter_blogs`, embed, etc. | `src/ETL/parse.py:248` | OOM as corpus grows; no streaming. | **Post-sprint.** |
| E6 | **`get_docs_from_json` in `main.py:13` re-reads the entire `blogs.json` for every search request** | `main.py:13` | Effectively scans the whole corpus per query — biggest latency offender for the API. | ✅ **in sprint scope** — corpus held in memory as `dict[id → doc]` in the registry; no per-query rescan. |
| E7 | **Heuristic labels: `score_threshold: 8` but features only sum to ~15 max with current weights and the "-20 too short" punishment** is mislabeled as "-3" | `heuristic_label.py:90` | Reasoning string is wrong; behavior is fine but confusing. The thresholds and weights also haven't been tuned against any ground truth. | **Post-sprint.** |
| E8 | **Self-training loop is one-shot.** TF-IDF predictions overwrite weak labels without confidence thresholding. | `src/models/train_tfidf.py:74` | Borderline cases get arbitrary flips; no semi-supervised stability. | **Post-sprint.** |
| E9 | **No language detection.** Non-English pages flow through everything. | parse stage | Pollutes the index. | **Post-sprint.** |
| E10 | **No near-duplicate detection.** Crossposts and RSS mirrors will appear repeatedly. | — | Hurts result diversity. | **Post-sprint.** |

### 3.3 Serving / API / UX

| # | Issue | Where | Why it matters | Sprint status |
|---|-------|-------|----------------|---------------|
| S1 | **CORS is `allow_origins=["*"]`** | `app.py:14` | Fine locally; not production-safe. | ✅ **in sprint scope** — tightened to the deployed frontend origin(s). |
| S2 | **Models loaded on first query, not at app startup.** Reranker re-instantiated per request. | `main.py:81` | First-query latency is many seconds; reranker reload wastes RAM/time. | ✅ **in sprint scope** — all models/indexes loaded once via a startup registry. |
| S3 | **No request validation beyond empty string**, no rate limiting, no auth | `app.py:50` | API is unhardened. | **Post-sprint** (max-query-length / empty-query checks may be tightened opportunistically; rate limiting + auth deferred). |
| S4 | **Frontend has no method/reranker toggle**, only reads from `/config` | `static/index.html` | Users can't experiment with the search modes that the backend already supports. | **Post-sprint** (a minimal toggle may ship with the demo, but it is not a committed Tier-1 item). |
| S5 | **No snippet/highlight in results** — only first 350 chars of body | `static/script.js:147` | Looks like "summary" rather than "evidence-of-match." | **Post-sprint.** |
| S6 | **No pagination** | both | `top_k`/`rerank_k` cap is hard. | **Post-sprint.** |
| S7 | **No query suggestions, query history, or "did you mean"** | — | Standard search UX missing. | **Post-sprint.** |
| S8 | **`SearchResponse.results: list`** is untyped — clients see whatever shape the backend ships | `app.py:38` | Brittle contract. | ✅ **in sprint scope** — typed `SearchResultItem` contract. |
| S9 | **No `/healthz` or `/readyz`** | — | Container orchestration unfriendly. | ✅ **in sprint scope** — both endpoints added; `/healthz` used as the platform health check. |
| S10 | **No structured logging or request IDs** | — | Debugging production issues will be painful. | ✅ **in sprint scope** — structured logging added. |

### 3.4 Code / project hygiene

| # | Issue | Where | Sprint status |
|---|-------|-------|---------------|
| C1 | Device-selection block (`cuda → mps → cpu`) is duplicated in 5+ files | `main.py`, `hybrid_search.py`, `embed.py`, `search_faiss.py`, `search_splade.py`, `reranker.py` | ✅ **in sprint scope** — centralized in `src/config.py` (`device()`). |
| C2 | `params.yaml` is loaded in 7+ different places, each parsing it independently | everywhere | ✅ **in sprint scope** — parsed once in `src/config.py`. |
| C3 | `project_root = Path(__file__).parent.parent.parent` (or `.parents[2]`) sprinkled throughout | everywhere — should be a single config module | ✅ **in sprint scope** — single `PROJECT_ROOT` in `src/config.py`. |
| C4 | `src/utils/storage.py` and `src/agents/__init__.py` exist but are empty/unused | dead code | **Post-sprint.** |
| C5 | No `pyproject.toml` / no pinned versions in `requirements.txt` | `requirements.txt` | ✅ **in sprint scope** — dependencies pinned to exact versions. |
| C6 | No tests at all (`pytest`, fixtures, sample data) | — | ✅ **in sprint scope** — pytest + Hypothesis property tests + fixture corpus. |
| C7 | No CI (GitHub Actions for lint/test/build) | — | **Post-sprint** (CI retrieval-regression gating explicitly cut). |
| C8 | `Dockerfile` uses Python 3.13 but `README.md` says 3.8+; also uses `pip` (not Poetry as README claims) | `Dockerfile`, `README.md` | **Post-sprint** (README cleanup is opportunistic, not a committed Tier-1 item). |
| C9 | `print()` everywhere — should be `logging` | most modules | Partially addressed — serving path moves to structured logging (S10); full sweep is **post-sprint**. |
| C10 | `train_tfidf.py` downloads NLTK data at import time | `train_tfidf.py:14` — flaky in offline/CI environments | **Post-sprint.** |
| C11 | `embed.py` is imported by `search_faiss.py` solely to embed a single query — heavy coupling | `search_faiss.py:11` | ✅ **in sprint scope** — search reuses the startup-loaded embed model from the registry. |
| C12 | SPLADE module-level `tokenizer`/`model` initialization in `build_splade.py:10-12` runs even on import | startup cost | **Post-sprint** (serving-side SPLADE load is fixed via the registry; the build-time import cost is deferred). |

---

## 4. Improvement plan — sprint scope vs. post-sprint

The plan is now split into **(A) committed sprint scope** and **(B) post-sprint / future work**.
The committed scope is what the Beetle Research Sprint delivers; everything in section B is
deferred and documented so the roadmap still reads as ambitious without bloating the sprint.

### A. Committed sprint scope (Tier-1 fixes + evaluation harness)

This is the full deliverable. Two strands: **(A1)** make the service production-shaped, and
**(A2)** build the evaluation/ablation layer that is the research deliverable.

#### A1 — Tier-1 engineering fixes

1. **Centralize config + paths** → new `src/config.py` exposing `PROJECT_ROOT`, parsed
   `params`, and `device()`. Removes ~50 lines of duplication (closes **C1**, **C2**, **C3**).
2. **Startup model/index loading (registry).** Load every heavy artifact exactly once at
   FastAPI startup (via `lifespan`): SentenceTransformer embed model, SPLADE
   tokenizer + model, CrossEncoder reranker, BM25 index, FAISS index + doc-id map, SPLADE
   inverted index + doc map, and **`blogs.json` as an in-memory `dict[id → doc]`**. First-query
   latency drops from O(seconds) to O(ms) (closes **S2**, **R2**, **R4**, **E6**, **C11**).
3. **FAISS cosine fix.** Normalize embeddings on insert (`faiss.normalize_L2`) and use
   `IndexFlatIP`; normalize the query to match. Highest-impact correctness bug (closes **R1**).
4. **SPLADE included in the hybrid fusion path.** Weighted RRF over
   `[BM25, dense, SPLADE]` with optional per-retriever weights (closes **R3**).
5. **Reranker truncation.** Truncate `title + body_text` to `max_chars` before scoring, and
   reuse the startup-loaded CrossEncoder (closes the truncation half of **R7**; full closure of **S2**).
6. **Typed API contract.** `SearchResultItem` (doc_id, title, url, snippet, score,
   per-source ranks) replaces the untyped `results: list` (closes **S8**).
7. **`/healthz` + `/readyz`.** 200 once the registry is loaded; 503 while unhealthy.
   `/healthz` doubles as the deploy platform's health check (closes **S9**).
8. **Structured logging.** `logging` + JSON formatter on the serving path (closes **S10**).
9. **Tighten CORS.** Replace `allow_origins=["*"]` with the deployed frontend origin(s)
   (closes **S1**).
10. **Pin dependencies.** Exact versions in `requirements.txt`, all available for Apple
    Silicon / CPU (closes **C5**).
11. **pytest + Hypothesis property tests.** Fixture corpus (5–10 docs) plus unit + property
    tests for RRF (weight-zeroing, completeness, rank-monotonicity), the metrics
    (nDCG/MRR/Recall bounds and hand-computed cases), FAISS cosine scaling-invariance, registry
    build, and an end-to-end `/search` test via `TestClient` (closes **C6**).

#### A2 — Evaluation harness (the research deliverable)

12. **BEIR data loaders.** Download/load **SciFact**, **NFCorpus**, **ArguAna** into a uniform
    `(corpus, queries, qrels)` shape.
13. **Metrics module.** nDCG@10 (primary), MRR@10, Recall@100, with mean aggregation over
    queries.
14. **Ablation runner.** Evaluate five systems — BM25, dense, SPLADE, hybrid (weighted RRF),
    hybrid + rerank — across the three datasets, plus a **fusion-weight sweep** and the
    **cosine-vs-L2 delta**. Emits results tables (CSV/JSON).
15. **Figures.** Pareto chart (quality vs. compute) and per-dataset grouped bar charts of
    nDCG@10, written to `eval/figures/` and feeding the LaTeX paper.

Together A1 + A2 close: **R1, R2, R3, R4, R7 (truncation), R8, E6, S1, S2, S8, S9, S10, C1, C2,
C3, C5, C6, C11.**

### B. Post-sprint / future work (deferred)

Everything below is intentionally **out of the sprint's committed scope**. It is kept here as
the roadmap; tackle it only after the sprint deliverable lands.

#### B1 — Retrieval quality (beyond the cosine + SPLADE fixes)

- **Score-semantics cleanup** so surfaced scores are consistent across methods (**R5**).
- **URL canonicalization + result dedup** across retrievers (**R6**).
- **Section-aware chunking** before embedding/reranking: split body on `\n\n`/headings, embed
  chunks, score doc = max(chunk_scores), feed the top chunk to the CrossEncoder (rest of **R7**).
- **Switch FAISS to `IndexHNSWFlat`** once the corpus exceeds ~10k docs (sub-linear queries,
  negligible recall loss).
- **Snippet highlighting** via Whoosh `Hit.highlights("body_text")`, surfaced to the frontend
  (**R5/S5** overlap).

#### B2 — ETL robustness

- **Respect `robots.txt`** + per-host token-bucket throttling (**E1**, **E2**;
  `urllib.robotparser` is stdlib).
- **Playwright/Selenium fallback** for JS-rendered pages, gated behind a `--render` flag (**E3**).
- **Parallelize the crawl phase** (**E4**).
- **Convert `parsed.json` → JSONL** so downstream stages can stream (**E5**).
- **Language detection** to drop non-English at parse time (**E9**).
- **Near-duplicate detection** via MinHash/SimHash on shingles (**E10**).
- **Improve labeling:** tune heuristic weights against a small hand-labeled set, fix the
  misleading reasoning string (**E7**), add a confidence threshold to `train_tfidf` (only flip
  labels when `predict_proba > 0.8`) (**E8**), and optionally an LLM-as-judge pass on borderline
  cases.

#### B3 — Product surface

- **Frontend method/reranker toggle** — dropdown for `bm25 / faiss / splade / hybrid`, checkbox
  for reranker (**S4**). *(A minimal version may ship alongside the demo, but it is not a committed
  Tier-1 item.)*
- **Pagination + result-count selector** (**S6**).
- **Query suggestions / autocomplete** off a prefix index over titles + a query log (**S7**).
- **"Why this result?"** affordance — surface RRF rank + per-source score breakdown.
- **Filter facets** by domain, year, `has_code_blocks`, `has_arxiv_citation` (features already
  extracted, unused at query time).
- **Query embedding LRU cache** to avoid re-encoding identical queries.
- **Request validation, rate limiting, auth** (**S3**).

#### B4 — Hygiene / infra (deferred)

- **Remove dead code** (`src/utils/storage.py`, `src/agents/__init__.py`) (**C4**).
- **CI** (GitHub Actions for lint/test/build) and **retrieval-regression gating** (**C7** —
  explicitly cut from the sprint).
- **Align `Dockerfile` / `README.md`** (Python version, pip vs. Poetry) (**C8**).
- **Full `print()` → `logging` sweep** beyond the serving path (**C9**).
- **Defer NLTK data download** out of import time in `train_tfidf.py` (**C10**).
- **Lazy SPLADE build-time model init** in `build_splade.py` (**C12**).

#### B5 — Stretch / interesting bets

- **LLM query rewrite** ("transformer NLP" → "transformer architectures for natural language
  processing"), feature-flagged and measured with the eval harness.
- **Query expansion via pseudo-relevance feedback** on BM25.
- **"Save / read later" bookmarks** (client-side or a tiny SQLite layer).
- **Incremental indexing** — incremental Whoosh writer; FAISS `IndexIVFFlat` with
  `add_with_ids` deltas; incremental SPLADE inverted-index merge.
- **Next.js frontend rebuild** for a portfolio-grade UI.
- **Rebuild the curated AI/ML blog corpus** for the live demo (currently the demo searches a
  BEIR dataset corpus).

---

## 5. Sprint deliverable summary (what "done" looks like)

The sprint is complete when:

1. `src/config.py` centralizes paths/params/device.
2. The FastAPI app loads all models/indexes once at startup (registry) and holds the corpus in
   memory.
3. FAISS uses cosine (normalized `IndexFlatIP`); SPLADE is in the weighted RRF hybrid path; the
   reranker reuses the loaded model and truncates inputs.
4. `/healthz` + `/readyz`, structured logging, tightened CORS, typed results, and pinned
   dependencies are in place.
5. `pytest` + Hypothesis property tests pass against a fixture corpus and a `/search` smoke test.
6. The evaluation harness (BEIR loaders + metrics + ablation runner) produces results tables and
   the Pareto + bar-chart figures over SciFact / NFCorpus / ArguAna.
7. A reachable end-to-end demo is deployed (searching a BEIR dataset corpus).

Everything in **section B** is out of scope for this sprint and tracked as future work.

---

## 6. Open questions (worth deciding before investing post-sprint)

These guide the **post-sprint** roadmap; they do not affect the committed sprint scope.

- **Scale target.** 5k docs (current seed list × 5k cap)? 50k? 500k? Index choices change a lot
  at each scale (drives B1's HNSW/IVF decisions).
- **Freshness target.** Daily recrawl? Real-time? Drives whether incremental indexing (B5)
  matters.
- **Who's the user?** ML researchers (favor citations, code, technical depth) vs. ML
  practitioners (favor tutorials, recency)? Reranker training and heuristic weights should
  reflect this (drives B2's labeling work).
- **Self-hostable or hosted?** Hosted means auth, rate limits (B3), and an embedding model that
  fits the budget — and HF access for the gated `embeddinggemma-300m` embedder.
- **Product, portfolio piece, or research playground?** Each leads to a different ordering of
  the section-B tiers.
