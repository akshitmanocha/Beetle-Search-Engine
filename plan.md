# Beetle Research Sprint — A Combine-and-Measure Retrieval Study (8–24h)

**Author:** Akshit Manocha (BTech)
**Hardware:** MacBook M4 Pro, 48 GB unified memory, MPS/CPU only (no external GPU, no cloud training)
**Duration:** One focused 8–24 hour sprint
**Outcome:** A bounded, fully-deliverable mini-research project — a reproducible evaluation harness over small public benchmarks, a quality–compute ablation, a set of high-impact engineering fixes, a reachable live demo, and a polished LaTeX paper. Engineered to read as credible to *both* a hiring recruiter and a research professor evaluating an internship application.

> **Thesis (the elevator pitch).** We do **not** invent a new retrieval algorithm. We rigorously **measure and combine** the approaches that already exist in the Beetle codebase — BM25 (lexical) + dense (FAISS) + SPLADE (learned-sparse) + weighted Reciprocal Rank Fusion + cross-encoder reranking — fix the single highest-impact correctness bug (FAISS cosine vs L2), and report a clean **quality–compute ablation** on small, widely-cited public benchmarks, plus a curated domain demo. The deliverable is honest, comparable, and reproducible end-to-end on a laptop. The ambition (novel routing, adaptive reranking, distilled embeddings, large corpora) is preserved as a clearly-labeled **Future Work / Research Roadmap**, not as in-sprint scope.

---

## 1. What Beetle is today

Beetle is a working **hybrid search engine** over curated AI/ML blog content (Distill, The Gradient, Lilian Weng, Karpathy, Anthropic interpretability posts, Sebastian Raschka, etc.). The serving recipe is the modern textbook stack:

```
Query
  ├── BM25        (Whoosh, lexical)
  ├── Dense       (FAISS over google/embeddinggemma-300m, 768-dim)
  └── SPLADE      (naver/splade-cocondenser-ensembledistil, learned sparse)
                │
                ▼
        weighted RRF fusion (k = 60)
                │
                ▼
   optional cross-encoder rerank (Alibaba-NLP/gte-reranker-modernbert-base)
                │
                ▼
        FastAPI /search + static UI
```

A DVC ETL pipeline (crawl → parse → feature-extract → heuristic-label → self-train filter → index) produces the domain corpus `data/clean/blogs.json`. Every retrieval component already exists; the composition is the standard recipe. What the project lacks is **measurement** — there are no benchmark numbers, and a search engine that has never been measured cannot be compared to anything. This sprint fixes exactly that.

The embedding model `google/embeddinggemma-300m` is **gated on Hugging Face**: running dense retrieval requires HF access and a token. This is documented so the omission, if HF access is unavailable in a given environment, is intentional rather than accidental.

---

## 2. Sprint goals and non-goals

### Goals
- Fit **all** work inside an 8–24 hour budget on one M4 Pro (no cloud, no external GPU, no training).
- Produce credible, *comparable* numbers on small public benchmarks: **nDCG@10, MRR@10, Recall@100**.
- Deliver a focused ablation that **combines** the existing retrievers and quantifies each component's contribution, plus the cosine-fix delta.
- Harden the serving path so it is production-shaped (Tier-1 fixes below).
- Ship a real, reachable deployed demo.
- Produce a recruiter- and professor-appealing **LaTeX paper** with metric tables and a Pareto figure, sourced directly from the eval outputs.

### Non-Goals (explicitly CUT from the previous 16-week program → see §9 Future Work)
- ❌ The three "novel contributions": **QARR** (Query-Adaptive Retriever Routing), **CR-CEE** (Cascade Reranking with Confidence-Based Early Exit), **LDDE** (LLM-Distilled Domain Embeddings).
- ❌ Large corpora: **S2ORC** (1–2.5M docs), **arXiv-bulk**, **full BEIR-13**, **LoTTE**.
- ❌ Embedding fine-tuning / contrastive training; synthetic triplet generation; LLM-as-judge labeling; citation-graph retrieval.
- ❌ Tantivy / PISA / seismic migrations, ONNX/CoreML conversion, gRPC, Next.js frontend rebuild.
- ❌ Full observability stack (OpenTelemetry, Prometheus) and CI retrieval-regression gating.

These are real, attractive directions — they are deferred, not abandoned. Keeping them visible in §9 is what lets the sprint read as the credible first chapter of a larger research story.

---

## 3. Datasets

### 3.1 Public benchmarks (for credible, comparable numbers)

Three **small** BEIR datasets that download in minutes and index in minutes on an M4 Pro, all producing numbers directly comparable to published BEIR results:

| Dataset | Corpus size | Queries | Why it's here |
|---|---|---|---|
| **SciFact** | ~5.2k docs | 300 | Small scientific-claim verification set; fast; widely cited. |
| **NFCorpus** | ~3.6k docs | 323 | Tiny medical IR set; very fast; standard BEIR entry; smallest, so it doubles as the smoke-test dataset. |
| **ArguAna** | ~8.7k docs | 1,406 | Argument retrieval; small; complements the above with a different query/relevance shape. |

These three sit comfortably under the "small datasets only" constraint and load via the `beir` package or Hugging Face `datasets`. FiQA (~57k docs) is an **optional 4th dataset** only if time remains in the upper (24h) budget; the default is the three above.

### 3.2 Domain demo corpus (the existing curated AI/ML blogs)

The DVC-produced `data/clean/blogs.json` corpus remains the **live demo** surface — what the deployed UI searches. It is the "yours, not off-the-shelf" element of the project. The demo needs no ground-truth relevance labels; the quantitative story is carried entirely by the BEIR datasets.

---

## 4. The evaluation harness and ablation grid

The harness is the heart of the research deliverable. It computes three metrics across five system configurations on each BEIR dataset, plus two extra ablation axes.

### 4.1 Metrics
- **nDCG@10** — primary quality metric (BEIR standard).
- **MRR@10** — early-precision / first-relevant-result quality.
- **Recall@100** — candidate-set quality before reranking.

### 4.2 The five systems (the ablation grid: 5 systems × 3 datasets)

1. **BM25** only — lexical baseline.
2. **Dense** only — FAISS, with the cosine fix applied.
3. **SPLADE** only — learned sparse.
4. **Hybrid** = weighted RRF(BM25, dense, SPLADE).
5. **Hybrid + Rerank** = system 4 → cross-encoder rerank of the top-k.

### 4.3 Additional ablation axes
- **Fusion-weight sweep**: vary RRF weights `(w_bm25, w_dense, w_splade)` over a small grid (e.g. `(1,1,1), (2,1,1), (1,2,1), (1,1,2), (0,1,1)`) to find the best combination and demonstrate that combining helps.
- **Cosine-vs-L2 delta**: run dense retrieval both ways to quantify the bug fix — a clean, honest "engineering improved a metric" result.
- *(Optional, cheap)* RRF `k`-constant sensitivity over `k ∈ {10, 60, 100}`.

### 4.4 Output artifacts (feed straight into the paper)
- `eval/results/<dataset>_metrics.csv` — one row per system × metric.
- `eval/results/ablation_fusion_weights.csv` — sweep results.
- `eval/figures/pareto.png` — quality (nDCG@10) vs compute (mean query latency / reranker calls).
- `eval/figures/per_dataset_bars.png` — grouped bar chart of nDCG@10 per system per dataset.

### 4.5 Correctness properties (tested with Hypothesis)

Eight universally-quantified properties make the harness trustworthy; they become property-based and example tests:

1. **RRF rank-monotonicity** — within one list, a higher-ranked doc never gets a lower fused score.
2. **RRF weight-zeroing** — a retriever with weight 0 contributes nothing.
3. **Cosine scaling invariance** — scaling a vector by any positive scalar leaves its FAISS cosine ranking unchanged.
4. **nDCG bounds** — nDCG@k ∈ [0, 1]; equals 1 on ideal ordering.
5. **MRR bounds** — MRR@k ∈ [0, 1]; the reciprocal rank of the first relevant doc, else 0.
6. **Recall monotonicity in k** — `recall_at_k` is non-decreasing in k.
7. **Fusion completeness** — every doc appearing in any input list appears in the fused output.
8. **Determinism** — the same system on the same dataset, twice, yields identical rankings and metrics (fixed seeds, no per-query model reloads).

---

## 5. Tier-1 engineering fixes (make serving production-shaped)

These are the highest-impact, lowest-risk fixes to the existing codebase. Together they remove the per-query reloading, fix the dense-retrieval correctness bug, and make the service deployable.

| Fix | What changes | Why it matters |
|---|---|---|
| **Centralized config** (`src/config.py`) | Single source of truth for paths, parsed `params.yaml` (parsed once), and device selection (`mps` on Apple Silicon, else `cpu`). | Removes ~50 lines of duplicated device/param logic across modules. |
| **Startup model loading (Registry)** | Load BM25 index, FAISS index + id map, SPLADE index + doc map, embedding model, SPLADE model + tokenizer, reranker, and the corpus exactly once at FastAPI `lifespan` startup. | Turns O(seconds) first-query latency into O(ms) steady state; no per-request reloads. |
| **FAISS cosine fix** | Build with L2-normalized vectors + `IndexFlatIP` (cosine) instead of `IndexFlatL2`; normalize the query identically. Keep an L2 path *only* for the ablation. | The single highest-impact correctness bug; dense recall now reflects intended embedding semantics. |
| **SPLADE in the hybrid path** | Include SPLADE in the weighted-RRF fusion (it was previously excluded). | The "combine all three" thesis requires all three retrievers actually fused. |
| **In-memory corpus** | Hold `blogs.json` as an id→doc dict in the registry. | Kills the per-query full-file scan. |
| **`/healthz` + `/readyz`** | 200 once the registry is loaded; not-ready (503) otherwise; used as the platform health check. | Deployability + honest readiness signalling. |
| **Structured logging** | Replace ad-hoc prints with structured logs. | Observability without the full OTel/Prometheus stack. |
| **Tightened CORS** | Replace `allow_origins=["*"]` with the deployed frontend origin(s). | Security hygiene a reviewer will check. |
| **Typed result contract** | `/search` returns `doc_id, title, url, snippet, score, source_ranks{bm25,dense,splade}`; reject empty/whitespace queries with a client error. | Clean, defensible API surface. |
| **Pinned dependencies** | Exact versions in `requirements.txt`; all install on Apple Silicon/CPU via prebuilt wheels (Python 3.9+). | Reproducibility — a selling point. |
| **pytest + Hypothesis** | Unit + property-based + one end-to-end API test against a small fixture corpus; runs without large downloads. | Encodes the eight correctness properties (§4.5). |

---

## 6. Time-boxed breakdown (fits 8–24 hours)

The lower bound (~8.5h) covers the three datasets and the core ablation; the upper bound (~24h) adds FiQA, deeper sweeps, a richer paper, and UI polish.

| Block | Work | Lower (≈8h) | Upper (≈24h) |
|---|---|---|---|
| 1 | Tier-1 fixes: `src/config.py`, startup registry, FAISS cosine, SPLADE-in-hybrid, in-memory corpus, `/healthz`, logging, pinned deps | 2.0h | 4.0h |
| 2 | pytest harness + property tests (RRF, metrics, cosine) | 1.0h | 2.5h |
| 3 | BEIR loaders + metrics module (nDCG/MRR/Recall) | 1.0h | 2.5h |
| 4 | Ablation runner + fusion-weight sweep + cosine-vs-L2 | 1.5h | 4.0h |
| 5 | Figures (Pareto, bar charts) + results tables | 0.5h | 2.0h |
| 6 | Deployment (HF Spaces / Docker) + UI method+reranker toggle | 1.0h | 3.0h |
| 7 | LaTeX paper (write-up, tables, figures) | 1.0h | 4.0h |
| 8 | Rewrite `plan.md` + `project.md` | 0.5h | 2.0h |
| **Total** | | **~8.5h** | **~24h** |

**Reproducibility & dependencies.** Reuse the existing DVC pipeline; pin exact versions in `requirements.txt`. Existing: `fastapi`, `uvicorn`, `faiss-cpu`, `sentence-transformers`, `whoosh`, `transformers`, `torch` (MPS), `dvc`, `beautifulsoup4`, `trafilatura`, `readability-lxml`, `pyyaml`. Added for the sprint: `beir` (or direct HuggingFace `datasets` loaders), `hypothesis`, `pytest`, `matplotlib`, `pandas`, and optionally `pytrec_eval` (metrics are otherwise self-implemented and property-tested). All install on Apple Silicon/CPU with prebuilt wheels on Python 3.9+.

---

## 7. Deployment and paper deliverables

### 7.1 Deployment (end-to-end, reachable)
**Goal:** a real URL a recruiter or professor can click.

- **Packaging:** reuse the existing `Dockerfile` / `docker-compose.yaml`; bake the small domain index + `blogs.json` into the image (or pull via DVC at build) so the container is self-contained.
- **Host (recommended default):** **Hugging Face Spaces (Docker)** — single container, free tier, holds the FastAPI app + static UI + small index. (Fly.io / Railway are alternatives if a custom domain is wanted.)
- **Frontend:** keep the existing static `index.html` / `script.js` / `style.css`, served by FastAPI. Add a **method + reranker toggle** so reviewers can try BM25 / dense / SPLADE / hybrid (± rerank) live.
- **Health:** `/healthz` is the platform health check. The demo is read-only public search; no auth is needed and that omission is documented as intentional.

### 7.2 LaTeX paper (`paper/beetle.tex`)
Tracked artifact alongside `paper/figures/` and `paper/references.bib`, in a lightweight short-paper style. Sections:

1. **Abstract** — rigorous comparison + combination of lexical/dense/sparse retrieval with a quality–compute analysis, on small public benchmarks, fully reproducible on a laptop.
2. **Introduction** — the question (which retrieval family wins, and does combining help?); sprint contributions (eval harness + ablation + engineering fixes + deployed demo).
3. **System** — architecture, the three retrieval families, weighted RRF, reranker.
4. **Experimental Setup** — datasets (SciFact/NFCorpus/ArguAna), metrics (nDCG@10, MRR@10, Recall@100), hardware (M4 Pro), reproducibility (DVC, pinned deps).
5. **Results & Ablation** — main metric table (5 systems × 3 datasets), fusion-weight sweep, cosine-vs-L2 delta, the **Pareto figure**, the per-dataset bar chart.
6. **Discussion** — when each retriever wins; whether hybrid dominates; the cost of reranking.
7. **Limitations & Future Work** — explicitly cites the deferred directions in §9 as the research roadmap (signals ambition).
8. **Conclusion** + **References**.

The paper pulls numbers and figures straight from `eval/results/` and `eval/figures/`, so it stays in sync with the actual run.

---

## 8. What "done" looks like (sprint success criteria)

1. `src/config.py` + startup registry + FAISS cosine fix + SPLADE-in-hybrid + in-memory corpus, all serving live.
2. `eval/` harness producing `metrics.csv`, the fusion-weight sweep CSV, the cosine-vs-L2 delta, and both figures, **reproducibly** (determinism property holds).
3. A green `pytest` run including the eight Hypothesis property tests and one end-to-end API test, with no large downloads required.
4. A reachable demo URL (HF Spaces) with the method/reranker toggle.
5. `paper/beetle.tex` compiling, with tables and the Pareto figure sourced from the eval outputs.

### Defensible 30-second pitch
> "Beetle is a hybrid retrieval system — BM25 + dense + SPLADE fused with weighted RRF, then optionally cross-encoder reranked. I built a reproducible evaluation harness over three small BEIR datasets (SciFact, NFCorpus, ArguAna) and ran a clean five-system ablation with nDCG@10 / MRR@10 / Recall@100. I quantified each retriever's contribution, swept the fusion weights, and measured the delta from fixing a FAISS cosine-vs-L2 bug. There's a quality–compute Pareto chart, a live demo, and a short paper — and the whole thing reproduces on a MacBook in under a day."

---

## 9. Future Work / Research Roadmap (NOT in this sprint)

> Everything below is **deferred**. It is kept here, described attractively, so the research arc reads as ambitious — but none of it is committed sprint scope. These are the natural next chapters once the combine-and-measure baseline above exists and is trusted.

### 9.1 Three candidate novel contributions

**A — Query-Adaptive Retriever Routing (QARR).** Hybrid search fires every retriever (BM25, dense, SPLADE) on every query at the same `top_k`, spending ~3× the needed compute on easy queries while under-resourcing hard ones. QARR would train a small classifier (~30–60M params, <5ms target) that emits a per-query *retrieval plan* — which retrievers to fire, at what depth, and how to fuse — learned by distilling an oracle that picks the cheapest retriever subset achieving ≥95% of full-hybrid MRR. *Expected:* 30–50% fewer retriever FLOPs at matched MRR@10. *Read:* query-performance-prediction (Carmel & Yom-Tov QPP survey); adaptive computation (DeeBERT, PABEE, FastBERT); distillation from oracle policies; "retriever selection" / "cascade retrieval" (Mokrii et al.).

**B — Cascade Reranking with Confidence-Based Early Exit (CR-CEE).** Cross-encoder rerankers are 50–100× slower than retrieval yet run on every (query, top-k) pair regardless of how confident the initial ranking already is. CR-CEE would use a three-stage cascade (retrieval → cheap reranker → strong reranker) with learned exit thresholds gated on a retriever-agreement signal (Kendall's τ / RBO) and a stage-2 stability signal. *Expected:* 40–60% fewer reranker FLOPs at matched nDCG@10. *Why novel:* adaptive depth at *cascade* granularity using *retriever-agreement* gates is not published in this form (existing early-exit work operates within a single transformer; cascade-rerank work uses fixed cascades). *Read:* cascade ranking (Wang et al., KDD 2018); monoT5→duoT5 cascades; selective prediction / abstention; RBO vs Kendall's τ; SIGIR ReNeuIR proceedings.

**C — LLM-Distilled Domain Embeddings (LDDE).** Off-the-shelf embedders under-perform on technical long-form content where the meaningful similarity axes are *method / domain / depth*, not surface topic. LDDE would (1) use a frontier LLM as an *annotator that selects hard negatives from real corpus docs* (not generates fake ones) to synthesize ~300k hard triplets, then (2) contrastively fine-tune a small embedder with InfoNCE. *Expected:* +3 to +6 nDCG@10 in-domain; mixed on BEIR (the domain-adaptation tradeoff story). *Why novel:* combines the realism of corpus-mined negatives with the discrimination of LLM judgment. *Read:* InPars, Promptagator, GPL, Doc2Query; hard-negative mining (ANCE, RocketQA, STAR/ADORE); contrastive learning (SimCSE, InfoNCE).

### 9.2 Scaling up the evaluation surface
- **Full BEIR-13** zero-shot subset (TREC-COVID, NQ, HotpotQA, FiQA, Touché-2020, CQADupStack, Quora, DBPedia, SCIDOCS, FEVER, Climate-FEVER, plus the three small sets) — numbers directly comparable to BGE/E5/SPLADE-v3 papers. Gated by SPLADE index build cost on the large datasets (HotpotQA, NQ, DBPedia, FEVER, Climate-FEVER each have millions of docs).
- **LoTTE** (long-tail, topic-stratified) as a held-out out-of-distribution stress test.
- **S2ORC** (~2M cs.CL/cs.LG/cs.AI papers) + **arXiv-bulk** + the curated blogs → a ~1M-doc domain corpus for the in-domain LDDE story. All of these are out of the laptop time/compute budget for this sprint.

### 9.3 Engineering upgrades (post-sprint)
Tantivy for BM25 (Rust, sub-ms queries), FAISS HNSW + IVF-PQ for scale, PISA/seismic mmap-backed SPLADE serving, ONNX/CoreML conversion for Apple-Silicon inference wins, multi-worker uvicorn / gRPC split, a Next.js 15 + shadcn/ui frontend rebuild, full OpenTelemetry + Prometheus observability, and a CI retrieval-regression gate (block a PR if BEIR mean nDCG@10 regresses).

---

## 10. Reading list

Direction, not a syllabus — read abstracts first, then the papers that catch your interest.

### In-sprint foundations (read these for the paper's related work)
- Thakur et al., **"BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation"** (NeurIPS 2021) — the benchmark and uniform loader/evaluator this sprint builds on.
- Robertson & Zaragoza, **"The Probabilistic Relevance Framework: BM25 and Beyond"** (2009) — the lexical baseline.
- Formal et al., **"SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"** (SIGIR 2021) — the learned-sparse retriever.
- Karpukhin et al., **"Dense Passage Retrieval for Open-Domain QA"** (EMNLP 2020) — DPR; the dense-retrieval lineage.
- Cormack et al., **"Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"** (SIGIR 2009) — the RRF fusion this sprint uses (k = 60).

### Future Work directions (read when starting §9)
- Chen et al., **"BGE-M3"** (2024) — strong hybrid baseline and the LoTTE benchmark.
- **QARR direction:** Carmel & Yom-Tov, "Estimating the Query Difficulty for Information Retrieval" (QPP survey, 2010); Xin et al., "DeeBERT" (ACL 2020); Zhou et al., "BERT Loses Patience" (PABEE, NeurIPS 2020).
- **CR-CEE direction:** Wang et al., "Cascade Ranking for Operational E-commerce Search" (KDD 2018); the SIGIR ReNeuIR workshop proceedings.
- **LDDE direction:** Bonifacio et al., "InPars" (2022); Dai et al., "Promptagator" (ICLR 2023); Wang et al., "GPL" (NAACL 2022); Gao et al., "SimCSE"; hard-negative mining (ANCE, RocketQA).
