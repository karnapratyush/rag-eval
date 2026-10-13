# RAG Evaluation

Learning project: a small RAG "doubt solver" over LLM-evals lecture transcripts,
evaluated end-to-end with [DeepEval](https://github.com/confident-ai/deepeval).

## Pipeline (`src/`)

| File | Role |
|------|------|
| `retriever.py` | Load `.vtt` transcripts from `data/`, chunk, embed (`text-embedding-3-large`), persist to Chroma (`chroma_store/`) |
| `reranker.py` | Over-fetch `fetch_k` with the bi-encoder, rerank to `top_k` with a cross-encoder |
| `generator.py` | Faithfulness-first prompt; `generate()` and `generate_stream()` |
| `rag_pipeline.py` | retrieve → rerank → generate; returns `{query, context, answer}` |

## Evals (`evals/`)

| Level | Script | Metrics |
|-------|--------|---------|
| Component: retriever | `eval_retriever.py`, `eval_retriever_with_reranker.py` | Contextual recall / precision |
| Component: generator | `eval_generator.py` | Faithfulness, answer relevancy (ideal context) |
| Pipeline (RAG triad) | `eval_rag_pipeline.py` | Contextual relevancy, faithfulness, answer relevancy |
| Application | `eval_correctness.py`, `eval_application.py` | G-Eval correctness / completeness / style |
| Safety | `eval_scope_safety.py`, `eval_toxicity.py`, `eval_leakage.py` | Scope adherence, toxicity, prompt / content / PII leakage |
| Operational | `evals_latency.py`, `eval_cost.py` | p50/p95/p99 latency + TTFT, tokens × price |

Golden datasets live in `goldens/`. `goldens/golden_via_deep_Eval.py` drafts
synthetic goldens with the DeepEval Synthesizer (review every one before use).

## Setup

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
```

Run from the repo root as modules so `src` imports resolve:

```bash
uv run python -m src.rag_pipeline          # smoke test (builds chroma_store on first run)
uv run python -m evals.eval_rag_pipeline
uv run python -m evals.evals_latency
```
