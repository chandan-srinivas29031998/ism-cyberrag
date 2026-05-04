# Sprint 3 RAG Pipeline Report

## What was built

Sprint 3 adds two features to the retrieval pipeline: multi-query expansion and a two-stage out-of-scope (OOS) guardrail. Both are designed to improve retrieval coverage for in-scope questions while blocking irrelevant queries before they reach the LLM.

## Multi-query expansion

**File:** `src/query_expansion.py`

When a user asks a question, the system generates three alternate phrasings using Groq Llama 3.1 before performing any retrieval. The original question is always kept as variant 0. Each variant (including the original) is embedded and searched independently via `hybrid_search`, then results are merged and deduplicated by chunk ID. The merged pool is reranked against the original question only, not the variants.

The point of this is coverage. A single phrasing might miss relevant chunks because of vocabulary mismatch. For example, "What does the ISM say about two-factor authentication?" and "ISM guidelines for multi-factor authentication" hit different keywords but mean the same thing. By searching all variants, we get a wider candidate pool (typically 15-25 unique chunks instead of 10) before the cross-encoder narrows it back down to the top 5.

**How it works step by step:**

1. The original question is sent to Llama 3.1 with a prompt asking for exactly 3 alternate phrasings, returned as a JSON array.
2. The system builds a list of 4 queries: the original plus the 3 alternates.
3. `multi_query_retrieve()` in `src/retrieval.py` runs `hybrid_search` for each query, collects all results, and removes duplicates by chunk ID.
4. The full merged set goes to the cross-encoder reranker, which scores every chunk against the original question and keeps the top 5.

If the LLM call for expansion fails (bad JSON, timeout), the system falls back to searching with only the original question. This is handled in `expand_query()` with a try/except around the JSON parse.

**Configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTI_QUERY_ENABLED` | `true` | Set to `false` to disable expansion entirely |
| `MULTI_QUERY_COUNT` | `3` | Number of alternate phrasings to generate |

Both are set via environment variables and read in `src/config.py`.


## Two-stage OOS guardrail

**File:** `src/guardrail.py`

Sprint 2 relied entirely on the LLM system prompt to refuse out-of-scope questions. This worked most of the time, but had two problems: it still wasted a Groq API call on obviously irrelevant questions, and the LLM occasionally tried to answer borderline OOS queries instead of refusing. Sprint 3 adds two filtering stages that catch OOS questions before the LLM is called.

**Stage 1: Keyword and intent pre-filter** (runs before embedding)

The `pre_filter()` function checks the question against two lists:

- A deny list of regex patterns for clearly off-topic terms: recipe, stock price, weather, sports scores, horoscope, dating, vendor-specific commands, exploit code, code/script requests, product pricing, platform setup instructions, and similar. If any pattern matches, the question is immediately blocked with a standard refusal message.
- An allow list of security-related signals: ism, encryption, firewall, authentication, mfa, essential eight, asd, acsc, and about 30 others. If any signal is found, the question passes through.
- If neither list matches, the question passes through with an "uncertain" label. The system gives the benefit of the doubt rather than blocking ambiguous queries.

This stage is fast (string matching, no model calls) and catches the most obvious junk before any embedding or search work happens.

**Stage 2: Rerank score threshold** (runs after reranking)

After the cross-encoder scores all candidate chunks, the `rerank_threshold_check()` function looks at the highest rerank score in the result set. If the best-scoring chunk is below the threshold, the question is blocked. The logic is simple: if even the best chunk the retriever found is a poor match, the LLM has nothing useful to work with.

This catches questions that are topically adjacent to security (so they pass the pre-filter) but are not actually covered by the ISM. The threshold is intentionally conservative because Sprint 2 showed a small number of very hard in-scope questions with low raw reranker scores. Known vendor/code/pricing OOS patterns are handled by Stage 1 so the score threshold does not need to block borderline valid ISM questions.

**Configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `OOS_PRE_FILTER_ENABLED` | `true` | Toggle the pre-filter on/off |
| `OOS_RERANK_THRESHOLD` | `-5.0` | Minimum max rerank score to proceed to LLM |


## Updated pipeline flow

The full Sprint 3 pipeline runs in this order:

```
User question
  -> Stage 1: OOS pre-filter (keyword/intent check)
  -> Embed original query
  -> Multi-query expansion (generate 3 alternates via Llama 3.1)
  -> Hybrid search for each variant (BM25 + vector + RRF)
  -> Deduplicate by chunk ID
  -> Cross-encoder reranking against original question (top 5)
  -> Stage 2: Rerank threshold check
  -> LLM generation with top 5 chunks as context
```

If Stage 1 blocks the question, no embedding or search happens. If Stage 2 blocks the question, no LLM call happens. Both stages return the same standard refusal message defined in `guardrail.py`.


## Sprint comparison

Sprint 1: Parse PDFs, fixed-size chunking (1000 chars), embed, vector search (top 5), LLM.

Sprint 2: Parse PDFs, ISM-aware chunking (control boundaries), embed, hybrid search (top 10), cross-encoder rerank (top 5), LLM.

Sprint 3: Parse PDFs, ISM-aware chunking, OOS pre-filter, embed, multi-query expansion, multi-query hybrid search, deduplicate, cross-encoder rerank (top 5), rerank threshold check, LLM.


## Sprint 3 metric targets

| Metric | Sprint 2 Actual | Sprint 3 Target |
|--------|----------------|----------------|
| Faithfulness | — | > 0.78 |
| Answer Relevancy | — | > 0.82 |
| Context Precision | — | > 0.85 |
| Context Recall | — | > 0.91 |
| Answer Similarity | — | > 0.93 |

The targets reflect the combined effect of multi-query expansion (better retrieval coverage should improve context recall and precision) and the OOS guardrail (blocking irrelevant queries should reduce the drag that OOS questions had on overall scores in Sprint 2).


## Files changed or added

```
Added:
  src/query_expansion.py   - multi-query expansion via Groq Llama 3.1
  src/guardrail.py         - pre-filter + rerank threshold guardrail

Modified:
  src/config.py            - Sprint 3 parameters (MULTI_QUERY_*, OOS_*)
  src/retrieval.py         - multi_query_retrieve() function
  app/routes.py            - pipeline uses expansion + guardrail stages
```
