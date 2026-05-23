# Sprint 3 Web Application

## Overview

The Sprint 3 web app keeps the same FastAPI + vanilla HTML/CSS/JS stack from Sprint 2 but reorganises the interface into three tabs and adds two new pages: a live pipeline explorer and an evaluations dashboard. The single-page chat UI from Sprint 2 is now the "Search ISM" tab.

Templates live in `app/templates/` and static assets in `app/static/`.


## Three-tab navigation

The top navigation bar now has three tabs:

1. **Search ISM** -- the main search interface (was the only page in Sprint 2).
2. **Pipeline Explorer** -- live visualisation of every pipeline step for a given query.
3. **Evaluations** -- static dashboard showing RAGAS metrics across all three sprints.

Each tab is a separate HTML template served by FastAPI: `index.html`, `pipeline.html`, `evaluations.html`.


## Search ISM tab

This is the same split-screen layout from Sprint 2 (question input on the left, reference documents on the right) with two additions:

**Query expansion display.** When multi-query expansion is enabled, the response includes the list of generated query variants. The UI shows these in a collapsible section below the answer. The original question is labelled "Original" and each alternate phrasing is numbered. Users can expand this to see what the system searched for, which adds transparency.

**Guardrail badge.** If a question is blocked by either the pre-filter or the rerank threshold guardrail, the answer area shows a badge indicating which stage caught it ("Blocked by pre-filter" or "Blocked by rerank threshold") alongside the standard refusal message. No reference documents are shown because the pipeline did not proceed to retrieval (pre-filter) or generation (rerank threshold).


## Pipeline Explorer tab

This is the main Sprint 3 UI addition. It lets users type a question and watch each pipeline step execute in real time, with timing, status, and detailed output for every stage.

**How it works technically:**

The frontend sends a POST request to `/pipeline/stream` with the question. The backend processes the full pipeline and emits Server-Sent Events (SSE) as each step completes. The frontend uses `fetch()` with a streaming `ReadableStream` reader so it can send the question in the POST body and parse the SSE frames as they arrive.

Each SSE event has a type (matching the pipeline step) and a JSON data payload containing the step name, execution time in milliseconds, and step-specific output.

**The 7 pipeline steps rendered as cards:**

1. **OOS Pre-Filter** -- Shows whether the question passed or was blocked, and the classification reason (off_topic, topic_match, or uncertain). If blocked, the pipeline stops here and the done event fires immediately.

2. **Query Embedding** -- Shows the embedding model (nomic-embed-text-v1.5), dimension (768), and the search_query prefix used.

3. **Multi-Query Expansion** -- Shows whether expansion is enabled, the original question, and the generated alternate phrasings. If expansion is disabled, shows a note that only the original query was used.

4. **Hybrid Search** -- Shows the search method (BM25 + vector + RRF), how many queries were used, the RRF k value, and a preview of each candidate chunk with its control ID, category, similarity score, and RRF score.

5. **Cross-Encoder Reranking** -- Shows the reranker model, input/output counts (e.g. 20 in, 5 out), and each reranked chunk with its score and content preview.

6. **OOS Rerank Guardrail** -- Shows the max rerank score, the threshold, and whether the question passed. If blocked, the pipeline stops and the done event includes the refusal message.

7. **LLM Generation** -- Shows the model (llama-3.1-8b-instant), provider (groq), number of context chunks sent, a preview of each context chunk, and the full generated answer in a collapsible section.

Each card appears with an animation as its SSE event arrives. Cards show a status indicator (pass/fail), the execution time, and an expandable detail section. The final "done" event includes the total pipeline time.

**Why SSE instead of polling or WebSockets:**

SSE is one-directional (server to client), which is exactly what we need here. The client sends one request and the server streams back events as processing happens. It works over standard HTTP, does not require a WebSocket upgrade, and can be consumed from the streamed response body. For a sequential pipeline where each step depends on the previous one, SSE framing is simpler than WebSockets and more responsive than polling.


## Evaluations tab

This page is a static dashboard that presents RAGAS evaluation results across all three sprints. It was hidden in Sprint 2 (the nav link was disabled) and is now fully built out.

The page has three sections:

**Sprint comparison table.** A table comparing all five RAGAS metrics (faithfulness, answer relevancy, context precision, context recall, answer similarity) across Sprint 1, Sprint 2, and Sprint 3. Each cell shows the actual score, with colour coding to indicate improvement or regression.

**"What We Built" sprint cards.** Three cards summarising the key pipeline additions in each sprint. Sprint 1: baseline RAG with vector search. Sprint 2: ISM-aware chunking, hybrid search, cross-encoder reranking. Sprint 3: multi-query expansion, OOS guardrail. These give viewers context for why the metrics changed between sprints.

**Chart grids.** Evaluation chart images generated by the Sprint 2 and Sprint 3 notebooks. These are static PNGs stored in `app/static/evaluations/` and displayed in a grid layout. The images are copied from `evaluations/sprint-2/` and `evaluations/sprint-3/` during deployment (see the deployment guide for details).


## Technical notes

The frontend uses no JavaScript frameworks. All three pages use vanilla JS with `fetch` for the chat endpoint and `fetch()` plus a `ReadableStream` reader for the pipeline stream. Styling is vanilla CSS in `app/static/style.css`. Templates are Jinja2, rendered by FastAPI.

The backend routes are defined in `app/routes.py`. The `/chat` endpoint returns a JSON response. The `/pipeline/stream` endpoint returns a `StreamingResponse` with `text/event-stream` content type.
