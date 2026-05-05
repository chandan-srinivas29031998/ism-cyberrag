---
title: ISM CyberRAG
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ISM-CyberRAG

A Retrieval-Augmented Generation (RAG) system for querying the Australian Information Security Manual (ISM) using natural language.

Built as a university capstone project across three development sprints for AI Studio (University of Technology Sydney).

## Project Structure

```
ism-cyberrag/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── routes.py            # API endpoints (/chat, /pipeline/stream)
│   ├── static/              # CSS and evaluation chart images
│   └── templates/           # Jinja2 HTML templates (search, pipeline, evaluations)
├── src/
│   ├── config.py            # Environment variables and settings
│   ├── parse_pdf.py         # PDF text extraction (pypdf)
│   ├── chunking.py          # ISM-aware chunking (control boundaries)
│   ├── embeddings.py        # Embedding generation (nomic-embed-text-v1.5)
│   ├── supabase_utils.py    # Supabase client and database helpers
│   ├── retrieval.py         # Hybrid search (BM25 + vector + RRF) and multi-query retrieval
│   ├── reranking.py         # Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
│   ├── query_expansion.py   # Multi-query expansion via LLM
│   ├── guardrail.py         # Two-stage OOS guardrail (pre-filter + rerank threshold)
│   ├── llm.py               # Answer generation via Groq (Llama 3.1 8B)
│   └── evaluation.py        # RAGAS evaluation + ClearML logging
├── data/                    # ISM PDF documents (01-25)
├── evaluations/
│   ├── eval_questions.json  # 100 Q&A pairs for RAGAS evaluation
│   ├── sprint-1/            # Sprint 1 evaluation results
│   ├── sprint-2/            # Sprint 2 evaluation results and charts
│   └── sprint-3/            # Sprint 3 evaluation results and charts
├── notebooks/
│   ├── sprint1_poc.ipynb
│   ├── sprint2_development.ipynb
│   └── sprint3_development.ipynb
├── docs/
│   ├── sprint-2/
│   └── sprint-3/            # Pipeline report, frontend docs, deployment guide, techniques
├── .github/workflows/
│   ├── ci.yml               # Lint + smoke test + Docker build on every push/PR
│   └── deploy.yml           # Auto-deploy to HF Spaces on merge to main
├── Dockerfile               # Docker image for HF Spaces deployment
├── requirements.txt
└── .env.example
```

## Tech Stack

| Component | Tool |
|-----------|------|
| PDF Parsing | pypdf |
| Chunking | ISM-aware (control boundary detection) |
| Embeddings | nomic-ai/nomic-embed-text-v1.5 (768-dim) |
| Vector Database | Supabase + pgvector (HNSW + GIN indexes) |
| Search | Hybrid: BM25 full-text + vector similarity + Reciprocal Rank Fusion |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Query Expansion | Multi-query via Llama 3.1 (3 alternate phrasings) |
| OOS Guardrail | Two-stage: keyword pre-filter + rerank score threshold |
| LLM | Llama 3.1 8B via Groq API |
| Web App | FastAPI + Jinja2 templates |
| Evaluation | RAGAS (5 metrics) |
| Experiment Tracking | ClearML |
| CI/CD | GitHub Actions (lint, smoke test, Docker build, HF Spaces deploy) |
| Deployment | Hugging Face Spaces (Docker) |

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/studiobuilders/ism-cyberrag.git
cd ism-cyberrag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your actual keys
```

Required:

| Variable | Source |
|----------|--------|
| `SUPABASE_URL` | Supabase project settings, API |
| `SUPABASE_PUBLISHABLE_KEY` | Supabase project settings, API (anon key) |
| `GROQ_API_KEY` | https://console.groq.com/keys |

Supabase is used as a read-only vector store at runtime. RLS should be enabled on `documents` and `chunks` with read-only `SELECT` policies for `anon` and `authenticated`; see `database/sprint3_rls.sql`.

Optional (for evaluation and experiment tracking):

| Variable | Default |
|----------|---------|
| `EVAL_LLM_PROVIDER` | `ollama` |
| `EVAL_LLM_MODEL` | `llama3.1:8b` |
| `CLEARML_API_ACCESS_KEY` | (from ClearML dashboard) |
| `CLEARML_API_SECRET_KEY` | (from ClearML dashboard) |

See `.env.example` for the full list including Sprint 3 parameters (multi-query, OOS threshold).

### 3. Run the web app

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000. Three tabs are available:

- **Search ISM**: Ask questions about the ISM, see retrieved chunks and answers with citations.
- **Pipeline Explorer**: Enter a question and watch each pipeline stage execute in real time (SSE streaming).
- **Evaluations**: RAGAS metrics across all three sprints, chart images from evaluation notebooks.

### 4. Run the evaluation notebook

Open `notebooks/sprint3_development.ipynb` in Jupyter, Google Colab, or AWS SageMaker. The notebook runs the full evaluation pipeline (100 questions), computes RAGAS scores, generates charts, and logs everything to ClearML.

## Pipeline (Sprint 3)

```
User question
  -> Stage 1: OOS pre-filter (keyword/regex deny list + allow list)
  -> Embed original query (nomic-embed-text-v1.5)
  -> Multi-query expansion (3 alternate phrasings via Llama 3.1)
  -> Hybrid search for each variant (BM25 + vector + RRF)
  -> Deduplicate by chunk ID
  -> Cross-encoder reranking against original question (top 5)
  -> Stage 2: Rerank threshold check (max score >= -5.0)
  -> LLM generation with top 5 chunks as context (Llama 3.1 8B)
```

## Sprint Progression

| Sprint | What was added |
|--------|---------------|
| Sprint 1 | Baseline RAG: fixed-size chunking (1000 char), vector-only search, Llama 3.1 via Groq |
| Sprint 2 | ISM-aware chunking (643 chunks), hybrid search (BM25 + vector + RRF), cross-encoder reranking, FastAPI web app |
| Sprint 3 | Multi-query expansion, two-stage OOS guardrail, pipeline explorer, evaluations dashboard, CI/CD, HF Spaces deployment |

## CI/CD

Two GitHub Actions workflows in `.github/workflows/`:

**ci.yml** runs on every push and PR:
- Lint with ruff (`src/` and `app/`)
- Smoke test (import all modules with dummy env vars)
- Docker build test

**deploy.yml** runs on merge to main:
- Syncs code to the HF Spaces repo (excludes data/, notebooks/, evaluations/, docs/)
- Copies evaluation chart images for the dashboard
- Pushes to HF, which triggers a container rebuild

Requires GitHub secrets: `HF_TOKEN` (HF write token) and `HF_SPACE_NAME` (e.g. `studiobuilders/ism-cyberrag`).

See `docs/sprint-3/DEPLOYMENT_GUIDE.md` for full setup instructions.

## Evaluation

The evaluation dataset (`evaluations/eval_questions.json`) contains 100 questions across five categories:
- Easy (30), Medium (30), Hard (20), Very Hard (10), Out of Scope (10)

| Metric | Sprint 1 | Sprint 2 | Sprint 3 Target |
|--------|----------|----------|-----------------|
| Faithfulness | 0.6834 | 0.7341 | > 0.78 |
| Answer Relevancy | 0.7216 | 0.7678 | > 0.82 |
| Context Precision | 0.7885 | 0.8598 | > 0.85 |
| Context Recall | 0.8224 | 0.8659 | > 0.91 |
| Answer Similarity | N/A | 0.9057 | > 0.93 |

## Team

| Member | Student ID |
|--------|-----------|
| Sreekar Reddy Edulapalli | 25617806 |
| Chandan Sreenivasaiah | 25674250 |
| Ruben Easo Thomas | 25598184 |

## License

MIT
