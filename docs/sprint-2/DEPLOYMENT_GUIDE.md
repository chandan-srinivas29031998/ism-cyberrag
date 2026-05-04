## How to Run Locally

### Prerequisites

- Python 3.10+
- A `.env` file with Supabase, Groq and ClearML credentials (copy from `.env.example`)
- Supabase project with the Sprint 2 schema applied

### Step 1: Install dependencies

```bash
cd ism-cyberrag
pip install -r requirements.txt
```

### Step 2: Apply database migration

If your Supabase already has the Sprint 1 schema, run `database/sprint2_migration.sql` in the Supabase SQL editor. This adds the fts column, GIN index and hybrid_search function.

If starting fresh, run `database/schema.sql` instead. It includes everything.

### Step 3: Run the notebook

Open `notebooks/sprint2_development.ipynb` in Jupyter or Google Colab. Run all cells in order:

1. Environment setup and ClearML init
2. Load models (embedding model + reranker)
3. Parse PDFs, chunk with ISM-aware chunker, embed, ingest into Supabase
4. Chunking statistics and metadata coverage charts
5. Test hybrid search + reranking on a sample query
6. Run RAGAS evaluation on all 100 questions
7. Compare Sprint 1 vs Sprint 2 scores
8. Log everything to ClearML and save CSVs

The notebook saves results to `evaluations/sprint2_eval_results.csv` and charts to `evaluations/sprint2_*.png`.

### Step 4: Run the web app (local)

```bash
cd ism-cyberrag
source .venv/bin/activate
uvicorn app.main:app --reload
```

Then open http://localhost:8000 in a browser.

### Step 5: Run the web app (Colab or remote)

If running on Colab or a remote server, the notebook handles setup automatically. The FastAPI app can be started from a notebook cell if needed, but is mainly for local development and the demo video.



# Deployment Guide: Hugging Face Spaces

This guide outlines how to deploy the ISM CyberRAG FastAPI application to Hugging Face (HF) Spaces using the Docker template.

## Prerequisites

1. A Hugging Face account and a new Space configured using the **Docker** template.
2. The Hugging Face Space should be set to at least the "Free" tier (which provides 16GB of RAM, sufficient for loading the chosen embedded/reranking models).
3. Secrets available for connecting to Supabase and the Groq API.

## Pre-Deployment (Local Check)

Before pushing to HF Spaces, ensure all evaluations are completed locally.

1. **Re-Run Notebook**: Run `notebooks/sprint2_development.ipynb` start-to-finish. This will:
   - Reset the Supabase schema and push the new chunks.
   - Run the 100 evaluation questions.
   - Generate UI images (`sprint2_oos_detail.png`, `sprint2_inscope_vs_oos.png`, etc.) locally inside `evaluations/`.
2. **Copy Images to App**:
   Copy the newly generated PNGs into the app's static directory so they are served in the `/evaluations` dashboard.
   ```bash
   mkdir -p app/static/evaluations
   cp evaluations/*.png app/static/evaluations/
   ```

## Creating the Docker Space

When creating the Docker Space on HF, you simply need to link this repository (or copy the current codebase). The root `Dockerfile` instructs Hugging Face to:
- Use Python 3.11 slim.
- Restrict PyTorch to the CPU-only version (dramatically reduces build times and RAM overhead).
- Install all remaining dependencies from `requirements.txt`.
- Pre-cache the `nomic-embed-text-v1.5` and `ms-marco-MiniLM-L-6-v2` models during the build phase so that initial requests are fully responsive.

## Setting Environment Variables (Secrets)

In your Hugging Face Space settings, navigate to "Variables and secrets" and add the following **New Secrets**:

| Secret Key | Description |
|------------|-------------|
| `SUPABASE_URL` | Your Supabase project URL (e.g., `https://xyz.supabase.co`) |
| `SUPABASE_PUBLISHABLE_KEY` | Your Supabase Anon/Publishable API Key |
| `GROQ_API_KEY` | Your Groq API Key required for querying Llama 3.1 |

> **Note**: You do not need to configure ClearML credentials in the Hugging Face Space. ClearML tracking is strictly used for offline development and notebook benchmarking.

## Monitoring the App

Once deployed, the Space should successfully launch on port 7860 (default HF Spaces mapping). 
- Visit the Space URL to use the conversational Split-Screen UI.
- Add `/evaluations` to the URL to view the static evaluations dashboard with your copied images.
