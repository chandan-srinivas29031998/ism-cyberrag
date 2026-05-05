# Sprint 3 Deployment Guide

## Local development

The local setup is the same as Sprint 2.

### Prerequisites

- Python 3.10+
- A `.env` file with Supabase, Groq, and ClearML credentials (copy from `.env.example`)
- Supabase project with the current schema applied

### Install and run

```bash
cd ism-cyberrag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the web app:

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 in a browser. The three tabs (Search ISM, Pipeline Explorer, Evaluations) will be available.


## CI/CD pipeline

Sprint 3 adds automated linting, testing, and deployment via two GitHub Actions workflows in `.github/workflows/`.

### ci.yml -- runs on every push and pull request

This workflow has three jobs that run in parallel:

**Lint.** Installs `ruff` and runs `ruff check src/ app/`. This catches style issues and common Python errors before code is merged.

**Smoke test.** Installs CPU-only PyTorch and all project dependencies, then runs a Python snippet that imports `app.main` with dummy environment variables. This verifies that all modules load correctly and there are no import errors or missing dependencies. It does not call Supabase or Groq because those credentials are not available in CI.

**Docker build.** Builds the Docker image using the root `Dockerfile` to make sure the container builds successfully. It does not push the image anywhere.

All three jobs must pass before a pull request can be merged.

### deploy.yml -- runs on merge to main

This workflow handles deployment to Hugging Face Spaces. When a commit lands on `main`, it:

1. Checks out the repo.
2. Clones the HF Spaces repo using the `HF_TOKEN` secret.
3. Syncs the codebase to the HF repo using rsync, excluding directories that are not needed in production (data/, notebooks/, evaluations/, docs/, .env, .venv).
4. Copies evaluation chart images from `evaluations/sprint-*/sprint*.png` into `hf-space/app/static/evaluations/` so the Evaluations tab has charts to display.
5. Commits and pushes to the HF repo, which triggers a rebuild on Hugging Face.

### Secrets required in GitHub

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Hugging Face access token with write permission to the Space |
| `HF_SPACE_NAME` | Full Space name, e.g. `studiobuilders/ism-cyberrag` |

These are set in the GitHub repo under Settings > Secrets and variables > Actions.


## Hugging Face Spaces deployment

The app is deployed as a Docker-based HF Space. The Dockerfile builds the container, installs CPU-only PyTorch, pre-caches the embedding and reranking models, and starts the FastAPI server on port 7860. HF Spaces rebuilds the container automatically whenever new code is pushed to the Space repo.

### Step 1: Create the Hugging Face Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - Owner: `studiobuilders` (or your HF org/username)
   - Space name: `ism-cyberrag`
   - License: MIT
   - SDK: **Docker**
   - Hardware: **CPU Basic** (free tier, 2 vCPU, 16GB RAM, enough for this app)
   - Visibility: Public (or Private if you prefer)
3. Click "Create Space". This creates an empty git repo at `https://huggingface.co/spaces/studiobuilders/ism-cyberrag`.

### Step 2: Set secrets on the HF Space

Go to the Space page, click Settings, scroll to "Variables and secrets", and add these as **Secrets** (not variables, so they are not visible in logs):

| Secret | Description | Where to get it |
|--------|-------------|-----------------|
| `SUPABASE_URL` | Supabase project URL | Supabase dashboard, Settings, API |
| `SUPABASE_PUBLISHABLE_KEY` | Supabase anon/publishable API key | Same page |
| `GROQ_API_KEY` | Groq API key for Llama 3.1 | https://console.groq.com/keys |

ClearML credentials are not needed on HF Spaces. ClearML is only used in notebook evaluation runs.

The app reads these via `os.getenv()` in `src/config.py`. The `.env.example` file is copied as `.env` in the Docker image, but `load_dotenv()` does not override environment variables that are already set, so the HF Spaces secrets take precedence over the placeholder values.

### Step 2a: Enable Supabase RLS

Supabase tables in the public schema should have Row Level Security enabled. Sprint 3 does not rebuild or mutate the corpus at runtime; it only reads the existing `documents` and `chunks` tables through `match_chunks` and `hybrid_search`.

For an existing Sprint 2 database, run `database/sprint3_rls.sql`. The core policy shape is:

```sql
alter table public.chunks enable row level security;
alter table public.documents enable row level security;

create policy "Allow read access to ISM chunks"
on public.chunks
for select
to anon, authenticated
using (true);

create policy "Allow read access to ISM documents"
on public.documents
for select
to anon, authenticated
using (true);
```

No insert, update, or delete policies are created for `anon` or `authenticated`. The web app continues to use `SUPABASE_PUBLISHABLE_KEY`, so these read policies are required; enabling RLS without them would block retrieval.

### Step 3: Set secrets on GitHub (for automated deployment)

Go to the GitHub repo, Settings, Secrets and variables, Actions, and add:

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Hugging Face access token with write permission. Create one at https://huggingface.co/settings/tokens with "Write" scope. |
| `HF_SPACE_NAME` | Full Space path, e.g. `studiobuilders/ism-cyberrag` |

### Step 4: Push to main

Once both sets of secrets are configured, any push to `main` triggers the `deploy.yml` workflow. It clones the HF Space repo, syncs the project files (excluding data/, notebooks/, evaluations/, docs/, .env), copies evaluation chart images, and pushes to HF. This triggers HF Spaces to rebuild the Docker container and restart the app.

The app will be available at `https://studiobuilders-ism-cyberrag.hf.space` (or whatever your org/space name is).

### Manual deployment (if GitHub Actions is not set up)

If you need to deploy without the CI/CD pipeline:

```bash
# Clone the HF Space repo
git clone https://huggingface.co/spaces/studiobuilders/ism-cyberrag hf-space

# Copy project files (exclude what is not needed in production)
rsync -av --delete \
    --exclude '.git' --exclude '.venv' --exclude 'venv' \
    --exclude 'data/' --exclude 'notebooks/' \
    --exclude 'evaluations/' --exclude 'docs/' \
    --exclude '.env' \
    ./ hf-space/

# Copy evaluation chart images for the dashboard
mkdir -p hf-space/app/static/evaluations
cp evaluations/sprint-2/sprint*.png hf-space/app/static/evaluations/ 2>/dev/null
cp evaluations/sprint-3/sprint*.png hf-space/app/static/evaluations/ 2>/dev/null

# Push to HF
cd hf-space
git add -A
git commit -m "Manual deploy"
git push
```

### What happens during container build

When HF Spaces receives a push, it runs the Dockerfile:

1. Starts from `python:3.11-slim`
2. Installs git and build-essential (needed for compiling some Python packages)
3. Installs CPU-only PyTorch (no GPU needed, saves ~500MB vs full PyTorch)
4. Installs all requirements from `requirements.txt`
5. Copies `src/` and `app/` directories
6. Copies `.env.example` as `.env` (placeholder values, overridden by HF secrets)
7. Downloads and caches the embedding model (nomic-embed-text-v1.5, ~280MB) and the reranker model (ms-marco-MiniLM-L-6-v2, ~80MB) so they are baked into the image. Without this step, the first request to the app would take 30-60 seconds while models download.
8. Creates a non-root user (HF Spaces runs containers as user 1000)
9. Starts uvicorn on port 7860

Build takes 5-10 minutes on HF Spaces. After that, the app starts in under 5 seconds because models are already cached.

### Troubleshooting

**Space shows "Building" for a long time:** The first build downloads PyTorch and both ML models, which takes 5-10 minutes. Subsequent builds use Docker layer caching and are faster (1-2 minutes) unless requirements.txt changes.

**App starts but shows errors:** Check the Space logs (click "Logs" tab on the Space page). Most likely a missing secret. If `SUPABASE_URL` is not set, the app will crash on the first request.

**Evaluations tab shows no charts:** The evaluation chart images need to be in `app/static/evaluations/`. If deploying via GitHub Actions, the workflow copies them automatically. If deploying manually, copy them yourself (see manual deployment steps above).


## Environment variables

Full list of environment variables used by the application. All are set in `.env` for local development and as secrets on HF Spaces for production.

### Core (required)

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_PUBLISHABLE_KEY` | Supabase anon/publishable API key |
| `GROQ_API_KEY` | Groq API key |

### LLM configuration (optional, have defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | LLM provider. Set to `ollama` for local Ollama. |
| `LLM_MODEL_NAME` | `llama-3.1-8b-instant` | Model name for the main LLM |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (only used when LLM_PROVIDER=ollama) |

### Sprint 3: Multi-query expansion

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTI_QUERY_ENABLED` | `true` | Enable or disable multi-query expansion |
| `MULTI_QUERY_COUNT` | `3` | Number of alternate query phrasings to generate |

### Sprint 3: OOS guardrail

| Variable | Default | Description |
|----------|---------|-------------|
| `OOS_PRE_FILTER_ENABLED` | `true` | Enable or disable the keyword/intent pre-filter |
| `OOS_RERANK_THRESHOLD` | `-5.0` | Minimum max rerank score to proceed to LLM generation. Questions where the best chunk scores below this are blocked. Calibrated from Sprint 2 to avoid blocking known in-scope questions; obvious vendor/code/pricing OOS cases are handled by the pre-filter. |

### Evaluation (notebook only, not needed on HF Spaces)

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_LLM_PROVIDER` | `ollama` | LLM provider for RAGAS evaluation |
| `EVAL_LLM_MODEL` | `llama3.1:8b` | Model for RAGAS evaluation |
| `CLEARML_TASK` | `Sprint 3 - Multi-Query + OOS Guardrail + Deployment` | ClearML task name |


## Static evaluation images

The Evaluations tab in the web app displays chart images generated by the evaluation notebooks. These images are not committed to the app's static directory. Instead, the notebooks save them to `evaluations/sprint-3/`, and the deploy workflow copies them into `app/static/evaluations/` during deployment.

If you are running locally and want the Evaluations tab to show charts, copy them manually:

```bash
mkdir -p app/static/evaluations
cp evaluations/sprint-2/sprint*.png app/static/evaluations/
cp evaluations/sprint-3/sprint*.png app/static/evaluations/
```

The deploy workflow handles this automatically for HF Spaces.
