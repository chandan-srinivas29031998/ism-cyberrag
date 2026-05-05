import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (works whether running from notebooks/ or root)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# ── Supabase ──────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "")

# ── Groq ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Ollama ────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Main LLM (for generating answers to questions) ───────
# Change these in .env. Values below are only fallback defaults.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")

# ── Evaluation LLM (for RAGAS metric scoring) ────────────
# Change these in .env. Values below are only fallback defaults.
# These are INDEPENDENT from the main LLM settings above.
EVAL_LLM_PROVIDER = os.getenv("EVAL_LLM_PROVIDER", "ollama")
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "llama3.1:8b")

# ── ClearML ───────────────────────────────────────────────
# ClearML SDK auto-reads these env vars when Task.init() is called:
#   CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY
#   CLEARML_API_HOST, CLEARML_WEB_HOST, CLEARML_FILES_HOST
# No need to reference them in code. Just set them in .env.
CLEARML_PROJECT = "ISM-CyberRAG"
CLEARML_TASK = os.getenv(
    "CLEARML_TASK",
    "Sprint 3 - Multi-Query + OOS Guardrail + Deployment",
)

# ── Embedding Models ─────────────────────────────────────
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIMENSION = 768

# ── RAG Parameters ────────────────────────────────────────
CHUNK_SIZE = 1000       # characters (Sprint 1 fixed-size, kept for reference)
CHUNK_OVERLAP = 200     # characters (Sprint 1 fixed-size, kept for reference)
MATCH_COUNT = 5         # final top-k after reranking

# ── Sprint 2: ISM-Aware Chunking ─────────────────────────
CHUNK_MIN_WORDS = 100   # merge chunks smaller than this
CHUNK_MAX_WORDS = 800   # split chunks larger than this

# ── Sprint 2: Hybrid Search ──────────────────────────────
INITIAL_RETRIEVE_COUNT = 10  # retrieve this many from hybrid search before reranking
RRF_K = 50              # RRF constant (as per sprint 2 plan)

# ── Sprint 2: Cross-Encoder Reranking ────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 5        # keep top-k after reranking

# ── Sprint 3: Multi-Query Expansion ─────────────────────
MULTI_QUERY_ENABLED = os.getenv("MULTI_QUERY_ENABLED", "true").lower() == "true"
MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "3"))
QUERY_EXPANSION_PROVIDER = os.getenv("QUERY_EXPANSION_PROVIDER", "ollama").lower()
QUERY_EXPANSION_MODEL = os.getenv("QUERY_EXPANSION_MODEL", EVAL_LLM_MODEL)
QUERY_EXPANSION_RETRIES = int(os.getenv("QUERY_EXPANSION_RETRIES", "2"))

# ── Sprint 3: OOS Guardrail ─────────────────────────────
OOS_PRE_FILTER_ENABLED = os.getenv("OOS_PRE_FILTER_ENABLED", "true").lower() == "true"
OOS_RERANK_THRESHOLD = float(os.getenv("OOS_RERANK_THRESHOLD", "-5.0"))

# ── Paths ─────────────────────────────────────────────────
PROJECT_ROOT = str(_project_root)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluations")
EVAL_DATASET_PATH = os.path.join(EVAL_DIR, "eval_questions.json")
