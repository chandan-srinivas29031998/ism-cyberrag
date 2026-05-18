"""
HPO base task: runs the Sprint 3 retrieval + generation pipeline against a
30-question stratified subset, scores it with RAGAS, and reports a single
optimisation objective (composite score) to ClearML.

This script is designed to be cloned and re-run by the ClearML
HyperParameterOptimizer (see `hpo_controller.py`). The optimizer overrides
`General/OOS_RERANK_THRESHOLD` per child run; this script reads the override
back from the connected dict after `Task.init()` and passes it explicitly to
`rerank_threshold_check`. The module-level default in `src/config.py` is NOT
used during HPO runs - that would be a silent bug.

Standalone usage:
    python scripts/hpo_base_task.py --dry-run    # validate env, print plan, no API calls
    python scripts/hpo_base_task.py              # full run with default threshold (-5.0)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSET_PATH = PROJECT_ROOT / "evaluations" / "sprint-3" / "hpo" / "hpo_subset.json"
RESULTS_DIR = PROJECT_ROOT / "evaluations" / "sprint-3" / "hpo" / "results"
CLEARML_PROJECT = "ISM-CyberRAG"
CLEARML_TASK_NAME = "HPO base - OOS threshold sweep"
CLEARML_TAGS = ["hpo-sweep"]

DEFAULT_THRESHOLD = -5.0
PER_QUESTION_TIMEOUT_S = 60  # hard cap on generation per question


sys.path.insert(0, str(PROJECT_ROOT))

# Force the answer-generation LLM to Ollama for HPO runs so the sweep doesn't
# spend Groq quota. The user's main .env keeps LLM_PROVIDER=groq for normal
# Sprint 3 eval; this override only affects this script's process. `setdefault`
# means an explicit shell env can still override (e.g. for debugging).
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL_NAME", "llama3.1:8b")


# ---------------------------------------------------------------------------
# Pre-flight checks (no third-party imports needed for these)
# ---------------------------------------------------------------------------


def _print_check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line = f"{line} - {detail}"
    print(line)


def check_subset_file() -> tuple[bool, str]:
    if not SUBSET_PATH.exists():
        return (
            False,
            f"missing - run `python scripts/hpo_subset.py` first to build it",
        )
    try:
        with SUBSET_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        n = len(data.get("questions", []))
        return True, f"{n} questions"
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"unreadable - {exc}"


def check_clearml_env() -> tuple[bool, str]:
    keys = ["CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY"]
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        return False, f"missing env vars: {missing}"
    return True, "credentials present"


def check_supabase_env() -> tuple[bool, str]:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_PUBLISHABLE_KEY", "") or os.getenv("SUPABASE_SECRET_KEY", "")
    if not url or not key:
        return False, "SUPABASE_URL or SUPABASE_*_KEY missing"
    return True, "credentials present"


def check_ollama() -> tuple[bool, str]:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
    except (requests.ConnectionError, requests.Timeout) as exc:
        return False, f"unreachable at {base} - {type(exc).__name__}"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"
    try:
        models = [m.get("name", "") for m in resp.json().get("models", [])]
    except ValueError:
        return False, "invalid JSON from /api/tags"
    needed = os.getenv("EVAL_LLM_MODEL", "llama3.1:8b")
    present = any(m.startswith(needed.split(":")[0]) for m in models)
    if not present:
        return False, f"model {needed} not pulled - run `ollama pull {needed}`"
    return True, f"reachable, {needed} available"


def run_pre_flight() -> bool:
    print("Pre-flight checks:")
    checks = [
        ("Subset file", check_subset_file),
        ("ClearML credentials", check_clearml_env),
        ("Supabase credentials", check_supabase_env),
        ("Ollama", check_ollama),
    ]
    all_ok = True
    for name, fn in checks:
        ok, detail = fn()
        _print_check(name, ok, detail)
        all_ok = all_ok and ok
    return all_ok


# ---------------------------------------------------------------------------
# Per-question timeout helper (POSIX only - macOS and Linux)
# ---------------------------------------------------------------------------


class QuestionTimeout(Exception):
    pass


@contextmanager
def _hard_timeout(seconds: int):
    """Raise QuestionTimeout if the wrapped block exceeds `seconds`."""
    if not hasattr(signal, "SIGALRM"):
        # Windows fallback: no timeout enforcement.
        yield
        return

    def _handler(signum, frame):
        raise QuestionTimeout(f"exceeded {seconds}s")

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


# ---------------------------------------------------------------------------
# Main run logic (heavy imports happen inside so dry-run stays fast)
# ---------------------------------------------------------------------------


def build_retrieve_fn(threshold: float):
    """
    Build the retrieve closure used by `run_ragas_evaluation`. Mirrors
    `retrieve_sprint3` from `notebooks/sprint3_development.ipynb` but threads the
    swept threshold value through `rerank_threshold_check` explicitly.
    """
    from src.config import INITIAL_RETRIEVE_COUNT, RERANK_TOP_K
    from src.embeddings import embed_query, load_embedding_model
    from src.guardrail import pre_filter, rerank_threshold_check
    from src.query_expansion import expand_query
    from src.reranking import load_reranker, rerank
    from src.retrieval import multi_query_retrieve
    from src.supabase_utils import get_supabase_client

    # Load heavy resources once.
    print("Loading Supabase client, embedding model, reranker...")
    supabase = get_supabase_client()
    embed_model = load_embedding_model()
    load_reranker()  # warm the model cache; rerank() uses the cached instance

    def retrieve(question: str):
        metadata: dict[str, Any] = {
            "guardrail_stage": None,
            "pre_filter_reason": None,
            "query_variants": [],
            "max_rerank_score": 0.0,
        }
        try:
            passed, reason = pre_filter(question)
        except Exception as exc:
            print(f"  DIAG pre_filter_error: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            metadata["guardrail_stage"] = "error"
            metadata["pre_filter_reason"] = f"pre_filter_error: {type(exc).__name__}"
            return [], metadata
        metadata["pre_filter_reason"] = reason
        if not passed:
            metadata["guardrail_stage"] = "pre_filter"
            return [], metadata

        try:
            queries = expand_query(question)
        except Exception as exc:
            print(f"  query_expansion failed: {type(exc).__name__}: {exc}. Falling back to single query.")
            queries = [question]
        metadata["query_variants"] = list(queries)

        try:
            chunks = multi_query_retrieve(
                supabase,
                lambda q: embed_query(embed_model, q),
                queries,
                match_count=INITIAL_RETRIEVE_COUNT,
            )
        except Exception as exc:
            print(f"  DIAG retrieve_error: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            metadata["guardrail_stage"] = "error"
            metadata["pre_filter_reason"] = f"retrieve_error: {type(exc).__name__}: {exc}"
            return [], metadata

        try:
            reranked = rerank(question, chunks, top_k=RERANK_TOP_K)
        except Exception as exc:
            print(f"  DIAG rerank_error: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            metadata["guardrail_stage"] = "error"
            metadata["pre_filter_reason"] = f"rerank_error: {type(exc).__name__}: {exc}"
            return [], metadata

        passed, max_score = rerank_threshold_check(reranked, threshold=threshold)
        metadata["max_rerank_score"] = float(max_score)
        if not passed:
            metadata["guardrail_stage"] = "rerank_threshold"
            return [], metadata

        return reranked, metadata

    return retrieve


def safe_generate_fn(generate_answer):
    """Wrap generate_answer with a per-question hard timeout."""

    def generate(question: str, chunks: list[dict]) -> str:
        try:
            with _hard_timeout(PER_QUESTION_TIMEOUT_S):
                return generate_answer(question, chunks)
        except QuestionTimeout as exc:
            return f"Error: generation timed out ({exc})"
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"

    return generate


def compute_composite(metrics: dict, results: list[dict], subset: list[dict]) -> tuple[float, dict]:
    """
    Composite objective: 0.5 * inscope_faithfulness + 0.5 * oos_block_rate.

    Returns (composite, extras) where extras is the breakdown logged for
    transparency. If anything is missing or NaN, composite = -1.0 and extras
    contains a 'warning' message.
    """
    extras: dict[str, float | str] = {}
    inscope_faith = metrics.get("inscope_faithfulness")
    if inscope_faith is None or (isinstance(inscope_faith, float) and math.isnan(inscope_faith)):
        extras["warning"] = "inscope_faithfulness missing or NaN"
        return -1.0, extras

    oos_total = sum(1 for q in subset if q.get("category") == "out_of_scope")
    if oos_total == 0:
        extras["warning"] = "no OOS questions in subset"
        return -1.0, extras

    oos_blocked = 0
    for q, r in zip(subset, results):
        if q.get("category") != "out_of_scope":
            continue
        stage = r.get("guardrail_stage")
        if stage in {"pre_filter", "rerank_threshold"}:
            oos_blocked += 1
    oos_block_rate = oos_blocked / oos_total

    composite = 0.5 * float(inscope_faith) + 0.5 * float(oos_block_rate)
    extras["inscope_faithfulness"] = float(inscope_faith)
    extras["oos_block_rate"] = float(oos_block_rate)
    extras["oos_total_in_subset"] = float(oos_total)
    extras["oos_blocked_count"] = float(oos_blocked)
    return composite, extras


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pre-flight checks and the run plan, then exit. Makes zero API calls.",
    )
    args = parser.parse_args()

    pre_flight_ok = run_pre_flight()
    if not pre_flight_ok:
        print("\nPre-flight failed. Fix the issues above and re-run.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n--dry-run: pre-flight passed. Exiting without running the pipeline.")
        return 0

    # --- Heavy imports past this point ---
    try:
        from clearml import Task
    except ImportError:
        print("ERROR: clearml is not installed. Run `pip install clearml`.", file=sys.stderr)
        return 1

    task = None
    try:
        print("\nInitialising ClearML task...")
        task = Task.init(
            project_name=CLEARML_PROJECT,
            task_name=CLEARML_TASK_NAME,
            tags=CLEARML_TAGS,
            reuse_last_task_id=False,
        )

        # Connect the hyperparameter dict. When this script runs as an HPO
        # child, the controller has already overridden the value before the
        # script starts; `connect` returns the (possibly mutated) dict.
        params: dict[str, Any] = {"OOS_RERANK_THRESHOLD": DEFAULT_THRESHOLD}
        params = task.connect(params)
        threshold = float(params["OOS_RERANK_THRESHOLD"])
        print(f"Threshold for this run: {threshold}")

        # Load subset
        with SUBSET_PATH.open("r", encoding="utf-8") as f:
            subset_payload = json.load(f)
        subset = subset_payload["questions"]
        print(f"Loaded {len(subset)} subset questions.")

        # Build retrieve_fn and generate_fn
        retrieve_fn = build_retrieve_fn(threshold=threshold)
        from src.llm import generate_answer
        generate_fn = safe_generate_fn(generate_answer)

        # Run eval
        from src.evaluation import compute_ragas_scores, run_ragas_evaluation
        print("\nRunning RAGAS evaluation on subset...")
        t0 = time.time()
        eval_results = run_ragas_evaluation(subset, retrieve_fn, generate_fn)
        elapsed = time.time() - t0
        print(f"\nEval run complete in {elapsed:.1f}s.")

        n_errors = sum(1 for r in eval_results if str(r.get("guardrail_stage")) == "error")
        if n_errors > len(eval_results) / 2:
            print(
                f"ERROR: {n_errors}/{len(eval_results)} questions errored. "
                f"Reporting composite = -1.0 and bailing.",
                file=sys.stderr,
            )
            task.logger.report_scalar("composite", "score", -1.0, iteration=0)
            task.logger.report_scalar("diagnostics", "n_errors", n_errors, iteration=0)
            return 1

        # Compute RAGAS scores
        print("Computing RAGAS scores...")
        try:
            metrics, results_df = compute_ragas_scores(eval_results)
        except Exception as exc:
            print(f"ERROR: compute_ragas_scores failed: {exc}", file=sys.stderr)
            traceback.print_exc()
            task.logger.report_scalar("composite", "score", -1.0, iteration=0)
            task.logger.report_scalar("diagnostics", "ragas_failed", 1.0, iteration=0)
            return 1

        # Composite + scalars
        composite, extras = compute_composite(metrics, eval_results, subset)
        print(f"\nComposite score: {composite:.4f}")
        for k, v in extras.items():
            print(f"  {k}: {v}")

        task.logger.report_scalar("composite", "score", float(composite), iteration=0)
        for k, v in extras.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                task.logger.report_scalar("composite_breakdown", k, float(v), iteration=0)

        # Log every metric for transparency.
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv) or math.isinf(fv):
                continue
            task.logger.report_scalar("RAGAS Metrics", k, fv, iteration=0)

        task.logger.report_scalar("diagnostics", "n_errors", float(n_errors), iteration=0)
        task.logger.report_scalar("diagnostics", "n_questions", float(len(eval_results)), iteration=0)
        task.logger.report_scalar("diagnostics", "wall_clock_s", float(elapsed), iteration=0)

        # Local saves so doc-writing later doesn't depend on the ClearML API.
        # Filename includes threshold + UTC timestamp so multiple runs coexist.
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
        thr_tag = f"thr{threshold:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
        local_csv = RESULTS_DIR / f"{thr_tag}_{ts}.csv"
        local_metrics = RESULTS_DIR / f"{thr_tag}_{ts}_metrics.json"
        try:
            results_df.to_csv(local_csv, index=False)
            print(f"Local CSV: {local_csv.relative_to(PROJECT_ROOT)}")
        except Exception as exc:
            print(f"WARN: failed to write local CSV: {exc}")
        try:
            metrics_payload = {
                "threshold": threshold,
                "composite_score": composite,
                "composite_breakdown": extras,
                "ragas_metrics": {
                    k: (float(v) if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else None)
                    for k, v in metrics.items()
                },
                "n_questions": len(eval_results),
                "n_errors": n_errors,
                "wall_clock_s": elapsed,
                "task_id": task.id,
                "timestamp_utc": ts,
            }
            with local_metrics.open("w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, indent=2, default=str)
            print(f"Local metrics: {local_metrics.relative_to(PROJECT_ROOT)}")
        except Exception as exc:
            print(f"WARN: failed to write local metrics JSON: {exc}")

        # ClearML artifacts (in addition to local copies above).
        try:
            task.upload_artifact("eval_results", artifact_object=results_df)
        except Exception as exc:
            print(f"WARN: failed to upload results_df: {exc}")

        try:
            task.upload_artifact("subset_used", artifact_object=subset_payload)
        except Exception as exc:
            print(f"WARN: failed to upload subset_used: {exc}")

        print("\nTask complete.")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nUNHANDLED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        if task is not None:
            try:
                task.logger.report_scalar("composite", "score", -1.0, iteration=0)
            except Exception:
                pass
        return 1
    finally:
        if task is not None:
            try:
                task.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
