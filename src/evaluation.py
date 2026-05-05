import json
import time
from typing import Callable

import pandas as pd


def load_eval_dataset(path: str) -> list[dict]:
    """
    Loads the evaluation dataset from a JSON file.

    Expected format:
    [
        {"question": "...", "ground_truth": "...", "category": "easy|medium|hard|out_of_scope"},
        ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} evaluation questions.")
    return data


def run_ragas_evaluation(
    eval_dataset: list[dict],
    retrieve_fn: Callable,
    generate_fn: Callable,
) -> list[dict]:
    """
    Runs the full RAG pipeline on every evaluation question.
    Tracks per-question latency for retrieval and generation.

    Args:
        eval_dataset: List of dicts with 'question' and 'ground_truth'.
        retrieve_fn:  callable(question: str) -> list[dict]  (each dict has 'content').
                      It may also return (chunks, metadata) for guardrail details.
        generate_fn:  callable(question: str, chunks: list[dict]) -> str.

    Returns:
        List of result dicts with keys:
        question, ground_truth, answer, contexts,
        retrieval_time_s, generation_time_s, total_time_s,
        max_rerank_score, guardrail_stage, query_variants.
    """
    results = []
    total = len(eval_dataset)

    max_retries = 3

    for i, item in enumerate(eval_dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        retrieval_time = 0.0
        generation_time = 0.0

        max_rerank_score = 0.0
        guardrail_stage = None
        pre_filter_reason = None
        query_variants = []
        contexts = []
        answer = ""

        for attempt in range(1, max_retries + 1):
            try:
                # Retrieve (timed)
                t0 = time.time()
                retrieved = retrieve_fn(question)
                retrieval_time = time.time() - t0

                retrieval_meta = {}
                if isinstance(retrieved, tuple) and len(retrieved) == 2:
                    chunks, retrieval_meta = retrieved
                else:
                    chunks = retrieved

                contexts = [c["content"] for c in chunks]
                guardrail_stage = retrieval_meta.get("guardrail_stage")
                pre_filter_reason = retrieval_meta.get("pre_filter_reason")
                query_variants = retrieval_meta.get("query_variants", [])

                # Capture max rerank score for threshold analysis
                if "max_rerank_score" in retrieval_meta:
                    max_rerank_score = float(retrieval_meta["max_rerank_score"])
                elif chunks:
                    max_rerank_score = max(
                        c.get("rerank_score", 0.0) for c in chunks
                    )

                # Generate (timed)
                t0 = time.time()
                answer = generate_fn(question, chunks)
                generation_time = time.time() - t0
                break
            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"  [{i}/{total}] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [{i}/{total}] ERROR after {max_retries} attempts: {e}")
                    contexts = []
                    answer = f"Error: {e}"
                    guardrail_stage = "error"
                    pre_filter_reason = None

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
            "category": item.get("category", "unknown"),
            "guardrail_stage": guardrail_stage,
            "pre_filter_reason": pre_filter_reason,
            "query_variants": query_variants,
            "retrieval_time_s": round(retrieval_time, 4),
            "generation_time_s": round(generation_time, 4),
            "total_time_s": round(retrieval_time + generation_time, 4),
            "max_rerank_score": round(max_rerank_score, 4),
        })
        print(f"  [{i}/{total}] ({retrieval_time + generation_time:.2f}s) {question[:60]}...")

    # Print latency summary
    avg_retrieval = sum(r["retrieval_time_s"] for r in results) / len(results)
    avg_generation = sum(r["generation_time_s"] for r in results) / len(results)
    avg_total = sum(r["total_time_s"] for r in results) / len(results)
    print(f"\n  Avg retrieval: {avg_retrieval:.3f}s | Avg generation: {avg_generation:.3f}s | Avg total: {avg_total:.3f}s")

    return results


def compute_ragas_scores(eval_results: list[dict]) -> tuple[dict, "pd.DataFrame"]:
    """
    Computes RAGAS metrics using the configured Eval LLM as the judge and
    HuggingFace (nomic-embed-text-v1.5) for embeddings.

    Args:
        eval_results: List of dicts from run_ragas_evaluation(), each
                      containing 'question', 'answer', 'contexts', 'ground_truth'.

    Returns:
        Tuple of (score_dict, results_df):
        - score_dict: dict of metric_name -> average float score.
        - results_df: pandas DataFrame with per-question scores and latency.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
    )
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from src.config import GROQ_API_KEY, EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL, EVAL_LLM_PROVIDER, EVAL_LLM_MODEL
    
    # ── LLM judge (Groq or Ollama) ──
    if EVAL_LLM_PROVIDER == "ollama":
        from langchain_community.chat_models import ChatOllama
        eval_llm = ChatOllama(
            model=EVAL_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )
        print(f"Using Ollama as RAGAS judge (model: {EVAL_LLM_MODEL})")
    else:
        from langchain_groq import ChatGroq
        eval_llm = ChatGroq(
            model_name=EVAL_LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0,
        )
        print(f"Using Groq as RAGAS judge (model: {EVAL_LLM_MODEL})")

    # ── Embedding model (local, same as pipeline) ──
    eval_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"trust_remote_code": True},
    )

    # ── Build RAGAS-compatible dataset ──
    ragas_data = {
        "question": [r["question"] for r in eval_results],
        "answer": [r["answer"] for r in eval_results],
        "contexts": [r["contexts"] for r in eval_results],
        "ground_truth": [r["ground_truth"] for r in eval_results],
    }
    ragas_dataset = Dataset.from_dict(ragas_data)

    # ── Evaluate (sequential with retries to avoid timeouts) ──
    run_cfg = RunConfig(max_workers=1, max_retries=5, max_wait=300)
    print("Computing RAGAS metrics (sequential, max_retries=5, timeout=300s)...")
    result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_cfg,
    )

    # ── Build per-question results DataFrame ──
    if hasattr(result, "to_pandas"):
        ragas_df = result.to_pandas()
    else:
        ragas_df = pd.DataFrame([dict(result)])

    # Merge latency and category data from eval_results into the RAGAS DataFrame
    def is_oos_refusal(result: dict) -> bool:
        if result.get("category") != "out_of_scope":
            return False

        answer = str(result.get("answer", "")).strip().lower()
        return (
            "outside the scope" in answer
            and "ism" in answer
            and "enough information" in answer
        )

    extra_df = pd.DataFrame([{
        "category": r.get("category", "unknown"),
        "is_oos_refusal": is_oos_refusal(r),
        "guardrail_stage": r.get("guardrail_stage"),
        "pre_filter_reason": r.get("pre_filter_reason"),
        "query_variants": r.get("query_variants", []),
        "retrieval_time_s": r.get("retrieval_time_s", 0),
        "generation_time_s": r.get("generation_time_s", 0),
        "total_time_s": r.get("total_time_s", 0),
        "max_rerank_score": r.get("max_rerank_score", 0),
    } for r in eval_results])

    if len(extra_df) == len(ragas_df):
        results_df = pd.concat([ragas_df, extra_df], axis=1)
    else:
        results_df = ragas_df

    # ── Handle NaN / misleading scores from correct OOS refusal answers ──
    # RAGAS was designed for answer-from-context cases, not refusal answers:
    #   - Faithfulness may be NaN because the refusal has zero factual claims.
    #   - Answer Relevancy is often forced to 0.0 by the noncommittal detector.
    #   - Context Recall may be NaN/low because there is no supporting context.
    # For rows that are both labelled out_of_scope AND actually refused, set
    # those refusal-specific metrics to 1.0. Do not adjust context_precision:
    # irrelevant retrieved context should remain visible as retrieval noise.
    # See: https://github.com/explodinggradients/ragas/issues/733
    #      https://github.com/explodinggradients/ragas/issues/1651
    if "is_oos_refusal" in results_df.columns:
        refusal_mask = results_df["is_oos_refusal"].fillna(False).astype(bool)
    else:
        refusal_mask = pd.Series(False, index=results_df.index)

    oos_refusal_fills = {"faithfulness": 1.0, "answer_relevancy": 1.0, "context_recall": 1.0}
    for col, fill_val in oos_refusal_fills.items():
        if col in results_df.columns:
            adjust_mask = refusal_mask & (results_df[col].isna() | (results_df[col] < fill_val))
            adjust_count = int(adjust_mask.sum())
            if adjust_count > 0:
                results_df.loc[adjust_mask, col] = fill_val
                print(f"  Set {adjust_count} {col} scores to {fill_val} for correct OOS refusals.")

    # ── Compute average scores (all questions) ──
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_similarity"]
    score_dict = {}
    for col in metric_cols:
        if col in results_df.columns:
            score_dict[col] = float(results_df[col].mean())

    # Add latency averages to score_dict
    for col in ["retrieval_time_s", "generation_time_s", "total_time_s", "max_rerank_score"]:
        if col in results_df.columns:
            score_dict[f"avg_{col}"] = float(results_df[col].mean())

    print("\n══════ RAGAS Evaluation Results (All 100 Questions) ══════")
    for name, score in score_dict.items():
        print(f"  {name:25s} {score:.4f}")

    # ── In-scope vs OOS breakdown ──
    if "category" in results_df.columns:
        is_oos = results_df["category"] == "out_of_scope"
        inscope_df = results_df[~is_oos]
        oos_df = results_df[is_oos]

        if len(inscope_df) > 0:
            print(f"\n══════ In-Scope Only ({len(inscope_df)} questions) ══════")
            for col in metric_cols:
                if col in inscope_df.columns:
                    val = float(inscope_df[col].mean())
                    score_dict[f"inscope_{col}"] = val
                    print(f"  {col:25s} {val:.4f}")

        if len(oos_df) > 0:
            print(f"\n══════ Out-of-Scope Only ({len(oos_df)} questions) ══════")
            for col in metric_cols:
                if col in oos_df.columns:
                    val = float(oos_df[col].mean())
                    score_dict[f"oos_{col}"] = val
                    print(f"  {col:25s} {val:.4f}")

    return score_dict, results_df


def log_metrics_to_clearml(
    metrics: dict,
    params: dict | None = None,
    results_df: "pd.DataFrame | None" = None,
    eval_results: list[dict] | None = None,
):
    """
    Logs RAGAS metrics, pipeline parameters, evaluation results, and
    sample Q&A outputs to the current ClearML task.

    ClearML must already be initialized via Task.init() before calling this.

    Args:
        metrics:      dict of metric_name -> float score.
        params:       dict of parameter_name -> value (logged as hyperparameters).
        results_df:   pandas DataFrame with per-question RAGAS scores (optional).
        eval_results: list of dicts from run_ragas_evaluation() for sample Q&A (optional).
    """
    from clearml import Task

    task = Task.current_task()
    if task is None:
        print("WARNING: No active ClearML task found. Skipping metric logging.")
        return

    logger = task.get_logger()

    # ── 1. Log each RAGAS metric as a scalar ──
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.report_scalar(title="RAGAS Metrics", series=name, value=value, iteration=0)

    # ── 2. Log pipeline parameters as hyperparameters ──
    if params:
        task.connect(params, name="Pipeline Parameters")

    # ── 3. Upload full results DataFrame as CSV artifact ──
    if results_df is not None:
        task.upload_artifact(
            name="eval_results",
            artifact_object=results_df,
        )
        print(f"Uploaded eval_results artifact ({len(results_df)} rows).")

        # ── 4. Log results as a table in ClearML ──
        # Select display columns (exclude raw contexts for readability)
        display_cols = [c for c in results_df.columns if c != "contexts"]
        logger.report_table(
            title="Evaluation Results",
            series="Per-Question Scores",
            table_plot=results_df[display_cols],
        )

    # ── 5. Log sample Q&A outputs ──
    if eval_results:
        import pandas as pd
        sample_count = min(10, len(eval_results))
        samples = []
        for r in eval_results[:sample_count]:
            samples.append({
                "question": r["question"][:120],
                "answer": r["answer"][:200],
                "ground_truth": r["ground_truth"][:200],
                "retrieval_time_s": r.get("retrieval_time_s", ""),
                "generation_time_s": r.get("generation_time_s", ""),
            })
        sample_df = pd.DataFrame(samples)
        logger.report_table(
            title="Sample Q&A Outputs",
            series="First 10 Questions",
            table_plot=sample_df,
        )
        print(f"Logged {sample_count} sample Q&A outputs to ClearML.")

    print(f"Logged {len(metrics)} metrics to ClearML.")
