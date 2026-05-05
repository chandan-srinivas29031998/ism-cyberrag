import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.embeddings import load_embedding_model, embed_query
from src.retrieval import (
    control_id_search,
    control_ids_from_question,
    hybrid_search,
    terminology_search,
)
from src.reranking import rerank
from src.llm import GENERATION_ERROR_RESPONSE, generate_answer
from src.query_expansion import expand_query
from src.guardrail import (
    pre_filter,
    query_specificity_check,
    rerank_threshold_check,
    OOS_REFUSAL,
)
from src.supabase_utils import get_supabase_client
from src.config import (
    INITIAL_RETRIEVE_COUNT,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    LLM_PROVIDER,
    MULTI_QUERY_COUNT,
    MULTI_QUERY_ENABLED,
    OOS_RERANK_THRESHOLD,
    QUERY_EXPANSION_MODEL,
    QUERY_EXPANSION_PROVIDER,
    RERANKER_MODEL,
    RRF_K,
    RERANK_TOP_K,
)

router = APIRouter()

_embedding_model = None
_supabase_client = None

WEB_CACHE_ENABLED = os.getenv("WEB_CACHE_ENABLED", "true").lower() == "true"
WEB_CACHE_TTL_SECONDS = int(os.getenv("WEB_CACHE_TTL_SECONDS", "900"))
WEB_CACHE_MAX_ENTRIES = int(os.getenv("WEB_CACHE_MAX_ENTRIES", "256"))
WEB_RETRIEVAL_PARALLEL_ENABLED = os.getenv("WEB_RETRIEVAL_PARALLEL_ENABLED", "true").lower() == "true"
WEB_RETRIEVAL_MAX_WORKERS = int(os.getenv("WEB_RETRIEVAL_MAX_WORKERS", "4"))
WEB_RERANK_CANDIDATE_LIMIT = int(os.getenv("WEB_RERANK_CANDIDATE_LIMIT", "30"))
WEB_ANSWER_CACHE_VERSION = "broad-question-v8"
WEB_RETRIEVAL_CACHE_VERSION = "control-terminology-v2"

DEMO_ERROR_RESPONSE = (
    "The ISM pipeline hit a temporary retrieval or generation error. "
    "Please retry the question in a moment."
)
RETRIEVAL_EMPTY_RESPONSE = (
    "I could not retrieve relevant ISM chunks for this question. "
    "Try asking with a specific ISM topic, control ID, or keyword."
)

_embedding_model_lock = Lock()

_cache_lock = Lock()
_cache: dict[str, dict] = {}


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                _embedding_model = load_embedding_model()
    return _embedding_model


def _get_supabase():
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = get_supabase_client()
    return _supabase_client


class ChatRequest(BaseModel):
    question: str
    multi_query: bool | None = None


class ChunkResponse(BaseModel):
    content: str
    control_id: str | None = None
    category: str | None = None
    sub_topic: str | None = None
    similarity: float | None = None
    rerank_score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    chunks: list[ChunkResponse]
    query_variants: list[str] = []
    guardrail_stage: str | None = None
    guardrail_details: dict | None = None
    retrieval_trace: dict | None = None


def _chunk_summary(chunk: dict, preview_chars: int = 150) -> dict:
    similarity = chunk.get("similarity")
    rrf_score = chunk.get("rrf_score")
    rerank_score = chunk.get("rerank_score")

    def safe_round(value, places: int):
        try:
            return round(float(value), places)
        except (TypeError, ValueError):
            return None

    return {
        "id": chunk.get("id"),
        "control_id": chunk.get("control_id", "N/A"),
        "category": chunk.get("category", ""),
        "sub_topic": chunk.get("sub_topic", ""),
        "similarity": safe_round(similarity, 4),
        "rrf_score": safe_round(rrf_score, 4),
        "rerank_score": safe_round(rerank_score, 3),
        "content_preview": chunk.get("content", "")[:preview_chars],
    }


def _cache_get(key: str):
    if not WEB_CACHE_ENABLED:
        return None

    now = time.monotonic()
    with _cache_lock:
        item = _cache.get(key)
        if not item:
            return None
        if now - item["created_at"] > WEB_CACHE_TTL_SECONDS:
            _cache.pop(key, None)
            return None
        return deepcopy(item["value"])


def _cache_set(key: str, value):
    if not WEB_CACHE_ENABLED:
        return

    with _cache_lock:
        if WEB_CACHE_MAX_ENTRIES > 0 and len(_cache) >= WEB_CACHE_MAX_ENTRIES:
            oldest_key = min(_cache, key=lambda item_key: _cache[item_key]["created_at"])
            _cache.pop(oldest_key, None)
        _cache[key] = {
            "created_at": time.monotonic(),
            "value": deepcopy(value),
        }


def _normalise_question(question: str) -> str:
    return " ".join(question.lower().split())


def _dependency_error_details(stage: str) -> dict:
    return {
        "stage": "error",
        "passed": False,
        "error_stage": stage,
        "message": DEMO_ERROR_RESPONSE,
    }


def _control_not_found_response(missing_control_ids: list[str]) -> str:
    controls = ", ".join(missing_control_ids)
    return (
        f"I could not find {controls} in the indexed ISM chunks. "
        "Please check the control ID or ask about the topic instead."
    )


def _use_multi_query(req: ChatRequest) -> bool:
    if req.multi_query is None:
        return MULTI_QUERY_ENABLED
    return bool(req.multi_query) and MULTI_QUERY_ENABLED


def _expand_query_cached(question: str, use_multi_query: bool) -> list[str]:
    if not use_multi_query:
        return [question]

    cache_key = (
        "expand:"
        f"{QUERY_EXPANSION_PROVIDER}:{QUERY_EXPANSION_MODEL}:{MULTI_QUERY_COUNT}:"
        f"{_normalise_question(question)}"
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    queries = expand_query(question)
    _cache_set(cache_key, queries)
    return queries


def _embed_query_cached(model, query_text: str) -> list[float]:
    cache_key = f"embed:{EMBEDDING_MODEL_NAME}:{query_text}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    query_embedding = embed_query(model, query_text)
    _cache_set(cache_key, query_embedding)
    return query_embedding


def _candidate_score(chunk: dict) -> float:
    rrf = chunk.get("rrf_score")
    similarity = chunk.get("similarity")
    try:
        if rrf is not None:
            return float(rrf)
        if similarity is not None:
            return float(similarity)
    except (TypeError, ValueError):
        pass
    return 0.0


def _is_pinned_candidate(chunk: dict) -> bool:
    if chunk.get("_retrieval_pin"):
        return True
    try:
        return float(chunk.get("rrf_score") or 0) >= 1.0
    except (TypeError, ValueError):
        return False


def _limit_rerank_candidates(chunks: list[dict]) -> list[dict]:
    if WEB_RERANK_CANDIDATE_LIMIT <= 0 or len(chunks) <= WEB_RERANK_CANDIDATE_LIMIT:
        return chunks

    pinned = [chunk for chunk in chunks if _is_pinned_candidate(chunk)]
    pinned_ids = {chunk.get("id") for chunk in pinned}
    remaining = [chunk for chunk in chunks if chunk.get("id") not in pinned_ids]
    remaining = sorted(remaining, key=_candidate_score, reverse=True)
    return (pinned + remaining)[:WEB_RERANK_CANDIDATE_LIMIT]


def _rerank_cached(question: str, chunks: list[dict]) -> list[dict]:
    chunk_ids = ",".join(str(c.get("id", "")) for c in chunks)
    cache_key = f"rerank:{RERANKER_MODEL}:{_normalise_question(question)}:{RERANK_TOP_K}:{chunk_ids}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    full_ranked = rerank(question, deepcopy(chunks), top_k=len(chunks))
    pinned = [chunk for chunk in full_ranked if _is_pinned_candidate(chunk)]
    unpinned = [chunk for chunk in full_ranked if not _is_pinned_candidate(chunk)]

    reranked = []
    seen_ids = set()
    for chunk in pinned[:2] + unpinned:
        chunk_id = chunk.get("id")
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        reranked.append(chunk)
        if len(reranked) >= RERANK_TOP_K:
            break

    _cache_set(cache_key, reranked)
    return reranked


def _generate_answer_cached(question: str, chunks: list[dict]) -> str:
    chunk_ids = ",".join(str(c.get("id", "")) for c in chunks)
    scores = ",".join(f"{float(c.get('rerank_score') or 0):.4f}" for c in chunks)
    cache_key = (
        f"answer:{WEB_ANSWER_CACHE_VERSION}:{LLM_PROVIDER}:{LLM_MODEL_NAME}:"
        f"{_normalise_question(question)}:{chunk_ids}:{scores}"
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    answer = generate_answer(question, chunks)
    if answer == GENERATION_ERROR_RESPONSE:
        return answer

    _cache_set(cache_key, answer)
    return answer


def _retrieve_with_trace(client, model, queries: list[str]) -> tuple[list[dict], dict]:
    cache_key = "retrieve:" + json.dumps(
        {
            "version": WEB_RETRIEVAL_CACHE_VERSION,
            "queries": queries,
            "match_count": INITIAL_RETRIEVE_COUNT,
            "parallel": WEB_RETRIEVAL_PARALLEL_ENABLED,
            "rrf_k": RRF_K,
        },
        sort_keys=True,
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    seen_ids = set()
    merged = []
    per_query = []
    total_returned = 0
    warnings = []
    requested_control_ids = control_ids_from_question(queries[0]) if queries else []
    matched_control_ids = set()

    if queries:
        try:
            control_results = control_id_search(client, queries[0])
        except Exception as exc:
            control_results = []
            warnings.append(f"Control ID lookup failed: {exc}")

        new_unique = 0
        for chunk in control_results:
            chunk_id = chunk.get("id")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(chunk)
                new_unique += 1
            control_id = chunk.get("control_id")
            if control_id:
                matched_control_ids.add(control_id)

        if control_results or requested_control_ids:
            total_returned += len(control_results)
            per_query.append({
                "label": "Control ID lookup",
                "query_index": 0,
                "query": queries[0],
                "is_original": False,
                "returned_count": len(control_results),
                "new_unique_count": new_unique,
                "duplicate_count": len(control_results) - new_unique,
                "chunks": [_chunk_summary(c) for c in control_results[:5]],
            })

        try:
            terminology_results = terminology_search(client, queries[0])
        except Exception as exc:
            terminology_results = []
            warnings.append(f"Terminology fallback failed: {exc}")

        new_unique = 0
        for chunk in terminology_results:
            chunk_id = chunk.get("id")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(chunk)
                new_unique += 1

        if terminology_results:
            total_returned += len(terminology_results)
            per_query.append({
                "label": "Terminology fallback",
                "query_index": 0,
                "query": queries[0],
                "is_original": False,
                "returned_count": len(terminology_results),
                "new_unique_count": new_unique,
                "duplicate_count": len(terminology_results) - new_unique,
                "chunks": [_chunk_summary(c) for c in terminology_results[:5]],
            })

    indexed_queries = [
        (index, query_text, _embed_query_cached(model, query_text))
        for index, query_text in enumerate(queries)
    ]

    def run_hybrid_search(index_query_embedding: tuple[int, str, list[float]], use_fresh_client: bool = False):
        index, query_text, query_embedding = index_query_embedding
        search_client = get_supabase_client() if use_fresh_client else client
        results = hybrid_search(
            search_client,
            query_text,
            query_embedding,
            match_count=INITIAL_RETRIEVE_COUNT,
        )
        return index, query_text, results

    parallel_requested = WEB_RETRIEVAL_PARALLEL_ENABLED and len(indexed_queries) > 1
    parallel_used = False
    if WEB_RETRIEVAL_PARALLEL_ENABLED and len(indexed_queries) > 1:
        try:
            workers = max(1, min(WEB_RETRIEVAL_MAX_WORKERS, len(indexed_queries)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                query_results = list(
                    executor.map(
                        lambda item: run_hybrid_search(item, use_fresh_client=True),
                        indexed_queries,
                    )
                )
            parallel_used = True
        except Exception as exc:
            print(f"WARNING: Parallel hybrid search failed ({exc}). Falling back to sequential search.")
            warnings.append(f"Parallel hybrid search failed; sequential fallback used: {exc}")
            query_results = []
            for item in indexed_queries:
                try:
                    query_results.append(run_hybrid_search(item, use_fresh_client=False))
                except Exception as item_exc:
                    warnings.append(f"Hybrid search failed for query {item[0] + 1}: {item_exc}")
    else:
        query_results = []
        for item in indexed_queries:
            try:
                query_results.append(run_hybrid_search(item, use_fresh_client=False))
            except Exception as item_exc:
                warnings.append(f"Hybrid search failed for query {item[0] + 1}: {item_exc}")

    for index, query_text, results in query_results:
        total_returned += len(results)

        new_unique = 0
        duplicate_count = 0
        for chunk in results:
            chunk_id = chunk.get("id")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(chunk)
                new_unique += 1
            else:
                duplicate_count += 1

        per_query.append({
            "query_index": index,
            "query": query_text,
            "is_original": index == 0,
            "returned_count": len(results),
            "new_unique_count": new_unique,
            "duplicate_count": duplicate_count,
            "chunks": [_chunk_summary(c) for c in results[:5]],
        })

    value = (merged, {
        "method": "hybrid (BM25 + vector + RRF)",
        "queries_used": len(queries),
        "candidates_before_dedupe": total_returned,
        "candidates_after_dedupe": len(merged),
        "parallel_enabled": WEB_RETRIEVAL_PARALLEL_ENABLED,
        "parallel_requested": parallel_requested,
        "parallel_used": parallel_used,
        "cache_enabled": WEB_CACHE_ENABLED,
        "warnings": warnings,
        "control_ids_requested": requested_control_ids,
        "control_ids_found": sorted(matched_control_ids),
        "missing_control_ids": [
            control_id for control_id in requested_control_ids
            if control_id not in matched_control_ids
        ],
        "per_query": per_query,
    })
    _cache_set(cache_key, value)
    return value


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        return ChatResponse(answer="Please enter a question.", chunks=[])

    # Stage 1: Pre-filter guardrail
    passed, reason = pre_filter(question)
    if not passed:
        return ChatResponse(
            answer=OOS_REFUSAL,
            chunks=[],
            guardrail_stage="pre_filter",
            guardrail_details={
                "stage": "pre_filter",
                "passed": False,
                "reason": reason,
                "threshold": OOS_RERANK_THRESHOLD,
            },
        )

    clarification = query_specificity_check(question)
    if clarification:
        return ChatResponse(
            answer=clarification,
            chunks=[],
            guardrail_stage="query_specificity",
            guardrail_details={
                "stage": "query_specificity",
                "passed": False,
                "reason": "question_too_broad",
            },
        )

    try:
        model = _get_embedding_model()
        client = _get_supabase()
        use_multi_query = _use_multi_query(req)

        # Multi-query expansion
        queries = _expand_query_cached(question, use_multi_query)

        # Retrieve each query variant, then merge/deduplicate candidates.
        raw_chunks, retrieval_trace = _retrieve_with_trace(client, model, queries)
        missing_control_ids = retrieval_trace.get("missing_control_ids") or []
        if missing_control_ids:
            return ChatResponse(
                answer=_control_not_found_response(missing_control_ids),
                chunks=[],
                query_variants=queries,
                guardrail_stage="control_id_lookup",
                guardrail_details={
                    "stage": "control_id_lookup",
                    "passed": False,
                    "missing_control_ids": missing_control_ids,
                },
                retrieval_trace=retrieval_trace,
            )

        if not raw_chunks:
            return ChatResponse(
                answer=RETRIEVAL_EMPTY_RESPONSE,
                chunks=[],
                query_variants=queries,
                guardrail_stage="retrieval_empty",
                guardrail_details={
                    "stage": "retrieval_empty",
                    "passed": False,
                },
                retrieval_trace=retrieval_trace,
            )

        rerank_input = _limit_rerank_candidates(raw_chunks)

        # Cross-encoder reranking
        reranked = _rerank_cached(question, rerank_input)

        # Stage 2: Rerank threshold guardrail
        passed, max_score = rerank_threshold_check(reranked, OOS_RERANK_THRESHOLD)
        if not passed:
            return ChatResponse(
                answer=OOS_REFUSAL,
                chunks=[],
                query_variants=queries,
                guardrail_stage="rerank_threshold",
                guardrail_details={
                    "stage": "rerank_threshold",
                    "passed": False,
                    "max_rerank_score": round(max_score, 3),
                    "threshold": OOS_RERANK_THRESHOLD,
                },
                retrieval_trace=retrieval_trace,
            )

        # Generate answer
        answer = _generate_answer_cached(question, reranked)

        chunk_responses = []
        for c in reranked:
            chunk_responses.append(ChunkResponse(
                content=c.get("content", ""),
                control_id=c.get("control_id"),
                category=c.get("category"),
                sub_topic=c.get("sub_topic"),
                similarity=c.get("similarity"),
                rerank_score=c.get("rerank_score"),
            ))

        return ChatResponse(
            answer=answer,
            chunks=chunk_responses,
            query_variants=queries,
            guardrail_details={
                "stage": "passed",
                "passed": True,
                "max_rerank_score": round(max_score, 3),
                "threshold": OOS_RERANK_THRESHOLD,
            },
            retrieval_trace=retrieval_trace,
        )
    except Exception as exc:
        print(f"WARNING: /chat pipeline failed: {exc}")
        return ChatResponse(
            answer=DEMO_ERROR_RESPONSE,
            chunks=[],
            guardrail_stage="error",
            guardrail_details=_dependency_error_details("chat"),
        )


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _guarded_sse_events(events):
    try:
        for event in events:
            yield event
    except Exception as exc:
        print(f"WARNING: /pipeline/stream failed: {exc}")
        yield _sse_event("error", {"message": DEMO_ERROR_RESPONSE})
        yield _sse_event("done", {
            "blocked_at": "error",
            "answer": DEMO_ERROR_RESPONSE,
        })


@router.post("/pipeline/stream")
async def pipeline_stream(req: ChatRequest):
    question = req.question.strip()
    if not question:
        return StreamingResponse(
            iter([_sse_event("error", {"message": "Empty question"})]),
            media_type="text/event-stream",
        )

    def generate():
        total_start = time.time()

        # Step 1: Pre-filter guardrail
        t0 = time.time()
        passed, reason = pre_filter(question)
        elapsed = round((time.time() - t0) * 1000, 1)
        yield _sse_event("pre_filter", {
            "step": "OOS Pre-Filter",
            "time_ms": elapsed,
            "output": {"passed": passed, "reason": reason, "question": question},
        })

        if not passed:
            yield _sse_event("done", {
                "total_time_ms": round((time.time() - total_start) * 1000, 1),
                "blocked_at": "pre_filter",
                "answer": OOS_REFUSAL,
            })
            return

        clarification = query_specificity_check(question)
        if clarification:
            yield _sse_event("done", {
                "total_time_ms": round((time.time() - total_start) * 1000, 1),
                "blocked_at": "query_specificity",
                "answer": clarification,
            })
            return

        model = _get_embedding_model()
        client = _get_supabase()
        use_multi_query = _use_multi_query(req)

        # Step 2: Query embedding
        t0 = time.time()
        _embed_query_cached(model, question)
        elapsed = round((time.time() - t0) * 1000, 1)
        yield _sse_event("embedding", {
            "step": "Query Embedding",
            "time_ms": elapsed,
            "output": {
                "model": "nomic-embed-text-v1.5",
                "dimension": 768,
                "prefix": "search_query:",
            },
        })

        # Step 3: Multi-query expansion
        t0 = time.time()
        queries = _expand_query_cached(question, use_multi_query)
        elapsed = round((time.time() - t0) * 1000, 1)
        yield _sse_event("query_expansion", {
            "step": "Multi-Query Expansion",
            "time_ms": elapsed,
            "output": {
                "enabled": use_multi_query,
                "provider": QUERY_EXPANSION_PROVIDER,
                "model": QUERY_EXPANSION_MODEL,
                "original": question,
                "variants": queries[1:] if len(queries) > 1 else [],
                "total_queries": len(queries),
            },
        })

        # Step 4: Hybrid search (multi-query or single)
        t0 = time.time()
        raw_chunks, retrieval_trace = _retrieve_with_trace(client, model, queries)
        elapsed = round((time.time() - t0) * 1000, 1)

        search_output = {
            "method": "hybrid (BM25 + vector + RRF)",
            "queries_used": len(queries),
            "candidates_returned": len(raw_chunks),
            "candidates_before_dedupe": retrieval_trace["candidates_before_dedupe"],
            "candidates_after_dedupe": retrieval_trace["candidates_after_dedupe"],
            "parallel_enabled": retrieval_trace.get("parallel_enabled", False),
            "parallel_used": retrieval_trace.get("parallel_used", False),
            "cache_enabled": retrieval_trace.get("cache_enabled", False),
            "warnings": retrieval_trace.get("warnings", []),
            "control_ids_requested": retrieval_trace.get("control_ids_requested", []),
            "missing_control_ids": retrieval_trace.get("missing_control_ids", []),
            "rrf_k": RRF_K,
            "per_query": retrieval_trace["per_query"],
            "chunks": [
                {
                    "control_id": c.get("control_id", "N/A"),
                    "category": c.get("category", ""),
                    "similarity": round(c.get("similarity", 0), 4),
                    "rrf_score": round(c.get("rrf_score", 0), 4),
                    "content_preview": c.get("content", "")[:150],
                }
                for c in raw_chunks[:10]
            ],
        }
        yield _sse_event("hybrid_search", {
            "step": "Hybrid Search",
            "time_ms": elapsed,
            "output": search_output,
        })

        missing_control_ids = retrieval_trace.get("missing_control_ids") or []
        if missing_control_ids:
            yield _sse_event("done", {
                "total_time_ms": round((time.time() - total_start) * 1000, 1),
                "blocked_at": "control_id_lookup",
                "answer": _control_not_found_response(missing_control_ids),
            })
            return

        if not raw_chunks:
            yield _sse_event("done", {
                "total_time_ms": round((time.time() - total_start) * 1000, 1),
                "blocked_at": "retrieval_empty",
                "answer": RETRIEVAL_EMPTY_RESPONSE,
            })
            return

        # Step 5: Cross-encoder reranking
        t0 = time.time()
        rerank_input = _limit_rerank_candidates(raw_chunks)
        reranked = _rerank_cached(question, rerank_input)
        elapsed = round((time.time() - t0) * 1000, 1)
        yield _sse_event("reranking", {
            "step": "Cross-Encoder Reranking",
            "time_ms": elapsed,
            "output": {
                "model": "ms-marco-MiniLM-L-6-v2",
                "input_count": len(rerank_input),
                "candidate_count_before_limit": len(raw_chunks),
                "candidate_limit": WEB_RERANK_CANDIDATE_LIMIT,
                "output_count": len(reranked),
                "chunks": [
                    {
                        "control_id": c.get("control_id", "N/A"),
                        "category": c.get("category", ""),
                        "rerank_score": round(c.get("rerank_score", 0), 3),
                        "content_preview": c.get("content", "")[:150],
                    }
                    for c in reranked
                ],
            },
        })

        # Step 6: Rerank threshold guardrail
        t0 = time.time()
        passed, max_score = rerank_threshold_check(reranked, OOS_RERANK_THRESHOLD)
        elapsed = round((time.time() - t0) * 1000, 1)
        yield _sse_event("guardrail", {
            "step": "OOS Rerank Guardrail",
            "time_ms": elapsed,
            "output": {
                "max_rerank_score": round(max_score, 3),
                "threshold": OOS_RERANK_THRESHOLD,
                "passed": passed,
            },
        })

        if not passed:
            yield _sse_event("done", {
                "total_time_ms": round((time.time() - total_start) * 1000, 1),
                "blocked_at": "rerank_threshold",
                "answer": OOS_REFUSAL,
            })
            return

        # Step 7: LLM generation
        t0 = time.time()
        answer = _generate_answer_cached(question, reranked)
        elapsed = round((time.time() - t0) * 1000, 1)

        context_sent = []
        for i, c in enumerate(reranked, 1):
            context_sent.append({
                "chunk_index": i,
                "control_id": c.get("control_id", "N/A"),
                "content_preview": c.get("content", "")[:200],
            })

        yield _sse_event("generation", {
            "step": "LLM Generation",
            "time_ms": elapsed,
            "output": {
                "model": "llama-3.1-8b-instant",
                "provider": "groq",
                "context_chunks_sent": len(reranked),
                "context_sent": context_sent,
                "answer": answer,
            },
        })

        # Done
        yield _sse_event("done", {
            "total_time_ms": round((time.time() - total_start) * 1000, 1),
            "answer": answer,
        })

    return StreamingResponse(_guarded_sse_events(generate()), media_type="text/event-stream")
