import json
import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.embeddings import load_embedding_model, embed_query
from src.retrieval import hybrid_search, terminology_search
from src.reranking import rerank
from src.llm import generate_answer
from src.query_expansion import expand_query
from src.guardrail import pre_filter, rerank_threshold_check, OOS_REFUSAL
from src.supabase_utils import get_supabase_client
from src.config import (
    INITIAL_RETRIEVE_COUNT,
    MULTI_QUERY_ENABLED,
    OOS_RERANK_THRESHOLD,
    QUERY_EXPANSION_MODEL,
    QUERY_EXPANSION_PROVIDER,
    RERANK_TOP_K,
)

router = APIRouter()

_embedding_model = None
_supabase_client = None


def _get_embedding_model():
    global _embedding_model
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
    return {
        "id": chunk.get("id"),
        "control_id": chunk.get("control_id", "N/A"),
        "category": chunk.get("category", ""),
        "sub_topic": chunk.get("sub_topic", ""),
        "similarity": round(chunk.get("similarity", 0), 4),
        "rrf_score": round(chunk.get("rrf_score", 0), 4),
        "rerank_score": round(chunk.get("rerank_score", 0), 3)
        if chunk.get("rerank_score") is not None
        else None,
        "content_preview": chunk.get("content", "")[:preview_chars],
    }


def _retrieve_with_trace(client, model, queries: list[str]) -> tuple[list[dict], dict]:
    seen_ids = set()
    merged = []
    per_query = []
    total_returned = 0

    if queries:
        terminology_results = terminology_search(client, queries[0])
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

    for index, query_text in enumerate(queries):
        query_embedding = embed_query(model, query_text)
        results = hybrid_search(
            client,
            query_text,
            query_embedding,
            match_count=INITIAL_RETRIEVE_COUNT,
        )
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

    return merged, {
        "method": "hybrid (BM25 + vector + RRF)",
        "queries_used": len(queries),
        "candidates_before_dedupe": total_returned,
        "candidates_after_dedupe": len(merged),
        "per_query": per_query,
    }


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

    model = _get_embedding_model()
    client = _get_supabase()

    # Multi-query expansion
    queries = expand_query(question)

    # Retrieve each query variant, then merge/deduplicate candidates.
    raw_chunks, retrieval_trace = _retrieve_with_trace(client, model, queries)

    # Cross-encoder reranking
    reranked = rerank(question, raw_chunks, top_k=RERANK_TOP_K)

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
    answer = generate_answer(question, reranked)

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


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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

        model = _get_embedding_model()
        client = _get_supabase()

        # Step 2: Query embedding
        t0 = time.time()
        embed_query(model, question)
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
        queries = expand_query(question)
        elapsed = round((time.time() - t0) * 1000, 1)
        yield _sse_event("query_expansion", {
            "step": "Multi-Query Expansion",
            "time_ms": elapsed,
            "output": {
                "enabled": MULTI_QUERY_ENABLED,
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
            "rrf_k": 50,
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

        # Step 5: Cross-encoder reranking
        t0 = time.time()
        reranked = rerank(question, raw_chunks, top_k=RERANK_TOP_K)
        elapsed = round((time.time() - t0) * 1000, 1)
        yield _sse_event("reranking", {
            "step": "Cross-Encoder Reranking",
            "time_ms": elapsed,
            "output": {
                "model": "ms-marco-MiniLM-L-6-v2",
                "input_count": len(raw_chunks),
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
        answer = generate_answer(question, reranked)
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

    return StreamingResponse(generate(), media_type="text/event-stream")
