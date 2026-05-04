import json
import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.embeddings import load_embedding_model, embed_query
from src.retrieval import hybrid_search, multi_query_retrieve
from src.reranking import rerank
from src.llm import generate_answer
from src.query_expansion import expand_query
from src.guardrail import pre_filter, rerank_threshold_check, OOS_REFUSAL
from src.supabase_utils import get_supabase_client
from src.config import INITIAL_RETRIEVE_COUNT, RERANK_TOP_K, OOS_RERANK_THRESHOLD, MULTI_QUERY_ENABLED

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
        )

    model = _get_embedding_model()
    client = _get_supabase()

    # Multi-query expansion
    queries = expand_query(question)

    # Retrieve with multi-query
    if len(queries) > 1:
        raw_chunks = multi_query_retrieve(
            client,
            lambda q: embed_query(model, q),
            queries,
            match_count=INITIAL_RETRIEVE_COUNT,
        )
    else:
        query_embedding = embed_query(model, question)
        raw_chunks = hybrid_search(client, question, query_embedding, match_count=INITIAL_RETRIEVE_COUNT)

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
        query_embedding = embed_query(model, question)
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
                "original": question,
                "variants": queries[1:] if len(queries) > 1 else [],
                "total_queries": len(queries),
            },
        })

        # Step 4: Hybrid search (multi-query or single)
        t0 = time.time()
        if len(queries) > 1:
            raw_chunks = multi_query_retrieve(
                client,
                lambda q: embed_query(model, q),
                queries,
                match_count=INITIAL_RETRIEVE_COUNT,
            )
        else:
            raw_chunks = hybrid_search(client, question, query_embedding, match_count=INITIAL_RETRIEVE_COUNT)
        elapsed = round((time.time() - t0) * 1000, 1)

        search_output = {
            "method": "hybrid (BM25 + vector + RRF)",
            "queries_used": len(queries),
            "candidates_returned": len(raw_chunks),
            "rrf_k": 50,
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
