import re

from src.config import MATCH_COUNT, INITIAL_RETRIEVE_COUNT, RRF_K


CHUNK_SELECT_COLUMNS = (
    "id,content,control_id,category,sub_topic,applicability,essential_8,revision"
)


def _definition_terms(question: str) -> list[str]:
    """Extract likely glossary terms from definition-style questions."""
    cleaned = question.strip().rstrip("?.!")
    patterns = [
        r"(?i)\bdefinition of (?:a |an |the )?(.+)$",
        r"(?i)\bdefine (?:a |an |the )?(.+)$",
        r"(?i)\bhow does .+ define (?:a |an |the )?(.+)$",
        r"(?i)\bwhat does (?:a |an |the )?(.+?) mean$",
        r"(?i)\bwhat is (?:a |an )(.+)$",
    ]

    terms = []
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue

        term = match.group(1).strip()
        term = re.sub(r"\b(in|within|according to)\b.*$", "", term, flags=re.IGNORECASE).strip()
        term = re.sub(r"\s+", " ", term)
        if term:
            terms.append(term)

        without_parentheses = re.sub(r"\s*\([^)]*\)", "", term).strip()
        if without_parentheses and without_parentheses != term:
            terms.append(without_parentheses)

    deduped = []
    for term in terms:
        key = term.lower()
        if key not in {t.lower() for t in deduped}:
            deduped.append(term)
    return deduped


def terminology_search(client, question: str, limit: int = 2) -> list[dict]:
    """
    Fetch glossary chunks for definition-style questions.

    Hybrid/vector retrieval can miss glossary definitions because terminology
    chunks contain many compact definitions. This read-only fallback only runs
    for definition-like questions and lets the cross-encoder decide whether the
    glossary chunk belongs in the final context.
    """
    results = []
    seen_ids = set()

    for term in _definition_terms(question):
        response = (
            client.table("chunks")
            .select(CHUNK_SELECT_COLUMNS)
            .ilike("category", "%Cyber security terminology%")
            .ilike("content", f"%{term}%")
            .limit(limit)
            .execute()
        )

        for row in response.data or []:
            chunk_id = row.get("id")
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            row.setdefault("similarity", 0.0)
            row.setdefault("rrf_score", 0.0)
            results.append(row)

    return results


def match_chunks(client, query_embedding: list[float], match_count: int = MATCH_COUNT) -> list[dict]:
    """
    Sprint 1 vector-only search. Kept for backward compatibility.
    """
    response = client.rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": match_count,
        },
    ).execute()
    return response.data or []


def hybrid_search(
    client,
    query_text: str,
    query_embedding: list[float],
    match_count: int = INITIAL_RETRIEVE_COUNT,
    full_text_weight: float = 1.0,
    semantic_weight: float = 1.0,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """
    Hybrid search combining vector similarity and BM25 full-text search
    using Reciprocal Rank Fusion (RRF).

    Returns a list of dicts with keys:
        id, content, control_id, category, sub_topic, applicability,
        essential_8, revision, similarity, rrf_score
    """
    response = client.rpc(
        "hybrid_search",
        {
            "query_text": query_text,
            "query_embedding": query_embedding,
            "match_count": match_count,
            "full_text_weight": full_text_weight,
            "semantic_weight": semantic_weight,
            "rrf_k": rrf_k,
        },
    ).execute()
    return response.data or []


def multi_query_retrieve(
    client,
    embed_fn,
    queries: list[str],
    match_count: int = INITIAL_RETRIEVE_COUNT,
) -> list[dict]:
    """
    Run hybrid search for each query variant, merge results, deduplicate by chunk ID.
    Returns a single list of unique chunks (typically 15-25 before reranking).
    """
    seen_ids = set()
    merged = []

    if queries:
        for chunk in terminology_search(client, queries[0]):
            chunk_id = chunk.get("id")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(chunk)

    for query_text in queries:
        query_embedding = embed_fn(query_text)
        results = hybrid_search(client, query_text, query_embedding, match_count=match_count)
        for chunk in results:
            chunk_id = chunk.get("id")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(chunk)

    return merged
