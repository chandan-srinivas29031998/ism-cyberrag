from threading import Lock

from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL, RERANK_TOP_K


_model = None
_model_lock = Lock()


def load_reranker() -> CrossEncoder:
    """Loads the cross-encoder model. Caches it after first load."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Re-scores retrieved chunks using a cross-encoder and returns the top-k.

    The cross-encoder takes (query, chunk_content) pairs and produces a relevance
    score that captures the semantic relationship better than bi-encoder similarity.

    Each returned chunk gets an added 'rerank_score' field.
    """
    if not chunks:
        return []

    chunks = [chunk for chunk in chunks if chunk.get("content")]
    if not chunks:
        return []

    model = load_reranker()
    pairs = [[query, chunk["content"]] for chunk in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
