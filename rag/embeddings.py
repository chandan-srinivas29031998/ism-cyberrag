from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Embedder:
    model_id: str
    normalize: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        # Nomic embedding models commonly require trust_remote_code in some environments.
        self.model = SentenceTransformer(self.model_id, device=self.device, trust_remote_code=True)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        vectors_list = [v.tolist() for v in vectors]
        self._assert_dim(vectors_list)
        return vectors_list

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode([text], normalize_embeddings=self.normalize)[0].tolist()
        self._assert_dim([vec])
        return vec

    @staticmethod
    def _assert_dim(vectors: List[List[float]], expected: int = 768) -> None:
        if not vectors:
            raise ValueError("No vectors produced.")
        if any(len(v) != expected for v in vectors):
            bad = {len(v) for v in vectors}
            raise ValueError(f"Embedding dimensionality mismatch. Expected {expected}, got {bad}.")
