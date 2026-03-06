from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    content: str
    similarity: float
    metadata: Dict[str, Any]
    control_id: Optional[str] = None


def to_retrieved_chunks(rows: List[Dict[str, Any]]) -> List[RetrievedChunk]:
    out: List[RetrievedChunk] = []
    for r in rows:
        out.append(
            RetrievedChunk(
                id=str(r.get("id")),
                content=r.get("content", ""),
                similarity=float(r.get("similarity", 0.0)),
                metadata=r.get("metadata") or {},
                control_id=r.get("control_id"),
            )
        )
    return out
