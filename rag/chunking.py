from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Chunk:
    content: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: Dict


def _build_page_offsets(pages: List[str], separator: str = "\n\n") -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Returns:
      full_text: concatenated pages
      spans: list of (start_char, end_char, page_number_1_indexed)
    """
    spans: List[Tuple[int, int, int]] = []
    parts: List[str] = []
    cursor = 0

    for idx, page in enumerate(pages, start=1):
        page_text = page.strip()
        parts.append(page_text)

        start = cursor
        end = cursor + len(page_text)
        spans.append((start, end, idx))

        cursor = end + len(separator)

    full_text = separator.join(parts)
    return full_text, spans


def _page_range_for_span(spans: List[Tuple[int, int, int]], start: int, end: int) -> Tuple[int, int]:
    pages = []
    for s, e, p in spans:
        if e < start:
            continue
        if s > end:
            break
        # overlap exists
        if not (e <= start or s >= end):
            pages.append(p)
    if not pages:
        return (1, 1)
    return (min(pages), max(pages))


def chunk_document(
    pages: List[str],
    source_file: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    full_text, spans = _build_page_offsets(pages)
    step = chunk_size - chunk_overlap

    chunks: List[Chunk] = []
    idx = 0
    for start in range(0, max(len(full_text), 1), step):
        end = min(start + chunk_size, len(full_text))
        if start >= end:
            break

        content = full_text[start:end].strip()
        if len(content) < 50:
            continue

        p_start, p_end = _page_range_for_span(spans, start, end)

        chunks.append(
            Chunk(
                content=content,
                chunk_index=idx,
                char_start=start,
                char_end=end,
                metadata={
                    "source_file": source_file,
                    "page_start": p_start,
                    "page_end": p_end,
                },
            )
        )
        idx += 1

        if end >= len(full_text):
            break

    return chunks
