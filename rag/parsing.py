from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedPDF:
    pages: List[str]
    extractor: str  # "pypdf" or "pymupdf"
    page_count: int


def _looks_like_bad_extraction(pages: List[str]) -> bool:
    # Heuristic: too little text overall or too many near-empty pages
    if not pages:
        return True
    total_chars = sum(len(p.strip()) for p in pages)
    empty_pages = sum(1 for p in pages if len(p.strip()) < 50)
    if total_chars < 2000:
        return True
    if empty_pages / max(len(pages), 1) > 0.4:
        return True
    return False


def parse_pdf(path: str) -> ParsedPDF:
    """
    Primary parser: pypdf
    Fallback: pymupdf when extraction quality is poor.
    """
    pages: List[str] = []
    extractor = "pypdf"

    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(path)
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
    except Exception as e:
        logger.warning("pypdf failed on %s (%s). Will try pymupdf.", path, e)
        pages = []

    if pages and not _looks_like_bad_extraction(pages):
        return ParsedPDF(pages=pages, extractor=extractor, page_count=len(pages))

    # Fallback to pymupdf
    try:
        import fitz  # pymupdf
        extractor = "pymupdf"
        pages = []
        doc = fitz.open(path)
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pages.append(page.get_text("text") or "")
        return ParsedPDF(pages=pages, extractor=extractor, page_count=len(pages))
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF with both pypdf and pymupdf: {path}. Error: {e}") from e
