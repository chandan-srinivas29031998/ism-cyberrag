from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

from rag.chunking import chunk_document
from rag.config import Settings
from rag.embeddings import Embedder
from rag.logging_config import configure_logging
from rag.parsing import parse_pdf
from rag.supabase_store import SupabaseStore

logger = logging.getLogger(__name__)


def title_from_filename(path: str) -> str:
    name = Path(path).stem
    return name.replace("_", " ").replace("-", " ").strip()


def run_ingestion(pdf_dir: str) -> None:
    settings = Settings.from_env()
    configure_logging(logging.INFO)

    store = SupabaseStore(url=settings.supabase_url, service_key=settings.supabase_service_key)
    embedder = Embedder(model_id=settings.embedding_model_id, device="cpu")

    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in directory: {pdf_dir}")

    for pdf_path in pdf_paths:
        source_file = os.path.basename(pdf_path)
        title = title_from_filename(pdf_path)

        logger.info("Parsing PDF: %s", source_file)
        parsed = parse_pdf(pdf_path)
        logger.info("Parsed successfully: %s | extractor=%s | pages=%d", source_file, parsed.extractor, parsed.page_count)

        logger.info("Upserting document row for: %s", source_file)
        doc_id = store.upsert_document(
            title=title,
            source_file=source_file,
            metadata={
                "extractor": parsed.extractor,
                "page_count": parsed.page_count,
            },
        )
        logger.info("Document upserted successfully: %s | doc_id=%s", source_file, doc_id)

        logger.info("Chunking document: %s", source_file)
        chunks = chunk_document(
            pages=parsed.pages,
            source_file=source_file,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        logger.info("Chunking complete: %s -> %d chunks", source_file, len(chunks))

        logger.info("Generating embeddings for: %s", source_file)
        vectors = embedder.embed_texts([c.content for c in chunks])
        logger.info("Embeddings generated: %s -> %d vectors", source_file, len(vectors))

        rows = []
        for chunk, vector in zip(chunks, vectors):
            rows.append(
                {
                    "document_id": doc_id,
                    "content": chunk.content,
                    "control_id": None,
                    "category": None,
                    "sub_topic": None,
                    "applicability": None,
                    "essential_8": None,
                    "revision": None,
                    "embedding": vector,
                    "metadata": chunk.metadata,
                }
            )

        logger.info("Inserting chunk rows into Supabase for: %s", source_file)
        store.insert_chunks(rows, batch_size=settings.insert_batch_size)
        logger.info("Chunk insert completed for: %s", source_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", required=True, help="Directory containing ISM guideline PDFs.")
    args = parser.parse_args()
    run_ingestion(args.pdf_dir)
