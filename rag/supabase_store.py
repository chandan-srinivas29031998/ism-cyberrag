from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from supabase import create_client

logger = logging.getLogger(__name__)


class SupabaseStore:
    def __init__(self, url: str, service_key: str) -> None:
        self.client = create_client(url, service_key)

    def upsert_document(
        self,
        title: str,
        source_file: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = {
            "title": title,
            "source_file": source_file,
            "metadata": metadata or {},
        }

        try:
            result = self.client.table("documents").upsert(
                payload,
                on_conflict="source_file",
            ).execute()
        except Exception as exc:
            logger.exception("Supabase upsert failed for document: %s", source_file)
            raise RuntimeError(f"Supabase upsert failed for {source_file}: {exc}") from exc

        if result.data:
            return result.data[0]["id"]

        try:
            lookup = (
                self.client.table("documents")
                .select("id")
                .eq("source_file", source_file)
                .limit(1)
                .execute()
            )
        except Exception as exc:
            logger.exception("Supabase lookup failed for document: %s", source_file)
            raise RuntimeError(f"Supabase lookup failed for {source_file}: {exc}") from exc

        if not lookup.data:
            raise RuntimeError(f"Failed to fetch document id for {source_file}")

        return lookup.data[0]["id"]

    def insert_chunks(self, rows: List[Dict[str, Any]], batch_size: int = 100) -> None:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            try:
                result = self.client.table("chunks").insert(batch).execute()
            except Exception as exc:
                logger.exception("Supabase chunk insert failed at batch starting index %d", i)
                raise RuntimeError(f"Chunk insert failed at batch starting index {i}: {exc}") from exc

            if result.data is None:
                raise RuntimeError(f"Insert failed at batch starting index {i}")

            logger.info("Inserted %d chunk rows", len(batch))

    def match_chunks(self, query_embedding: List[float], match_count: int = 5):
        try:
            result = self.client.rpc(
                "match_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_count": match_count,
                },
            ).execute()
        except Exception as exc:
            logger.exception("Supabase RPC match_chunks failed")
            raise RuntimeError(f"match_chunks RPC failed: {exc}") from exc

        return result.data or []