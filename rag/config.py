from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load the .env file from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_service_key: str
    embedding_model_id: str
    chunk_size: int
    chunk_overlap: int
    insert_batch_size: int

    @staticmethod
    def from_env() -> "Settings":
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

        if not url or not key:
            raise RuntimeError(
                "Missing required environment variables: SUPABASE_URL, SUPABASE_SERVICE_KEY. "
                "Set them in your environment or .env."
            )

        return Settings(
            supabase_url=url,
            supabase_service_key=key,
            embedding_model_id=os.getenv(
                "EMBEDDING_MODEL_ID",
                "nomic-ai/nomic-embed-text-v1.5",
            ),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            insert_batch_size=int(os.getenv("INSERT_BATCH_SIZE", "100")),
        )