import sys
import os

# Ensure project root is on Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rag.config import Settings
from rag.embeddings import Embedder
from rag.supabase_store import SupabaseStore
from rag.retrieve import to_retrieved_chunks


def main() -> None:
    settings = Settings.from_env()

    print("Loading embedder...")
    embedder = Embedder(model_id=settings.embedding_model_id)

    print("Connecting to Supabase...")
    store = SupabaseStore(settings.supabase_url, settings.supabase_service_key)

    query = "What does the ISM say about cyber security roles and responsibilities?"
    print(f"\nQuery: {query}")

    print("\nGenerating query embedding...")
    query_embedding = embedder.embed_query(query)

    print("Calling match_chunks RPC...")
    rows = store.match_chunks(query_embedding=query_embedding, match_count=5)

    chunks = to_retrieved_chunks(rows)

    if not chunks:
        print("\nNo retrieval results found.")
        return

    print("\nTop Retrieved Chunks:\n")
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        print(f"[{i}] similarity={chunk.similarity:.4f}")
        print(
            f"source_file={meta.get('source_file')} | "
            f"pages={meta.get('page_start')}-{meta.get('page_end')}"
        )
        print(chunk.content[:500].replace("\n", " "))
        print("-" * 100)


if __name__ == "__main__":
    main()