import os
from pathlib import Path

from mem0 import Memory

BASE_DIR = Path(__file__).resolve().parents[2]
LOCAL_STORE_DIR = BASE_DIR / "locomo"
QDRANT_PATH = LOCAL_STORE_DIR / "qdrant"
META_DIR = LOCAL_STORE_DIR / "meta"

# Ensure directories exist for on-disk storage and metadata
for path in [LOCAL_STORE_DIR, QDRANT_PATH, META_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Keep mem0 metadata/history alongside the vector store
os.environ.setdefault("MEM0_DIR", str(META_DIR))


def build_local_memory():
    """Create a local Mem0 instance backed by on-disk Qdrant + OpenAI models."""
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dims = int(os.getenv("EMBEDDING_DIM", "1536"))
    llm_model = os.getenv("MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    config = {
        "version": "v1.1",
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": str(QDRANT_PATH),
                "collection_name": "mem0",
                "on_disk": True,
                "embedding_model_dims": embedding_dims,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "api_key": api_key,
                "openai_base_url": base_url,
                "model": embedding_model,
                "embedding_dims": embedding_dims,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": api_key,
                "openai_base_url": base_url,
                "model": llm_model,
                "temperature": 0.0,
            },
        },
        "history_db_path": str(LOCAL_STORE_DIR / "history.db"),
    }

    return Memory.from_config(config)
