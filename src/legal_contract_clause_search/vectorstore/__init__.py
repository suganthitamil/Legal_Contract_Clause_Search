"""Vector database adapters. Use `get_store(settings)`."""

from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.vectorstore.base import CollectionMissingError, StoredHit, VectorStore


def get_store(settings: Settings, name: str | None = None) -> VectorStore:
    name = name or settings.vector_store
    if name == "qdrant":
        from legal_contract_clause_search.vectorstore.qdrant_store import QdrantStore
        return QdrantStore(settings)
    if name == "chroma":
        from legal_contract_clause_search.vectorstore.chroma_store import ChromaStore
        return ChromaStore(settings)
    raise ValueError(f"Unknown vector store {name!r} (qdrant | chroma)")


__all__ = ["CollectionMissingError", "StoredHit", "VectorStore", "get_store"]
