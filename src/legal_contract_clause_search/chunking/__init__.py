"""Chunking strategies. Use `get_chunker(name, settings)` to build one by name."""

from legal_contract_clause_search.chunking.base import Chunker, Span
from legal_contract_clause_search.chunking.clause import ClauseChunker
from legal_contract_clause_search.chunking.fixed import FixedChunker, SlidingWindowChunker
from legal_contract_clause_search.chunking.recursive import RecursiveChunker
from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.tokens import get_token_counter

CHUNKERS = ["clause", "fixed", "sliding", "recursive", "semantic"]


def get_chunker(name: str, settings: Settings, embedder=None) -> Chunker:
    counter = get_token_counter(settings.models_dir)
    if name == "clause":
        return ClauseChunker(settings, counter)
    if name == "fixed":
        return FixedChunker(settings, counter)
    if name == "sliding":
        return SlidingWindowChunker(settings, counter)
    if name == "recursive":
        return RecursiveChunker(settings, counter)
    if name == "semantic":
        from legal_contract_clause_search.chunking.semantic import SemanticChunker
        from legal_contract_clause_search.embeddings import get_embedder
        return SemanticChunker(settings, counter, embedder or get_embedder(settings.embedder, settings))
    raise ValueError(f"Unknown chunker {name!r}. Choose from {CHUNKERS}")


__all__ = ["CHUNKERS", "Chunker", "Span", "get_chunker"]
