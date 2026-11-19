"""The interface every vector database adapter implements.

Code elsewhere only talks to `VectorStore`, so Qdrant <-> Chroma is a config switch
(VECTOR_STORE=chroma) and evaluation can compare them on equal terms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from legal_contract_clause_search.model import Chunk, SearchFilters
from legal_contract_clause_search.sparse import SparseVec


@dataclass
class StoredHit:
    score: float
    payload: dict


class CollectionMissingError(RuntimeError):
    pass


def chunk_payload(chunk: Chunk) -> dict:
    """Metadata stored next to each vector (Qdrant calls this the 'payload')."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "contract_type": chunk.contract_type,
        "sign_date": chunk.sign_date.isoformat() if chunk.sign_date else None,   # ISO for range filters
        "section_path": chunk.section_path,
        "text": chunk.text,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "chunker": chunk.chunker,
    }


class VectorStore(ABC):
    name: str
    supports_sparse: bool = False

    @abstractmethod
    def create(self, collection: str, dim: int, recreate: bool = False) -> None: ...

    @abstractmethod
    def exists(self, collection: str) -> bool: ...

    @abstractmethod
    def upsert(self, collection: str, chunks: list[Chunk], dense: np.ndarray,
               sparse: list[SparseVec] | None = None) -> None: ...

    @abstractmethod
    def search(self, collection: str, filters: SearchFilters, limit: int, mode: str,
               dense: np.ndarray | None = None, sparse: SparseVec | None = None) -> list[StoredHit]: ...

    @abstractmethod
    def count(self, collection: str) -> int: ...

    def close(self) -> None:
        pass
