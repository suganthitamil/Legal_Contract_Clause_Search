"""Chroma adapter — for comparison with Qdrant.

Differences you'll notice (and why Qdrant is the primary store here):
  - Dense vectors only: no sparse/BM25 vectors -> no hybrid search.
  - Metadata values must be str/int/float/bool (no None, no dates), so the sign
    date is stored as an integer YYYYMMDD to allow range filters ($gte / $lte).
  - Filters use a MongoDB-like syntax: {"$and": [{"contract_type": {"$in": [...]}}, ...]}
  - Returns cosine DISTANCE (0 = identical); we convert to similarity = 1 - distance.
Chroma also uses HNSW internally (via hnswlib).
"""

import chromadb
from chromadb.config import Settings as ChromaSettings

from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.model import Chunk, SearchFilters
from legal_contract_clause_search.vectorstore.base import (
    CollectionMissingError,
    StoredHit,
    VectorStore,
    chunk_payload,
)


def _date_int(iso: str | None) -> int | None:
    return int(iso.replace("-", "")) if iso else None


def build_where(f: SearchFilters) -> dict | None:
    conds: list[dict] = []
    if f.contract_types:
        conds.append({"contract_type": {"$in": f.contract_types}})
    if f.signed_after:
        conds.append({"sign_date_int": {"$gte": _date_int(f.signed_after.isoformat())}})
    if f.signed_before:
        conds.append({"sign_date_int": {"$lte": _date_int(f.signed_before.isoformat())}})
    if f.doc_ids:
        conds.append({"doc_id": {"$in": f.doc_ids}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


class ChromaStore(VectorStore):
    name = "chroma"
    supports_sparse = False

    def __init__(self, settings: Settings):
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path),
                                                settings=ChromaSettings(anonymized_telemetry=False))

    def exists(self, collection: str) -> bool:
        return collection in [c if isinstance(c, str) else c.name for c in self.client.list_collections()]

    def create(self, collection: str, dim: int, recreate: bool = False) -> None:
        if self.exists(collection):
            if not recreate:
                return
            self.client.delete_collection(collection)
        self.client.create_collection(collection, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def _col(self, collection: str):
        if not self.exists(collection):
            raise CollectionMissingError(f"Collection {collection!r} not found. Run `uv run lccs ingest --store chroma` first.")
        return self.client.get_collection(collection, embedding_function=None)

    def upsert(self, collection: str, chunks: list[Chunk], dense, sparse=None, batch_size: int = 1000) -> None:
        col = self._col(collection)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            metas = []
            for c in batch:
                p = chunk_payload(c)
                p.pop("text")                                   # stored as the document instead
                p["sign_date_int"] = _date_int(p["sign_date"])
                metas.append({k: v for k, v in p.items() if v is not None})   # Chroma rejects None
            col.upsert(ids=[c.point_id for c in batch], embeddings=dense[i:i + batch_size].tolist(),
                       documents=[c.text for c in batch], metadatas=metas)

    def search(self, collection: str, filters: SearchFilters, limit: int, mode: str,
               dense=None, sparse=None) -> list[StoredHit]:
        res = self._col(collection).query(query_embeddings=[dense.tolist()], n_results=limit,
                                          where=build_where(filters),
                                          include=["metadatas", "documents", "distances"])
        hits = []
        for meta, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
            payload = dict(meta)
            payload["text"] = doc
            payload.setdefault("sign_date", None)
            payload.setdefault("section_path", None)
            hits.append(StoredHit(score=1.0 - float(dist), payload=payload))
        return hits

    def count(self, collection: str) -> int:
        return self._col(collection).count() if self.exists(collection) else 0
