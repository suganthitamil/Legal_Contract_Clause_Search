"""Qdrant adapter — the primary vector database.

Each point = one chunk, holding TWO named vectors plus a payload:
  "dense"  : 384/768/1024-dim embedding        -> semantic search (cosine)
  "sparse" : BM25 word weights                  -> keyword search (Qdrant applies IDF)
  payload  : doc_id, contract_type, sign_date, section_path, text, char offsets ...

FILTERING: payload indexes on contract_type (keyword) and sign_date (datetime)
let Qdrant apply filters DURING the vector search (not after), so a filtered
top-5 is always 5 matching results when 5 exist.

INDEXING: Qdrant uses HNSW (a layered graph of nearest neighbours) for fast
approximate search. In embedded "local mode" (no server) it searches exactly
(brute force) — 100% accurate and fast enough for ~20k chunks. Point
QDRANT_URL at a Qdrant server to get real HNSW (tunable m / ef_construct).

HYBRID: run a dense search and a sparse search (Prefetch), then fuse the two
rankings with Reciprocal Rank Fusion: score = sum 1/(k + rank) over both lists.
RRF needs no score calibration — it only uses positions.
"""

import warnings

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.model import Chunk, SearchFilters
from legal_contract_clause_search.sparse import SparseVec
from legal_contract_clause_search.vectorstore.base import (
    CollectionMissingError,
    StoredHit,
    VectorStore,
    chunk_payload,
)

DENSE, SPARSE = "dense", "sparse"


def build_filter(f: SearchFilters) -> qm.Filter | None:
    must: list[qm.Condition] = []
    if f.contract_types:
        must.append(qm.FieldCondition(key="contract_type", match=qm.MatchAny(any=f.contract_types)))
    if f.signed_after or f.signed_before:
        must.append(qm.FieldCondition(key="sign_date", range=qm.DatetimeRange(
            gte=f.signed_after.isoformat() if f.signed_after else None,
            lte=f.signed_before.isoformat() if f.signed_before else None)))
    if f.doc_ids:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=f.doc_ids)))
    return qm.Filter(must=must) if must else None


class QdrantStore(VectorStore):
    name = "qdrant"
    supports_sparse = True

    def __init__(self, settings: Settings | None = None, client: QdrantClient | None = None):
        if client is not None:
            self.client = client                       # tests pass QdrantClient(":memory:")
        elif settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            # Local mode locks the folder: only ONE process may open it at a time
            # (stop `lccs serve` before running `lccs ingest`, or use a Qdrant server).
            self.client = QdrantClient(path=str(settings.qdrant_path))

    def exists(self, collection: str) -> bool:
        return self.client.collection_exists(collection)

    def create(self, collection: str, dim: int, recreate: bool = False) -> None:
        if self.exists(collection):
            if not recreate:
                return
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config={DENSE: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),   # used in server mode
        )
        # Payload indexes make filtered search fast on a Qdrant SERVER. Local mode filters
        # without them (and warns), so we silence that warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.client.create_payload_index(collection, "contract_type", qm.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(collection, "sign_date", qm.PayloadSchemaType.DATETIME)
            self.client.create_payload_index(collection, "doc_id", qm.PayloadSchemaType.KEYWORD)

    def upsert(self, collection: str, chunks: list[Chunk], dense, sparse: list[SparseVec] | None = None,
               batch_size: int = 256) -> None:
        for i in range(0, len(chunks), batch_size):
            points = []
            for j in range(i, min(i + batch_size, len(chunks))):
                vector: dict = {DENSE: dense[j].tolist()}
                if sparse is not None:
                    vector[SPARSE] = qm.SparseVector(indices=sparse[j].indices, values=sparse[j].values)
                points.append(qm.PointStruct(id=chunks[j].point_id, vector=vector,
                                             payload=chunk_payload(chunks[j])))
            self.client.upsert(collection, points=points, wait=True)

    def search(self, collection: str, filters: SearchFilters, limit: int, mode: str,
               dense=None, sparse: SparseVec | None = None) -> list[StoredHit]:
        if not self.exists(collection):
            raise CollectionMissingError(f"Collection {collection!r} not found. Run `uv run lccs ingest` first.")
        flt = build_filter(filters)
        sparse_q = qm.SparseVector(indices=sparse.indices, values=sparse.values) if sparse else None

        if mode == "dense":
            res = self.client.query_points(collection, query=dense.tolist(), using=DENSE,
                                           query_filter=flt, limit=limit, with_payload=True)
        elif mode == "sparse":
            if not sparse_q or not sparse_q.indices:
                return []
            res = self.client.query_points(collection, query=sparse_q, using=SPARSE,
                                           query_filter=flt, limit=limit, with_payload=True)
        elif mode == "hybrid":
            depth = max(limit * 4, 40)                 # each side retrieves deeper before fusing
            prefetch = [qm.Prefetch(query=dense.tolist(), using=DENSE, filter=flt, limit=depth)]
            if sparse_q and sparse_q.indices:
                prefetch.append(qm.Prefetch(query=sparse_q, using=SPARSE, filter=flt, limit=depth))
            res = self.client.query_points(collection, prefetch=prefetch,
                                           query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                                           query_filter=flt, limit=limit, with_payload=True)
        else:
            raise ValueError(f"Unknown search mode {mode!r}")
        return [StoredHit(score=p.score, payload=p.payload or {}) for p in res.points]

    def count(self, collection: str) -> int:
        return self.client.count(collection).count if self.exists(collection) else 0

    def close(self) -> None:
        self.client.close()
