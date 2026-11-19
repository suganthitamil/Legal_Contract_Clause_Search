"""Ingestion pipeline:  load -> chunk -> embed (dense + BM25) -> store.

    uv run lccs ingest                        # defaults from settings (clause, bge-small, qdrant)
    uv run lccs ingest --chunker fixed --limit 100

Each (chunker, embedder) pair gets its OWN collection, e.g.
"contract_clauses__clause__bge-small", so several indexes can exist side by side
and evaluation can compare them.
"""

import logging
import random
import time
from dataclasses import asdict, dataclass

from legal_contract_clause_search.chunking import get_chunker
from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.embeddings import get_embedder
from legal_contract_clause_search.index_meta import write_meta
from legal_contract_clause_search.loaders.cuad import load_cuad
from legal_contract_clause_search.model import Chunk, Contract
from legal_contract_clause_search.sparse import BM25Encoder
from legal_contract_clause_search.vectorstore import VectorStore, get_store

log = logging.getLogger(__name__)


def collection_name(settings: Settings, chunker: str, embedder: str, prefix: str | None = None) -> str:
    base = prefix or settings.collection_name
    suffix = "" if settings.contextual_headers else "-nohdr"
    return f"{base}__{chunker}{suffix}__{embedder}"


def meta_key(store: str, collection: str) -> str:
    return f"{store}:{collection}"


def select_contracts(contracts: list[Contract], limit: int | None, seed: int = 42) -> list[Contract]:
    """A fixed random sample (same seed -> same contracts), so evaluation is repeatable."""
    if not limit or limit >= len(contracts):
        return contracts
    return sorted(random.Random(seed).sample(contracts, limit), key=lambda c: c.doc_id)


@dataclass
class IngestStats:
    collection: str
    contracts: int
    chunks: int
    avg_tokens: float
    seconds: float


def ingest(settings: Settings, chunker_name: str | None = None, embedder_name: str | None = None,
           store: VectorStore | None = None, limit: int | None = None, recreate: bool = True,
           collection_prefix: str | None = None) -> tuple[IngestStats, list[Chunk]]:
    t0 = time.time()
    chunker_name = chunker_name or settings.chunker
    embedder_name = embedder_name or settings.embedder
    own_store = store is None
    store = store or get_store(settings)
    collection = collection_name(settings, chunker_name, embedder_name, collection_prefix)

    contracts = select_contracts(load_cuad(settings), limit)
    embedder = get_embedder(embedder_name, settings)
    chunker = get_chunker(chunker_name, settings, embedder=embedder)
    log.info("Chunking %d contracts with %r ...", len(contracts), chunker_name)
    chunks: list[Chunk] = []
    for i, contract in enumerate(contracts, 1):
        chunks.extend(chunker.chunk(contract))
        if i % 100 == 0:
            log.info("  chunked %d/%d contracts (%d chunks)", i, len(contracts), len(chunks))
    texts = [c.embed_text for c in chunks]

    log.info("Embedding %d chunks with %r (dim %d) ...", len(chunks), embedder_name, embedder.dim)
    vectors = []
    step = 512
    for i in range(0, len(texts), step):
        vectors.append(embedder.embed_documents(texts[i:i + step]))
        done = min(i + step, len(texts))
        rate = done / max(time.time() - t0, 1e-6)
        log.info("  embedded %d/%d (~%.0f chunks/s overall)", done, len(texts), rate)
    import numpy as np
    dense = np.vstack(vectors) if vectors else np.zeros((0, embedder.dim), np.float32)

    bm25, sparse = None, None
    if store.supports_sparse:
        bm25 = BM25Encoder.fit(texts)
        sparse = [bm25.encode_document(t) for t in texts]

    log.info("Writing to %s collection %r ...", store.name, collection)
    store.create(collection, embedder.dim, recreate=recreate)
    store.upsert(collection, chunks, dense, sparse)

    stats = IngestStats(collection, len(contracts), len(chunks),
                        round(sum(c.token_count for c in chunks) / max(len(chunks), 1), 1),
                        round(time.time() - t0, 1))
    write_meta(settings.index_meta_path, meta_key(store.name, collection), {
        **asdict(stats), "store": store.name, "chunker": chunker_name, "embedder": embedder_name,
        "dim": embedder.dim, "bm25_avg_len": bm25.avg_len if bm25 else None,
        "chunk_max_tokens": settings.chunk_max_tokens, "contextual_headers": settings.contextual_headers,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    log.info("Done: %s", stats)
    if own_store:
        store.close()
    return stats, chunks
