"""Search:  query -> (dense and/or BM25 query vectors) -> filtered vector search -> [rerank] -> hits.

    retriever = Retriever(settings)
    hits = retriever.search(SearchRequest(query="governing law", contract_types=["License"]))

Search modes:
  dense  : embeddings only (meaning)
  sparse : BM25 only (exact words)
  hybrid : both, fused with Reciprocal Rank Fusion   <- default
Optional extras:
  rerank           : cross-encoder re-scores the top `rerank_candidates`
  max_per_contract : diversity — at most N chunks from the same contract
"""

import logging

from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.embeddings import get_embedder
from legal_contract_clause_search.index_meta import read_meta
from legal_contract_clause_search.ingest import collection_name, meta_key
from legal_contract_clause_search.model import SearchHit, SearchRequest
from legal_contract_clause_search.rerank import get_reranker
from legal_contract_clause_search.sparse import BM25Encoder
from legal_contract_clause_search.vectorstore import StoredHit, VectorStore, get_store

log = logging.getLogger(__name__)


def to_hit(h: StoredHit) -> SearchHit:
    p = h.payload
    return SearchHit(score=h.score, chunk_id=p["chunk_id"], doc_id=p["doc_id"], title=p.get("title", ""),
                     contract_type=p["contract_type"], sign_date=p.get("sign_date"),
                     section_path=p.get("section_path"), text=p["text"],
                     char_start=p["char_start"], char_end=p["char_end"])


def limit_per_contract(hits: list[SearchHit], n: int) -> list[SearchHit]:
    seen: dict[str, int] = {}
    out = []
    for h in hits:
        if seen.get(h.doc_id, 0) < n:
            out.append(h)
            seen[h.doc_id] = seen.get(h.doc_id, 0) + 1
    return out


class Retriever:
    def __init__(self, settings: Settings, chunker: str | None = None, embedder: str | None = None,
                 store: VectorStore | None = None, collection_prefix: str | None = None):
        self.settings = settings
        self.store = store or get_store(settings)
        self.embedder = get_embedder(embedder or settings.embedder, settings)
        self.collection = collection_name(settings, chunker or settings.chunker, self.embedder.name,
                                          collection_prefix)
        meta = read_meta(settings.index_meta_path, meta_key(self.store.name, self.collection)) or {}
        avg_len = meta.get("bm25_avg_len")
        self.bm25 = BM25Encoder(avg_len) if avg_len else None
        self._reranker_checked = False
        self._reranker = None

    @property
    def reranker(self):
        if not self._reranker_checked:
            self._reranker = get_reranker(self.settings)
            self._reranker_checked = True
        return self._reranker

    def resolve_mode(self, requested: str | None) -> str:
        mode = requested or self.settings.search_mode
        if mode in ("sparse", "hybrid") and (not self.store.supports_sparse or self.bm25 is None):
            log.warning("%s search unavailable for this index (%s) -> using dense", mode, self.store.name)
            return "dense"
        return mode

    def search(self, req: SearchRequest) -> list[SearchHit]:
        mode = self.resolve_mode(req.mode)
        use_rerank = (self.settings.rerank if req.rerank is None else req.rerank) and self.reranker is not None

        n = req.top_k
        if use_rerank:
            n = max(n, self.settings.rerank_candidates)
        if req.max_per_contract:
            n = min(n * 4, 200)        # over-fetch, then thin out per contract

        dense = self.embedder.embed_query(req.query) if mode in ("dense", "hybrid") else None
        sparse = self.bm25.encode_query(req.query) if mode in ("sparse", "hybrid") else None
        hits = [to_hit(h) for h in self.store.search(self.collection, req, n, mode, dense, sparse)]

        if use_rerank:
            hits = self.reranker.rerank(req.query, hits)
        if req.max_per_contract:
            hits = limit_per_contract(hits, req.max_per_contract)
        return hits[:req.top_k]

    def count(self) -> int:
        return self.store.count(self.collection)

    def close(self) -> None:
        self.store.close()
