"""Filters are the brief's hard requirement, so test them against a real (in-memory) Qdrant."""

from datetime import date

import numpy as np
import pytest
from qdrant_client import QdrantClient

from legal_contract_clause_search.model import Chunk, SearchFilters
from legal_contract_clause_search.sparse import BM25Encoder
from legal_contract_clause_search.vectorstore.chroma_store import build_where
from legal_contract_clause_search.vectorstore.qdrant_store import QdrantStore

ROWS = [  # doc_id, type, sign_date, text
    ("a", "License", date(2016, 5, 1), "licensee shall pay royalties"),
    ("b", "License", date(2010, 1, 1), "licensee may terminate for convenience"),
    ("c", "Distributor", date(2018, 3, 1), "distributor territory is exclusive"),
    ("d", "License", None, "governing law is new york"),
]


@pytest.fixture
def store():
    s = QdrantStore(client=QdrantClient(":memory:"))
    s.create("t", dim=4)
    chunks = [Chunk(chunk_id=f"{d}::x::0", doc_id=d, text=t, char_start=0, char_end=len(t), chunk_index=0,
                    section_path=None, contract_type=ty, sign_date=sd) for d, ty, sd, t in ROWS]
    rng = np.random.default_rng(0)
    dense = rng.normal(size=(len(chunks), 4)).astype(np.float32)
    bm25 = BM25Encoder.fit([c.text for c in chunks])
    s.upsert("t", chunks, dense, [bm25.encode_document(c.text) for c in chunks])
    s.bm25 = bm25
    return s


def _ids(hits):
    return sorted(h.payload["doc_id"] for h in hits)


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_type_and_date_filter(store, mode):
    f = SearchFilters(contract_types=["License"], signed_after=date(2015, 1, 1))
    hits = store.search("t", f, 10, mode, dense=np.ones(4), sparse=store.bm25.encode_query("licensee"))
    assert _ids(hits) == ["a"]          # b too old, c wrong type, d has no date


def test_no_filter_returns_all(store):
    assert _ids(store.search("t", SearchFilters(), 10, "dense", dense=np.ones(4))) == ["a", "b", "c", "d"]


def test_doc_filter_and_before(store):
    f = SearchFilters(signed_before=date(2012, 12, 31))
    assert _ids(store.search("t", f, 10, "dense", dense=np.ones(4))) == ["b"]
    f = SearchFilters(doc_ids=["c", "d"])
    assert _ids(store.search("t", f, 10, "dense", dense=np.ones(4))) == ["c", "d"]


def test_sparse_matches_keywords(store):
    hits = store.search("t", SearchFilters(), 1, "sparse", sparse=store.bm25.encode_query("exclusive territory"))
    assert hits[0].payload["doc_id"] == "c"


def test_chroma_where_clause():
    assert build_where(SearchFilters()) is None
    assert build_where(SearchFilters(contract_types=["License"])) == {"contract_type": {"$in": ["License"]}}
    w = build_where(SearchFilters(contract_types=["License"], signed_after=date(2015, 1, 2)))
    assert w == {"$and": [{"contract_type": {"$in": ["License"]}}, {"sign_date_int": {"$gte": 20150102}}]}
