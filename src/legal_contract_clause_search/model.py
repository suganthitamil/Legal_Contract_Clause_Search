"""Data shapes shared by every part of the system (pydantic models)."""

import uuid
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# Fixed namespace -> the same chunk_id always gives the same Qdrant point id,
# so re-running ingest overwrites points instead of duplicating them.
_POINT_NAMESPACE = uuid.UUID("6f1c2a52-6c1f-4f4e-9a52-6c6c1d1e7a01")


class Contract(BaseModel):
    doc_id: str
    title: str
    contract_type: str
    sign_date: date | None
    parties: list[str]
    text: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str                     # EXACT slice: contract.text[char_start:char_end]
    char_start: int
    char_end: int
    chunk_index: int
    section_path: str | None
    contract_type: str
    sign_date: date | None
    title: str = ""
    chunker: str = ""
    token_count: int = 0
    embed_text: str = ""          # what we actually embed (text + optional context header)

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(_POINT_NAMESPACE, self.chunk_id))


SearchMode = Literal["dense", "sparse", "hybrid"]


class SearchFilters(BaseModel):
    """Metadata filters. Every field is optional: None means 'don't filter on this'."""

    contract_types: list[str] | None = Field(None, examples=[["License"]])
    signed_after: date | None = Field(None, description="Inclusive. Contracts without a sign date are excluded when set.")
    signed_before: date | None = Field(None, description="Inclusive.")
    doc_ids: list[str] | None = Field(None, description="Restrict to specific contracts.")


class SearchRequest(SearchFilters):
    query: str = Field(min_length=1, examples=["termination for convenience"])
    top_k: int = Field(5, ge=1, le=50)
    mode: SearchMode | None = Field(None, description="dense | sparse | hybrid (default from settings)")
    rerank: bool | None = Field(None, description="Rerank candidates with a cross-encoder (default from settings)")
    max_per_contract: int | None = Field(None, ge=1, description="At most N chunks per contract")


class SearchHit(BaseModel):
    score: float
    chunk_id: str
    doc_id: str
    title: str
    contract_type: str
    sign_date: date | None
    section_path: str | None
    text: str
    char_start: int
    char_end: int
    retrieval_score: float | None = None   # score before reranking (if reranked)


class Citation(BaseModel):
    hit_index: int                # 1-based position in `hits`
    cited_text: str
    doc_id: str
    title: str
    section_path: str | None


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    hits: list[SearchHit]
    model: str

