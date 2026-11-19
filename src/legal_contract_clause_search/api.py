"""REST API (FastAPI).  Start:  uv run lccs serve   ->  http://localhost:8000/docs

Endpoints
  GET  /health           index status + which optional features are available
  GET  /contract-types   values allowed in the `contract_types` filter
  POST /search           ranked clauses (retrieval only)
  POST /answer           Claude answer with citations, built from the retrieved clauses
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.generation import GenerationError, answer
from legal_contract_clause_search.loaders.cuad import CONTRACT_TYPE_RULES
from legal_contract_clause_search.model import (
    AnswerResponse,
    SearchHit,
    SearchRequest,
)
from legal_contract_clause_search.model_cache import ModelNotAvailableError
from legal_contract_clause_search.retrieval import Retriever
from legal_contract_clause_search.vectorstore import CollectionMissingError

log = logging.getLogger(__name__)
settings = Settings()
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One Retriever for the whole app: loads the model once and holds the
    # Qdrant local-mode lock while the server runs.
    state["retriever"] = Retriever(settings)
    yield
    state["retriever"].close()


app = FastAPI(title="Legal Contract Clause Search", version="1.0", lifespan=lifespan,
              description="Clause-aware semantic + keyword search over CUAD contracts, "
                          "with contract-type and sign-date filters.")


def _retriever() -> Retriever:
    return state["retriever"]


@app.get("/health")
def health() -> dict:
    r = _retriever()
    return {
        "status": "ok",
        "collection": r.collection,
        "points": r.count(),
        "embedder": r.embedder.name,
        "default_search_mode": r.resolve_mode(None),
        "rerank_available": r.reranker is not None,
        "generation_model": settings.claude_model,
    }


@app.get("/contract-types")
def contract_types() -> list[str]:
    return sorted({label for _, label in CONTRACT_TYPE_RULES} | {"Other"})


@app.post("/search")
def search(req: SearchRequest) -> list[SearchHit]:
    try:
        return _retriever().search(req)
    except (CollectionMissingError, ModelNotAvailableError) as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/answer")
def answer_question(req: SearchRequest) -> AnswerResponse:
    hits = search(req)          # top_k clauses become the documents Claude answers from
    try:
        return answer(req.query, hits, settings)
    except GenerationError as e:
        raise HTTPException(status_code=503, detail=str(e))

