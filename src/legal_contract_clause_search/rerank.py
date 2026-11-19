"""Reranking with a cross-encoder (retrieve-then-rerank).

Embedding search encodes the query and each chunk SEPARATELY, then compares
vectors: fast, but the model never sees them together. A cross-encoder reads
(query, chunk) TOGETHER and outputs a relevance score: slower, more precise.
So: retrieve ~20 candidates fast, rerank them, keep the best top_k.
"""

import logging
from functools import lru_cache

from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.model import SearchHit
from legal_contract_clause_search.model_cache import is_cached

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str, cache_dir: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self.model = TextCrossEncoder(model_name, cache_dir=cache_dir)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return hits
        scores = list(self.model.rerank(query, [h.text for h in hits]))
        reranked = [h.model_copy(update={"retrieval_score": h.score, "score": float(s)})
                    for h, s in zip(hits, scores)]
        return sorted(reranked, key=lambda h: h.score, reverse=True)


@lru_cache
def _load(model_name: str, models_dir: str) -> Reranker | None:
    from pathlib import Path

    if not is_cached(model_name, "rerank", Path(models_dir)):
        log.warning("Reranker %s not downloaded -> reranking skipped. Run `lccs download-models` "
                    "on a network that allows huggingface.co.", model_name)
        return None
    return Reranker(model_name, models_dir)


def get_reranker(settings: Settings) -> Reranker | None:
    return _load(settings.reranker_model_name, str(settings.models_dir))
