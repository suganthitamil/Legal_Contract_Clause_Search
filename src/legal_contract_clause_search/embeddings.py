"""Embedding models: text -> vector.

All embedders return L2-NORMALIZED float32 vectors (length 1), so cosine
similarity == dot product (notebook 01, lesson 5).

Two methods, because retrieval models treat them differently:
  embed_documents(texts)  -> for chunks being stored
  embed_query(text)       -> for the user's search text

Available:
  bge-small     BAAI/bge-small-en-v1.5   384 dims  local (fastembed)   <- default
  bge-base      BAAI/bge-base-en-v1.5    768 dims  local (fastembed)   needs download
  voyage-law-2  Voyage AI legal model   1024 dims  API, needs VOYAGE_API_KEY
"""

import time
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx
import numpy as np

from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.model_cache import require_cached

EMBEDDERS: dict[str, tuple[str, str, int]] = {
    "bge-small": ("fastembed", "BAAI/bge-small-en-v1.5", 384),
    "bge-base": ("fastembed", "BAAI/bge-base-en-v1.5", 768),
    "voyage-law-2": ("voyage", "voyage-law-2", 1024),
}


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder(Embedder):
    def __init__(self, name: str, model_id: str, dim: int, settings: Settings):
        from fastembed import TextEmbedding

        require_cached(model_id, "dense", settings.models_dir)
        self.name, self.dim = name, dim
        self.batch_size = settings.embed_batch_size
        self.model = TextEmbedding(model_id, cache_dir=str(settings.models_dir))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.array(list(self.model.embed(texts, batch_size=self.batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self.model.query_embed([text])))[0])


class VoyageEmbedder(Embedder):
    """Voyage AI over plain HTTP (Anthropic's recommended embedding partner)."""

    URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, name: str, model_id: str, dim: int, settings: Settings):
        if not settings.voyage_api_key:
            raise RuntimeError("voyage-law-2 needs VOYAGE_API_KEY in .env")
        self.name, self.dim, self.model_id = name, dim, model_id
        self.client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {settings.voyage_api_key}"})

    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            for attempt in range(6):
                r = self.client.post(self.URL, json={"input": batch, "model": self.model_id,
                                                     "input_type": input_type, "truncation": True})
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2 ** attempt)        # rate limited / server error -> back off
                    continue
                r.raise_for_status()
                out.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
                break
            else:
                raise RuntimeError(f"Voyage API kept failing ({r.status_code}): {r.text[:200]}")
        return _normalize(np.array(out))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "document") if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "query")[0]


@lru_cache
def _build(name: str, settings_json: str) -> Embedder:
    settings = Settings.model_validate_json(settings_json)
    if name not in EMBEDDERS:
        raise ValueError(f"Unknown embedder {name!r}. Choose from {list(EMBEDDERS)}")
    kind, model_id, dim = EMBEDDERS[name]
    cls = FastEmbedEmbedder if kind == "fastembed" else VoyageEmbedder
    return cls(name, model_id, dim, settings)


def get_embedder(name: str, settings: Settings) -> Embedder:
    """Cached: loading a model takes a second or two, so build each one once per process."""
    return _build(name, settings.model_dump_json())
