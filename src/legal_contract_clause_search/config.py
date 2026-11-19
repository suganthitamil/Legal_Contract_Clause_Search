"""All settings in one place.

Every value has a sensible default and can be overridden in a `.env` file at the
project root (see `.env.example`), e.g.  SEARCH_MODE=dense  or  CHUNKER=fixed.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- data & folders ------------------------------------------------------
    cuad_dir: Path = PROJECT_ROOT / "data" / "raw" / "cuad" / "CUAD_v1"
    data_dir: Path = PROJECT_ROOT / "data"
    # Downloaded models live inside the project, NOT Windows Temp (which gets cleaned).
    models_dir: Path = PROJECT_ROOT / "models"
    results_dir: Path = PROJECT_ROOT / "results"

    # --- vector store ----------------------------------------------------------
    vector_store: str = "qdrant"                 # qdrant | chroma
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant"   # local (embedded) mode
    qdrant_url: str | None = None                # set e.g. http://localhost:6333 for server mode
    qdrant_api_key: str | None = None
    chroma_path: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "contract_clauses"    # base name; chunker + embedder are appended

    # --- embeddings ------------------------------------------------------------
    embedder: str = "bge-small"                  # bge-small | bge-base | voyage-law-2
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"   # used by notebooks/01
    embed_batch_size: int = 32
    voyage_api_key: str | None = None

    # --- chunking --------------------------------------------------------------
    chunker: str = "clause"                      # clause | fixed | sliding | recursive | semantic
    chunk_max_tokens: int = 384                  # bge models read at most 512 tokens
    chunk_overlap_tokens: int = 64               # used by the sliding-window chunker
    chunk_min_tokens: int = 15                   # heading-only fragments below this are merged forward
    # Prefix "type | section path" to each chunk before embedding. Measured to HURT
    # retrieval with bge-small (hit@1 0.359 vs 0.438 without) -> off. See DECISIONS.md D5.
    contextual_headers: bool = False

    # --- retrieval -------------------------------------------------------------
    search_mode: str = "hybrid"                  # dense | sparse | hybrid
    rerank: bool = True                          # used only if the reranker model is downloaded
    rerank_candidates: int = 20                  # retrieve this many, rerank, keep top_k
    reranker_model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- generation (Phase 6) --------------------------------------------------
    anthropic_api_key: str | None = None         # or ANTHROPIC_API_KEY / `ant auth login`
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"                # low | medium | high | xhigh | max
    claude_fallback: bool = True                 # server-side refusal fallback

    @property
    def index_meta_path(self) -> Path:
        return self.data_dir / "index_meta.json"
