# Legal Contract Clause Search Engine

Search clauses across 510 commercial contracts (CUAD) by meaning and keywords,
filtered by **contract type** and **sign date**, with clause-aware chunking in **Qdrant**.
Optional: cross-encoder reranking and Claude answers with citations.

```powershell
uv sync
uv run lccs ingest                     # build the index (plug the laptop in)
uv run lccs search "governing law" --type License --after 2015-01-01
uv run lccs serve                      # API + Swagger UI at http://localhost:8000/docs
uv run lccs eval                       # measure retrieval quality -> results/
uv run pytest
```

| Piece | Choice |
|---|---|
| Chunking | Clause-aware: headings → clause chunks with heading paths (+ fixed, sliding, recursive, semantic for comparison) |
| Embeddings | `BAAI/bge-small-en-v1.5` (local); bge-base and `voyage-law-2` optional |
| Keyword search | BM25 sparse vectors (own implementation, no download) |
| Vector DB | Qdrant (local mode or server); Chroma adapter for comparison |
| Search | Hybrid dense + BM25 with Reciprocal Rank Fusion, payload filters, optional reranker |
| Generation | Claude (`claude-opus-5-5`) with document citations |
| API | FastAPI: `/search`, `/answer`, `/contract-types`, `/health` |

Docs:
- [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md): follow one query through the code.
- [docs/RUNBOOK.md](docs/RUNBOOK.md): commands and troubleshooting.
- [docs/DECISIONS.md](docs/DECISIONS.md): why each choice was made, with evidence.
