# Walkthrough — follow one contract and one query through the code

Read this with the code open. Each step names the file and function to look at.
All paths are under `src/legal_contract_clause_search/`.

```
INGEST (offline, once)                                   SEARCH (every request)
──────────────────────                                   ──────────────────────
loaders/cuad.py      load_cuad()                          api.py / cli.py     POST /search, lccs search
      │  510 Contract objects                                   │  SearchRequest(query, filters, mode...)
chunking/clause.py   ClauseChunker.split()                retrieval.py        Retriever.search()
      │  Chunk objects with char offsets + heading path         │  embed query + BM25 query
embeddings.py        embed_documents()  -> dense vectors  vectorstore/qdrant_store.py  search()
sparse.py            BM25Encoder        -> sparse vectors       │  filter + dense/sparse/hybrid (RRF)
vectorstore/qdrant_store.py  create() + upsert()          rerank.py           (optional) cross-encoder
      │  points: vectors + payload                              │
data/qdrant/  +  data/index_meta.json                     list[SearchHit]  ──► generation.py (optional)
                                                                              Claude answer + citations
```

---

## Part A — Ingest: `uv run lccs ingest`

### A1. `cli.py` → `ingest()`
Parses the options and calls `ingest.ingest(settings, chunker, embedder, ...)`.
`config.py` supplies every default (chunker `clause`, embedder `bge-small`, store `qdrant`).

### A2. `loaders/cuad.py` → `load_cuad()`  *(you wrote this one)*
- Text from `CUAD_v1.json`, sign date + parties from `master_clauses.csv`, joined by `make_key()`.
- `parse_contract_type()` maps titles to ~25 types with `CONTRACT_TYPE_RULES`.
- Output: 510 `Contract` objects (`model.py`).

### A3. `ingest.py` → `select_contracts()`
With `--limit N`, takes a *seeded* random sample, so evaluation always sees the same contracts.

### A4. `chunking/clause.py` → `ClauseChunker.split()`  ← the core
1. `find_headings()` scans every line with 4 regexes (ARTICLE / Section / `6.9` / ALL-CAPS).
2. Splits the text at headings; keeps a **stack** of open headings to build the path
   `Article VI MISCELLANEOUS > 6.9 Governing Law`.
3. Merges *heading-only* fragments (< `chunk_min_tokens`) into the next segment.
   (A complete short clause stays alone — see the regression test in `tests/test_chunking.py`.)
4. Splits clauses longer than `chunk_max_tokens` with `split_recursive()` (`chunking/base.py`).
5. Documents with < 3 headings fall back to recursive splitting.

`chunking/base.py` → `Chunker.chunk()` turns spans into `Chunk` objects:
- `text` = exact slice `contract.text[char_start:char_end]` (needed for evaluation).
- `embed_text` = contextual header + text, e.g. `"Distributor agreement | 6 ... > 6.9 Governing Law\nThis Agreement..."`.
- Token counts come from the bge tokenizer (`tokens.py`).

### A5. `embeddings.py` → `FastEmbedEmbedder.embed_documents()`
Batches of text → 384-number vectors, normalized to length 1 (so cosine = dot product).

### A6. `sparse.py` → `BM25Encoder`
`fit()` computes the average chunk length; `encode_document()` turns each chunk into
`{word_id: weight}`. Qdrant adds the IDF part at search time.

### A7. `vectorstore/qdrant_store.py` → `create()` + `upsert()`
- Collection with two named vectors: `dense` (cosine) and `sparse` (IDF modifier).
- Payload indexes on `contract_type`, `sign_date`, `doc_id`.
- Point id = UUID from `chunk_id` → re-ingesting overwrites instead of duplicating.
- `index_meta.py` records chunk count, BM25 average length etc. in `data/index_meta.json`.

---

## Part B — Search: `POST /search`

Example body:
```json
{"query": "can either party end the agreement early?", "contract_types": ["License"],
 "signed_after": "2015-01-01", "top_k": 5}
```

### B1. `api.py` → `search()`
FastAPI validates the JSON into a `SearchRequest` (`model.py`) — bad input gets a 422
error automatically. One `Retriever` is created at startup (`lifespan`).

### B2. `retrieval.py` → `Retriever.search()`
1. `resolve_mode()`: `hybrid` by default (falls back to `dense` if the store has no BM25).
2. How many candidates: `top_k`, or `rerank_candidates` (20) if reranking.
3. Query vectors: `embed_query()` (dense) and `bm25.encode_query()` (sparse).
4. `store.search(...)` (next step).
5. Optional `reranker.rerank()` and `limit_per_contract()`.

### B3. `vectorstore/qdrant_store.py` → `build_filter()` + `search()`
- `build_filter()` turns the filters into Qdrant conditions:
  `MatchAny(contract_type)`, `DatetimeRange(sign_date gte/lte)`, `MatchAny(doc_id)`.
- `hybrid`: two `Prefetch` searches (dense and sparse, each with the filter) fused with
  **Reciprocal Rank Fusion**.

### B4. `rerank.py` (if the model is downloaded)
The cross-encoder reads (query, chunk) together, re-scores the 20 candidates, and the best `top_k` are kept.

### B5. Response
`list[SearchHit]`: score, contract title/type/date, section path, clause text, char offsets.

---

## Part C — Answer: `POST /answer` (needs `ANTHROPIC_API_KEY`)

`generation.py` → `answer()`
1. Runs the same search.
2. Sends each hit to Claude as a **document block with citations enabled**, plus the question.
3. The system prompt says: answer only from the clauses, compare contracts, and say so if they don't answer it.
4. Claude's text blocks carry `citations` (exact `cited_text` + `document_index`) → turned into `[1]`, `[2]` markers and a `citations` list.

---

## Part D — Evaluation: `uv run lccs eval`

`eval/runner.py` builds separate indexes in `data/qdrant_eval/` for a fixed contract sample, then:
- `eval/goldset.py`: CUAD lawyer highlights for 10 clause categories + 2 user-style questions each.
- **Within-contract**: filter `doc_id`, ask the question → is a chunk overlapping the highlight in the top k?
- **Corpus-wide**: no doc filter → precision@10.
- **Filter check**: every hit must satisfy random type/date filters.
- `eval/metrics.py`: hit@k, MRR, nDCG@10, precision@10, clause integrity.

Results: `results/eval_<stage>.csv`. Decisions taken from them: `docs/DECISIONS.md`.

---

## Where to change things
| I want to... | File |
|---|---|
| Change a default (mode, chunk size, model) | `.env` or `config.py` |
| Detect another heading style | `chunking/clause.py` regexes |
| Add a contract type | `CONTRACT_TYPE_RULES` in `loaders/cuad.py` |
| Add a filter (e.g. governing law) | `model.py` `SearchFilters` → `qdrant_store.build_filter` → payload in `vectorstore/base.py` |
| Try another embedding model | `EMBEDDERS` in `embeddings.py` |
| Add an evaluation question | `CATEGORY_QUERIES` in `eval/goldset.py` |
