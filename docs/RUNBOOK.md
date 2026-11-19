# Runbook — rebuild everything from zero

All commands run from the `Legal_Contract_Clause_Search/` folder.
Plug the laptop in: on battery, Windows throttles the CPU and embedding is ~10x slower.

## 1. One-time setup
```powershell
uv sync                                   # install dependencies from uv.lock
copy .env.example .env                    # optional: API keys and settings
```
Data: `data/raw/cuad/CUAD_v1/` must contain `CUAD_v1.json` and `master_clauses.csv`
(CUAD v1 from Zenodo, record 4595826).

Models: `models/` must contain bge-small (it does). Optional models need a network
that allows huggingface.co (the office network blocks it):
```powershell
uv run lccs download-models               # bge-base + reranker
```

## 2. Build the index
```powershell
uv run lccs ingest                        # all 510 contracts, clause chunker, bge-small, Qdrant
uv run lccs ingest --limit 20             # quick test with 20 contracts
uv run lccs stats                         # what's been built
```
Plugged in: ~10–15 min for all contracts. Each chunker/embedder gets its own collection.

## 3. Use it
```powershell
uv run lccs search "governing law" --type License --after 2015-01-01
uv run lccs search "can we end the deal early?" --mode dense --per-contract 1
uv run lccs answer "Which law governs the distributor agreements?" --type Distributor   # needs ANTHROPIC_API_KEY
uv run lccs serve                         # http://localhost:8000/docs
```

## 4. Evaluate
```powershell
uv run lccs eval                          # all stages, 50 contracts -> results/eval_*.csv
uv run lccs eval --stage search --limit 100
uv run pytest                             # unit tests
```

## Common problems
| Symptom | Cause | Fix |
|---|---|---|
| `Storage folder ... is already accessed by another instance` | Qdrant local mode allows ONE process | Stop `lccs serve` before `lccs ingest` (or use a Qdrant server via `QDRANT_URL`) |
| `CERTIFICATE_VERIFY_FAILED` / "Model ... is not downloaded yet" | Office network blocks huggingface.co | Run `lccs download-models` on another network once |
| `Collection ... not found` | Index for this chunker/embedder not built | `uv run lccs ingest` (same `--chunker`/`--embedder`) |
| `/answer` returns 503 "credentials" | No Claude API key | `ANTHROPIC_API_KEY=...` in `.env` |
| Embedding very slow (<10 chunks/s) | Laptop on battery / other heavy apps | Plug in; close heavy apps |
| Path errors on contract files | Windows 260-char path limit | Long paths are enabled (registry `LongPathsEnabled=1`) |
