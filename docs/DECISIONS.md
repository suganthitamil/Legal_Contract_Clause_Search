# Decisions

Format: Decision · Options considered · Why · Evidence

## D1 — Contract text comes from CUAD_v1.json, metadata from master_clauses.csv (2026-11-13)
- **Options:** (a) `full_contract_txt/*.txt` + CSV, (b) `CUAD_v1.json` + CSV
- **Chosen:** (b)
- **Why:**
  - txt folder has only 499 files; CSV and JSON both have 510.
  - The JSON text is exactly the text that CUAD's gold clause offsets (`answer_start`) point into → Phase 5 evaluation lines up with no re-alignment.
- **Join rule:** key = lower-case letters + digits of the name (`make_key`). Raw names differ in spaces, dots, `.pdf`, trailing spaces.
- **Evidence:** raw stem match left 22 unmatched txt↔CSV; alnum key gives 510/510 JSON↔CSV matches, 0 warnings.

## D2 — Sign date = CSV "Agreement Date-Answer", format `%m/%d/%y`
- **Evidence:** 416/510 (82%) parse; 45 empty, rest in other formats → `None` for now (improve later).

## D3 — Contract type = keyword rules on the title → ~25 canonical types
- **Options:** (a) text after the last "-" in the filename, (b) official CUAD PDF folder names, (c) ordered keyword rules on the title
- **Chosen:** (c) `CONTRACT_TYPE_RULES` in `loaders/cuad.py`
- **Why:**
  - (a) gave 253 distinct types, 200 used once (e.g. "10.1_Development Agreement", "Branding Agreement") → useless as a filter.
  - (b) are lawyer labels but only 407 PDFs exist → cover 406/510 contracts; also duplicated spellings (`Affiliate_Agreements` vs `Affiliate Agreement`).
  - (c) covers all 510; we use (b) as an **answer key** to measure (c).
- **Rule:** first keyword found wins; specific before general (`co-branding` before `service`). No match → "Other".
- **Evidence:** 25 types, 6 "Other"; 397/406 = 98% agreement with official folder labels.

## How the evidence below was measured
`uv run lccs eval` (see `eval/runner.py`). For a fixed random sample of contracts, CUAD's lawyer
highlights for 10 clause categories are the answer key; each category has 2 questions phrased
like a user would ask (not copied from the contract). A chunk is relevant if it overlaps a highlight.
- **hit@k**: correct clause in the top k (within the right contract). **MRR**: 1/rank of the first correct chunk.
- **corpus P@10**: share of the top 10 that are correct when searching all sample contracts.
- **clause integrity**: share of highlighted clauses that fit inside one chunk.
⚠️ Numbers so far are from a **20-contract sample (96 clause/contract pairs, 192 queries)**, run on battery.
Re-run with `uv run lccs eval` (50 contracts) to confirm.

## D4 — Search mode: hybrid (dense + BM25, Reciprocal Rank Fusion) — 2026-11-17
- **Options:** dense only, BM25 only, hybrid (+ reranking once the model is downloaded)
- **Chosen:** hybrid (`SEARCH_MODE=hybrid`, the default)
- **Evidence (clause chunker, bge-small, 20 contracts):**

  | mode | hit@1 | hit@5 | MRR | corpus P@10 | p50 latency |
  |---|---|---|---|---|---|
  | dense | 0.359 | 0.729 | 0.514 | 0.285 | 22 ms |
  | sparse (BM25) | 0.323 | 0.682 | 0.471 | 0.255 | 20 ms |
  | **hybrid** | **0.474** | **0.786** | **0.603** | **0.340** | 52 ms |
- **Why:** hybrid is best on every quality metric (+32% hit@1 over dense). Embeddings and keywords
  fail on different queries, so fusing them recovers misses from both. Cost: ~30 ms more per query.
- **Filters:** 0 violations across all modes (199 filtered hits checked).
- **Re-checked after D5 (no contextual headers):** hybrid still wins — hit@1 0.505, hit@5 0.786,
  MRR 0.627, corpus P@10 0.335 (dense 0.438 / 0.745 / 0.567 / 0.320; BM25 0.328 / 0.688 / 0.479 / 0.250).
- **Open:** reranking not measured yet (model needs `lccs download-models` off the office network).

## D5 — Chunking: clause-aware, WITHOUT contextual headers — 2026-11-18
- **Options:** fixed (384 tokens), sliding window (384, overlap 64), recursive, semantic, clause-aware;
  clause with vs without a "type | section path" header prepended before embedding.
- **Chosen:** clause chunker, `CONTEXTUAL_HEADERS=false` (section path is still stored and shown — just not embedded).
- **Evidence (dense search, bge-small, 20 contracts, 192 within-contract questions):**

  | chunker | hit@1 | hit@5 | MRR | corpus P@10 | clause integrity | chunks |
  |---|---|---|---|---|---|---|
  | fixed | 0.422 | 0.797 | 0.576 | 0.275 | 0.853 | 596 |
  | sliding | 0.406 | 0.812 | 0.574 | 0.290 | 0.955 | 710 |
  | recursive | 0.401 | 0.776 | 0.557 | 0.280 | 0.981 | 879 |
  | semantic | 0.370 | 0.693 | 0.507 | 0.260 | 0.979 | 1135 |
  | clause + headers | 0.359 | 0.729 | 0.514 | 0.285 | 0.988 | 1211 |
  | **clause, no headers** | **0.438** | 0.745 | 0.567 | **0.320** | **0.988** | 1211 |
- **Why:**
  - Clause (no headers) has the best hit@1 and corpus precision, and keeps 98.8% of lawyer-highlighted
    clauses whole — fixed-size chunking cuts 15% of them in two.
  - Fixed/sliding look strong on hit@5/MRR partly because their chunks are 2x bigger: a big chunk
    overlaps the answer more easily (the overlap metric rewards size), but returns ~380 tokens with
    the clause buried inside. For a CLAUSE search engine the clause itself is the better result.
  - **Headers hurt (surprise):** prepending "type | section path" lowered every metric (hit@1 −18%).
    Hypothesis (not yet proven): the prefix is a large share of short chunks, and identical
    "X agreement" prefixes inside one contract pull vectors together. Worth re-testing with a
    section-path-only header and with larger models.
  - Semantic chunking: worst ranking and slowest to build (embeds every sentence) → dropped.
- **Ignore latency in this table:** later runs happened under memory pressure (same search: 22 ms earlier vs 162 ms).
