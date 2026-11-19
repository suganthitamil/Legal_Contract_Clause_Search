"""Run evaluation experiments and write results/eval_<stage>.csv.

    uv run lccs eval --stage chunkers     # which chunking strategy?       (dense search)
    uv run lccs eval --stage search       # dense vs BM25 vs hybrid (+ rerank)
    uv run lccs eval --stage embedders    # bge-small vs bge-base vs voyage-law-2 (if available)
    uv run lccs eval --stage stores       # Qdrant vs Chroma
    uv run lccs eval                      # all of the above

Uses a fixed random sample of contracts (default 50) and a SEPARATE Qdrant folder
(data/qdrant_eval), so it never touches the main index. Indexes are reused on
re-runs if they already exist.

Two evaluation views:
  WITHIN-CONTRACT : "In contract X, where is the governing-law clause?"  (filter doc_id=X)
                    -> hit@1/3/5, MRR, nDCG@10. Tests chunking + embeddings + ranking.
  CORPUS-WIDE     : "Find governing-law clauses" across all sample contracts
                    -> precision@10. Tests ranking across many documents.
Plus:  FILTER CHECK (every hit must satisfy type/date filters), clause integrity,
       chunk stats, query latency.
"""

import csv
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import date

from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.embeddings import EMBEDDERS
from legal_contract_clause_search.eval.goldset import CATEGORY_QUERIES, load_gold
from legal_contract_clause_search.eval.metrics import (
    clause_integrity,
    hit_at_k,
    is_relevant,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    reciprocal_rank,
)
from legal_contract_clause_search.ingest import collection_name, ingest, select_contracts
from legal_contract_clause_search.loaders.cuad import load_cuad
from legal_contract_clause_search.model import SearchRequest
from legal_contract_clause_search.model_cache import is_cached
from legal_contract_clause_search.retrieval import Retriever
from legal_contract_clause_search.vectorstore.chroma_store import ChromaStore
from legal_contract_clause_search.vectorstore.qdrant_store import QdrantStore

log = logging.getLogger(__name__)


@dataclass
class Config:
    label: str
    chunker: str = "clause"
    embedder: str = "bge-small"
    mode: str = "dense"
    rerank: bool = False
    store: str = "qdrant"
    headers: bool | None = None      # None -> use settings.contextual_headers


def stage_configs(stage: str, settings: Settings) -> list[Config]:
    reranker_ok = is_cached(settings.reranker_model_name, "rerank", settings.models_dir)
    if stage == "chunkers":
        # every chunker WITH headers (as first measured), plus clause without headers
        return [Config(c, chunker=c, headers=True) for c in ["fixed", "sliding", "recursive", "semantic", "clause"]] + \
               [Config("clause-no-headers", headers=False)]
    if stage == "search":
        cfgs = [Config("dense", mode="dense"), Config("sparse (BM25)", mode="sparse"), Config("hybrid", mode="hybrid")]
        if reranker_ok:
            cfgs += [Config("dense + rerank", mode="dense", rerank=True),
                     Config("hybrid + rerank", mode="hybrid", rerank=True)]
        else:
            log.warning("Reranker not downloaded -> rerank configs skipped (run `lccs download-models`).")
        return cfgs
    if stage == "embedders":
        cfgs = [Config("bge-small", embedder="bge-small")]
        if is_cached(EMBEDDERS["bge-base"][1], "dense", settings.models_dir):
            cfgs.append(Config("bge-base", embedder="bge-base"))
        else:
            log.warning("bge-base not downloaded -> skipped (run `lccs download-models`).")
        if settings.voyage_api_key:
            cfgs.append(Config("voyage-law-2", embedder="voyage-law-2"))
        else:
            log.warning("VOYAGE_API_KEY not set -> voyage-law-2 skipped.")
        return cfgs
    if stage == "stores":
        return [Config("qdrant", store="qdrant"), Config("chroma", store="chroma")]
    raise ValueError(f"Unknown stage {stage!r}")


class Evaluator:
    def __init__(self, settings: Settings, limit: int, k: int):
        self.base = settings.model_copy(update={
            "qdrant_path": settings.data_dir / "qdrant_eval",
            "chroma_path": settings.data_dir / "chroma_eval",
            "qdrant_url": None,
        })
        self.k = k
        self.prefix = f"eval{limit}"
        self.contracts = select_contracts(load_cuad(settings), limit)
        self.doc_ids = {c.doc_id for c in self.contracts}
        self.gold = load_gold(settings, self.doc_ids)
        self.gold_by_key = {(g.doc_id, g.category): g for g in self.gold}
        self.contract_by_id = {c.doc_id: c for c in self.contracts}
        self.cache_dir = settings.data_dir / "eval_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stores: dict = {}
        log.info("Eval sample: %d contracts, %d gold (contract, category) pairs, %d queries per config",
                 len(self.contracts), len(self.gold), 2 * len(self.gold))

    def store(self, name: str):
        if name not in self._stores:
            self._stores[name] = QdrantStore(self.base) if name == "qdrant" else ChromaStore(self.base)
        return self._stores[name]

    def settings_for(self, cfg: Config) -> Settings:
        headers = self.base.contextual_headers if cfg.headers is None else cfg.headers
        return self.base.model_copy(update={"contextual_headers": headers, "vector_store": cfg.store})

    def ensure_index(self, cfg: Config) -> dict[str, list[tuple[int, int]]]:
        """Build the index once; cache chunk spans per contract (needed for nDCG and integrity)."""
        s = self.settings_for(cfg)
        coll = collection_name(s, cfg.chunker, cfg.embedder, self.prefix)
        cache = self.cache_dir / f"{cfg.store}__{coll}.json"
        store = self.store(cfg.store)
        if cache.exists() and store.exists(coll):
            log.info("Reusing index %s", coll)
            return {k: [tuple(x) for x in v] for k, v in json.loads(cache.read_text()).items()}
        log.info("Building index %s (%s)", coll, cfg.store)
        _, chunks = ingest(s, cfg.chunker, cfg.embedder, store=store, limit=len(self.contracts),
                           recreate=True, collection_prefix=self.prefix)
        spans: dict[str, list[tuple[int, int]]] = {}
        for c in chunks:
            spans.setdefault(c.doc_id, []).append((c.char_start, c.char_end))
        cache.write_text(json.dumps(spans))
        return spans

    def run_config(self, cfg: Config) -> dict:
        spans_by_doc = self.ensure_index(cfg)
        s = self.settings_for(cfg).model_copy(update={"rerank": cfg.rerank})
        retriever = Retriever(s, chunker=cfg.chunker, embedder=cfg.embedder, store=self.store(cfg.store),
                              collection_prefix=self.prefix)
        mode = retriever.resolve_mode(cfg.mode)
        n_chunks = sum(len(v) for v in spans_by_doc.values())

        # --- within-contract ------------------------------------------------------
        h1, h3, h5, mrr, ndcg, lat = [], [], [], [], [], []
        for g in self.gold:
            doc_spans = spans_by_doc.get(g.doc_id, [])
            n_rel = sum(is_relevant(a, b, g.spans) for a, b in doc_spans)
            for q in CATEGORY_QUERIES[g.category]:
                t = time.perf_counter()
                hits = retriever.search(SearchRequest(query=q, doc_ids=[g.doc_id], top_k=self.k, mode=mode,
                                                      rerank=cfg.rerank))
                lat.append((time.perf_counter() - t) * 1000)
                rels = [is_relevant(h.char_start, h.char_end, g.spans) for h in hits]
                h1.append(hit_at_k(rels, 1)); h3.append(hit_at_k(rels, 3)); h5.append(hit_at_k(rels, 5))
                mrr.append(reciprocal_rank(rels)); ndcg.append(ndcg_at_k(rels, n_rel, 10))

        # --- corpus-wide precision@10 ----------------------------------------------
        p10 = []
        for category, queries in CATEGORY_QUERIES.items():
            for q in queries:
                hits = retriever.search(SearchRequest(query=q, top_k=10, mode=mode, rerank=cfg.rerank))
                rels = []
                for h in hits:
                    g = self.gold_by_key.get((h.doc_id, category))
                    rels.append(bool(g) and is_relevant(h.char_start, h.char_end, g.spans))
                p10.append(precision_at_k(rels, 10))

        # --- filter check ------------------------------------------------------------
        violations, checked = self.filter_check(retriever, mode, cfg.rerank)

        # --- chunking quality ---------------------------------------------------------
        integrity = mean([clause_integrity(g.spans, spans_by_doc.get(g.doc_id, [])) for g in self.gold])

        return {
            "label": cfg.label, "chunker": cfg.chunker + ("" if s.contextual_headers else " (no headers)"),
            "embedder": cfg.embedder, "store": cfg.store, "mode": mode, "rerank": cfg.rerank,
            "hit@1": round(mean(h1), 3), "hit@3": round(mean(h3), 3), "hit@5": round(mean(h5), 3),
            "MRR": round(mean(mrr), 3), "nDCG@10": round(mean(ndcg), 3), "corpus_P@10": round(mean(p10), 3),
            "clause_integrity": round(integrity, 3), "chunks": n_chunks,
            "filter_violations": f"{violations}/{checked}",
            "latency_p50_ms": round(percentile(lat, 50), 1), "latency_p95_ms": round(percentile(lat, 95), 1),
            "queries": len(h1),
        }

    def filter_check(self, retriever: Retriever, mode: str, rerank: bool) -> tuple[int, int]:
        rng = random.Random(7)
        types = sorted({c.contract_type for c in self.contracts})
        violations = checked = 0
        for category, queries in CATEGORY_QUERIES.items():
            ctype = rng.choice(types)
            after = date(rng.randint(1996, 2016), 1, 1)
            hits = retriever.search(SearchRequest(query=queries[0], contract_types=[ctype], signed_after=after,
                                                  top_k=10, mode=mode, rerank=rerank))
            for h in hits:
                checked += 1
                if h.contract_type != ctype or h.sign_date is None or h.sign_date < after:
                    violations += 1
        return violations, checked

    def close(self):
        for s in self._stores.values():
            s.close()


def _print_table(rows: list[dict]) -> None:
    cols = ["label", "hit@1", "hit@3", "hit@5", "MRR", "nDCG@10", "corpus_P@10", "clause_integrity",
            "chunks", "filter_violations", "latency_p50_ms"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


def run(settings: Settings, limit: int = 50, stage: str = "all", k: int = 10) -> list[dict]:
    stages = ["chunkers", "search", "embedders", "stores"] if stage == "all" else [stage]
    ev = Evaluator(settings, limit, k)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    try:
        for st in stages:
            rows = []
            for cfg in stage_configs(st, settings):
                log.info("=== [%s] %s", st, cfg.label)
                rows.append(ev.run_config(cfg))
            out = settings.results_dir / f"eval_{st}.csv"
            with out.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"\n## Stage: {st}  ({len(ev.contracts)} contracts)  -> {out}")
            _print_table(rows)
            all_rows += rows
    finally:
        ev.close()
    return all_rows
