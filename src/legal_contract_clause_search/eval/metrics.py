"""Retrieval metrics. `rels` = list of booleans, one per returned hit in rank order:
True if that hit overlaps a lawyer-highlighted (gold) clause.

  hit@k (Recall@k)  : is at least one relevant chunk in the top k?        (1 or 0 per query)
  MRR               : 1 / rank of the FIRST relevant chunk (0 if none)     -> rewards ranking it high
  nDCG@k            : all relevant chunks, discounted by rank, vs the ideal ordering
  precision@k       : fraction of the top k that are relevant
  clause integrity  : fraction of gold clauses that fit ENTIRELY inside one chunk (chunking quality)
"""

import math


def overlaps(chunk_start: int, chunk_end: int, gold_start: int, gold_end: int) -> bool:
    """Ranges share at least one character (end is exclusive, like slicing)."""
    return chunk_start < gold_end and gold_start < chunk_end


def is_relevant(start: int, end: int, gold_spans: list[tuple[int, int]]) -> bool:
    return any(overlaps(start, end, gs, ge) for gs, ge in gold_spans)


def hit_at_k(rels: list[bool], k: int) -> float:
    return 1.0 if any(rels[:k]) else 0.0


def reciprocal_rank(rels: list[bool]) -> float:
    for i, r in enumerate(rels, 1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: list[bool], n_relevant: int, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rels[:k]) if r)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / ideal if ideal > 0 else 0.0


def precision_at_k(rels: list[bool], k: int) -> float:
    return sum(rels[:k]) / k if k else 0.0


def clause_integrity(gold_spans: list[tuple[int, int]], chunk_spans: list[tuple[int, int]]) -> float:
    if not gold_spans:
        return 0.0
    inside = sum(any(cs <= gs and ge <= ce for cs, ce in chunk_spans) for gs, ge in gold_spans)
    return inside / len(gold_spans)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    idx = min(len(v) - 1, max(0, math.ceil(p / 100 * len(v)) - 1))
    return v[idx]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
