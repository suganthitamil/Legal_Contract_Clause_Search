import pytest

from legal_contract_clause_search.eval.metrics import (
    clause_integrity,
    hit_at_k,
    ndcg_at_k,
    overlaps,
    precision_at_k,
    reciprocal_rank,
)


def test_overlaps():
    assert overlaps(51800, 52300, 52061, 52151)        # gold fully inside chunk
    assert overlaps(51800, 52100, 52061, 52151)        # chunk holds the start
    assert overlaps(52100, 52400, 52061, 52151)        # chunk holds the end
    assert not overlaps(3000, 3400, 52061, 52151)      # far away
    assert not overlaps(52151, 52400, 52061, 52151)    # touching edges only (end is exclusive)


def test_rank_metrics():
    rels = [False, True, False, True]
    assert hit_at_k(rels, 1) == 0.0
    assert hit_at_k(rels, 2) == 1.0
    assert reciprocal_rank(rels) == 0.5
    assert reciprocal_rank([False, False]) == 0.0
    assert precision_at_k(rels, 4) == 0.5


def test_ndcg():
    assert ndcg_at_k([True, False], n_relevant=1, k=10) == 1.0          # perfect
    assert ndcg_at_k([False, False], n_relevant=1, k=10) == 0.0
    assert ndcg_at_k([False, True], n_relevant=1, k=10) == pytest.approx(0.6309, abs=1e-3)


def test_clause_integrity():
    gold = [(10, 20), (30, 60)]
    assert clause_integrity(gold, [(0, 25), (25, 50), (50, 80)]) == 0.5   # 2nd gold is split
    assert clause_integrity(gold, [(0, 100)]) == 1.0
