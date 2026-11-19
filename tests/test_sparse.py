from legal_contract_clause_search.sparse import BM25Encoder, stem, term_id, tokenize


def test_stemming_groups_word_forms():
    assert stem("termination") == stem("terminate") == stem("terminated")
    assert stem("assignment") == stem("assigns")


def test_tokenize_drops_stopwords():
    # stopwords (the, shall, not, this) removed; "agreement" stemmed to "agree"
    assert tokenize("The Party shall not assign this Agreement") == ["party", "assign", "agree"]


def test_term_ids_are_stable():
    assert term_id("indemn") == term_id("indemn")       # crc32, not Python's randomized hash()


def test_bm25_weights():
    enc = BM25Encoder.fit(["licensee shall indemnify licensor", "governing law of new york"])
    doc = enc.encode_document("indemnify indemnify licensor")
    weights = dict(zip(doc.indices, doc.values))
    assert weights[term_id(stem("indemnify"))] > weights[term_id("licensor")]   # tf=2 beats tf=1
    q = enc.encode_query("who must indemnify?")
    assert term_id(stem("indemnify")) in q.indices
