from datetime import date

import pytest

from legal_contract_clause_search.chunking import get_chunker
from legal_contract_clause_search.chunking.clause import find_headings
from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.model import Contract

BODY = "The Distributor shall use commercially reasonable efforts to promote the Products. " * 3
TEXT = f"""DISTRIBUTOR AGREEMENT

This Distributor Agreement is entered into by ACME Corp and Beta LLC.

ARTICLE I  APPOINTMENT

1.1 Grant. {BODY}

1.2 Territory. {BODY}

ARTICLE VI  MISCELLANEOUS

6.8 Notices. All notices under this Agreement shall be in writing and delivered by hand.

6.9 Governing Law. This Agreement is construed according to the laws of the State of Illinois.

7. NEW PRODUCTS.

7.1 Right of Option. {BODY}
"""


@pytest.fixture
def contract():
    return Contract(doc_id="acme", title="ACME Distributor Agreement", contract_type="Distributor",
                    sign_date=date(2015, 1, 1), parties=[], text=TEXT)


@pytest.fixture
def settings():
    return Settings()


def test_headings_detected():
    labels = [h.label for h in find_headings(TEXT)]
    assert "Article I APPOINTMENT" in labels
    assert "6.9 Governing Law" in labels
    assert "7 NEW PRODUCTS" in labels


@pytest.mark.parametrize("name", ["clause", "fixed", "sliding", "recursive"])
def test_spans_are_exact_slices(contract, settings, name):
    for c in get_chunker(name, settings).chunk(contract):
        assert contract.text[c.char_start:c.char_end] == c.text
        assert c.token_count <= settings.chunk_max_tokens + 5


def test_short_complete_clause_keeps_its_own_path(contract, settings):
    """Regression: 6.9 Governing Law is short but complete; it must not be merged into section 7."""
    chunks = get_chunker("clause", settings).chunk(contract)
    pos = TEXT.index("6.9 Governing Law")
    owner = next(c for c in chunks if c.char_start <= pos < c.char_end)
    assert owner.section_path == "Article VI MISCELLANEOUS > 6.9 Governing Law"


def test_heading_only_fragment_is_merged_forward(contract, settings):
    chunks = get_chunker("clause", settings).chunk(contract)
    pos = TEXT.index("7. NEW PRODUCTS")
    owner = next(c for c in chunks if c.char_start <= pos < c.char_end)
    assert "7.1 Right of Option" in owner.text
    assert owner.section_path.endswith("7.1 Right of Option")


def test_oversized_clause_is_split_with_same_path(settings):
    long_text = "1. TERM\n\n1.1 Duration. " + ("This Agreement continues in force. " * 300) + "\n\n2. FEES\n\n2.1 Fees. x\n"
    contract = Contract(doc_id="d", title="t", contract_type="Service", sign_date=None, parties=[], text=long_text)
    chunks = get_chunker("clause", settings).chunk(contract)
    duration = [c for c in chunks if c.section_path and c.section_path.endswith("1.1 Duration")]
    assert len(duration) > 1
    assert all(c.token_count <= settings.chunk_max_tokens + 5 for c in duration)


def test_sliding_windows_overlap(contract, settings):
    chunks = get_chunker("sliding", settings.model_copy(update={"chunk_max_tokens": 40})).chunk(contract)
    assert len(chunks) > 2
    assert all(b.char_start < a.char_end for a, b in zip(chunks, chunks[1:]))


def test_contextual_header(contract, settings):
    chunk = get_chunker("clause", settings.model_copy(update={"contextual_headers": True})).chunk(contract)[3]
    assert chunk.embed_text.startswith("Distributor agreement | ")
    no_hdr = get_chunker("clause", settings.model_copy(update={"contextual_headers": False})).chunk(contract)[3]
    assert no_hdr.embed_text == no_hdr.text
