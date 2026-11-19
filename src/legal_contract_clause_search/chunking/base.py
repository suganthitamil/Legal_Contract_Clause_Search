"""Shared chunking machinery.

KEY DESIGN IDEA: every chunker works with *character positions* (spans), never
with copied text. A span (start, end) means `contract.text[start:end]`.
Keeping exact positions is what lets evaluation compare chunks with CUAD's
lawyer-highlighted gold clauses (`answer_start`) later.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.model import Chunk, Contract
from legal_contract_clause_search.tokens import TokenCounter


@dataclass
class Span:
    start: int
    end: int
    section_path: str | None = None


# Recursive splitting tries the "biggest" natural boundary first.
SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]


def split_by_tokens(text: str, start: int, end: int, max_tokens: int, counter: TokenCounter,
                    overlap: int = 0) -> list[tuple[int, int]]:
    """Cut text[start:end] into windows of `max_tokens` tokens (last resort / fixed chunker)."""
    offsets = counter.offsets(text[start:end])
    if not offsets:
        return [(start, end)] if text[start:end].strip() else []
    step = max(1, max_tokens - overlap)
    windows = []
    for i in range(0, len(offsets), step):
        window = offsets[i:i + max_tokens]
        windows.append((start + window[0][0], start + window[-1][1]))
        if i + max_tokens >= len(offsets):
            break
    return windows


def split_recursive(text: str, start: int, end: int, max_tokens: int, counter: TokenCounter,
                    separators: list[str] = SEPARATORS) -> list[tuple[int, int]]:
    """Split text[start:end] into pieces of <= max_tokens, preferring big boundaries.

    1. Fits already? -> one piece.
    2. Otherwise cut at the first separator that occurs (paragraph, then line, then sentence...).
    3. Greedily glue neighbouring pieces back together while they still fit.
    4. Any single piece still too big -> recurse with the next, smaller separator.
    """
    if counter.count(text[start:end]) <= max_tokens:
        return [(start, end)]

    segment = text[start:end]
    for i, sep in enumerate(separators):
        cuts, pos = [], segment.find(sep)
        while pos != -1:
            cuts.append(start + pos + len(sep))      # keep the separator with the left piece
            pos = segment.find(sep, pos + len(sep))
        cuts = [c for c in cuts if start < c < end]
        if cuts:
            remaining = separators[i + 1:]
            break
    else:
        return split_by_tokens(text, start, end, max_tokens, counter)

    bounds = [start, *cuts, end]
    pieces = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
    sizes = counter.count_batch([text[a:b] for a, b in pieces])

    out: list[tuple[int, int]] = []
    cur_start, cur_end, cur_tokens = None, None, 0
    for (a, b), n in zip(pieces, sizes):
        if n > max_tokens:                           # piece itself too big -> go finer
            if cur_start is not None:
                out.append((cur_start, cur_end))
                cur_start, cur_tokens = None, 0
            out.extend(split_recursive(text, a, b, max_tokens, counter, remaining or [" "]))
            continue
        if cur_start is not None and cur_tokens + n > max_tokens:
            out.append((cur_start, cur_end))
            cur_start, cur_tokens = None, 0
        if cur_start is None:
            cur_start = a
        cur_end, cur_tokens = b, cur_tokens + n
    if cur_start is not None:
        out.append((cur_start, cur_end))
    return out


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Move span edges inward past whitespace (positions stay exact)."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


class Chunker(ABC):
    name: str = "base"

    def __init__(self, settings: Settings, counter: TokenCounter):
        self.settings = settings
        self.counter = counter
        self.max_tokens = settings.chunk_max_tokens

    @abstractmethod
    def split(self, text: str) -> list[Span]:
        """Return spans covering the meaningful parts of `text`."""

    def chunk(self, contract: Contract) -> list[Chunk]:
        spans = []
        for s in self.split(contract.text):
            a, b = trim(contract.text, s.start, s.end)
            if b > a:
                spans.append(Span(a, b, s.section_path))

        texts = [contract.text[s.start:s.end] for s in spans]
        token_counts = self.counter.count_batch(texts)
        chunks = []
        for i, (span, text, n_tokens) in enumerate(zip(spans, texts, token_counts)):
            chunks.append(Chunk(
                chunk_id=f"{contract.doc_id}::{self.name}::{i:04d}",
                doc_id=contract.doc_id,
                text=text,
                char_start=span.start,
                char_end=span.end,
                chunk_index=i,
                section_path=span.section_path,
                contract_type=contract.contract_type,
                sign_date=contract.sign_date,
                title=contract.title,
                chunker=self.name,
                token_count=n_tokens,
                embed_text=self.embed_text(contract, span.section_path, text),
            ))
        return chunks

    def embed_text(self, contract: Contract, section_path: str | None, text: str) -> str:
        """Contextual header: the chunk alone may not say what it is part of.

        "This Agreement shall be governed by..." embeds better as
        "Distributor agreement | 6. Miscellaneous > 6.9 Governing Law\\nThis Agreement...".
        """
        if not self.settings.contextual_headers:
            return text
        header = f"{contract.contract_type} agreement"
        if section_path:
            header += f" | {section_path}"
        return f"{header}\n{text}"
