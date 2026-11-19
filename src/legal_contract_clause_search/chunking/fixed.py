"""Baseline strategies: fixed-size and sliding-window chunks.

FIXED   : cut every N tokens. Simple and predictable, but cuts clauses mid-sentence.
SLIDING : same, but each window overlaps the previous one, so a clause cut at a
          boundary still appears whole in at least one chunk. Costs more chunks.
"""

from legal_contract_clause_search.chunking.base import Chunker, Span, split_by_tokens


class FixedChunker(Chunker):
    name = "fixed"
    overlap = 0

    def split(self, text: str) -> list[Span]:
        overlap = self.overlap_tokens()
        return [Span(a, b) for a, b in split_by_tokens(text, 0, len(text), self.max_tokens,
                                                       self.counter, overlap)]

    def overlap_tokens(self) -> int:
        return 0


class SlidingWindowChunker(FixedChunker):
    name = "sliding"

    def overlap_tokens(self) -> int:
        return self.settings.chunk_overlap_tokens
