"""Recursive character splitting (the LangChain-style default).

Split on paragraphs; if a piece is still too long split it on lines, then
sentences, then words. Respects natural text boundaries, but knows nothing
about contract structure (headings / clause numbers).
"""

from legal_contract_clause_search.chunking.base import Chunker, Span, split_recursive


class RecursiveChunker(Chunker):
    name = "recursive"

    def split(self, text: str) -> list[Span]:
        return [Span(a, b) for a, b in split_recursive(text, 0, len(text), self.max_tokens, self.counter)]
