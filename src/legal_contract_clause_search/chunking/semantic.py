"""Semantic chunking: split where the MEANING changes.

1. Split the text into sentences.
2. Embed every sentence.
3. Measure how different each sentence is from the next (1 - cosine similarity).
4. Start a new chunk where that difference is unusually large (above the
   document's own Nth percentile) — or when the chunk would exceed max tokens.

Pros: no reliance on formatting. Cons: needs an embedding call per sentence
(slow), and topic shifts inside legal text are subtle.
"""

import re

import numpy as np

from legal_contract_clause_search.chunking.base import Chunker, Span, split_recursive
from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.embeddings import Embedder
from legal_contract_clause_search.tokens import TokenCounter

# sentence end: . ; : ! ? followed by whitespace and a capital/quote/digit, or a blank line
_BOUNDARY = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z\"(0-9])|\n[ \t]*\n")


class SemanticChunker(Chunker):
    name = "semantic"

    def __init__(self, settings: Settings, counter: TokenCounter, embedder: Embedder,
                 percentile: float = 85.0, min_sentence_tokens: int = 8):
        super().__init__(settings, counter)
        self.embedder = embedder
        self.percentile = percentile
        self.min_sentence_tokens = min_sentence_tokens

    def _sentences(self, text: str) -> list[tuple[int, int]]:
        spans, start = [], 0
        for m in _BOUNDARY.finditer(text):
            if text[start:m.start()].strip():
                spans.append((start, m.start()))
            start = m.end()
        if text[start:].strip():
            spans.append((start, len(text)))
        # glue very short "sentences" (e.g. "1.", "(a)") onto the next one
        sizes = self.counter.count_batch([text[a:b] for a, b in spans])
        merged: list[tuple[int, int]] = []
        carry = None
        for (a, b), n in zip(spans, sizes):
            if carry is not None:
                a = carry
                carry = None
            if n < self.min_sentence_tokens:
                carry = a
                continue
            merged.append((a, b))
        if carry is not None:
            if merged:
                merged[-1] = (merged[-1][0], len(text))
            else:
                merged.append((carry, len(text)))
        return merged

    def split(self, text: str) -> list[Span]:
        sentences = self._sentences(text)
        if len(sentences) <= 1:
            return [Span(a, b) for a, b in split_recursive(text, 0, len(text), self.max_tokens, self.counter)]

        vectors = self.embedder.embed_documents([text[a:b] for a, b in sentences])  # normalized
        distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = float(np.percentile(distances, self.percentile))
        sizes = self.counter.count_batch([text[a:b] for a, b in sentences])

        spans: list[Span] = []
        cur_start, cur_end, cur_tokens = sentences[0][0], sentences[0][1], sizes[0]
        for i in range(1, len(sentences)):
            a, b = sentences[i]
            topic_shift = distances[i - 1] > threshold
            if topic_shift or cur_tokens + sizes[i] > self.max_tokens:
                spans.append(Span(cur_start, cur_end))
                cur_start, cur_tokens = a, 0
            cur_end, cur_tokens = b, cur_tokens + sizes[i]
        spans.append(Span(cur_start, cur_end))

        out: list[Span] = []
        for s in spans:   # a single huge sentence can still exceed the limit
            if self.counter.count(text[s.start:s.end]) > self.max_tokens:
                out.extend(Span(a, b) for a, b in split_recursive(text, s.start, s.end, self.max_tokens, self.counter))
            else:
                out.append(s)
        return out
