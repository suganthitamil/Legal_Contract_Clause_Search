"""Clause-aware (structure-based) chunking — the core of this project.

Contracts are already split into clauses by the lawyers who wrote them:

    ARTICLE VI  MISCELLANEOUS
    6.8   Notices. All notices under this Agreement ...
    6.9   Governing Law. This Agreement is to be construed ...

So instead of cutting every N tokens, we:
  1. DETECT heading lines (ARTICLE / Section / 6.9 / 1  TERM / ALL-CAPS titles)
  2. SPLIT the text at those headings -> one segment per clause
  3. BUILD a heading path for each segment, e.g. "Article VI Miscellaneous > 6.9 Governing Law"
  4. MERGE tiny segments (a heading alone, a table-of-contents line) into the next one
  5. SPLIT oversized clauses (> max tokens) recursively, keeping the same heading path
  6. FALL BACK to recursive splitting for documents without detectable structure

This is the "AST-based" idea from code chunking, applied to legal documents:
the heading hierarchy is the document's syntax tree.
"""

import re
from dataclasses import dataclass

from legal_contract_clause_search.chunking.base import Chunker, Span, split_recursive

_ARTICLE = re.compile(r"^[ \t]*(?:ARTICLE|Article)[ \t]+([IVXLC]+|\d+)\b[ \t.:\-]*(.*)$")
_SECTION = re.compile(r"^[ \t]*(?:SECTION|Section)[ \t]+(\d+(?:\.\d+)*)\.?[ \t]*(.*)$")
# "6.9  Governing Law. ..." / "1    TERM" / "12.2(a) ..." — at most 2 digits per level,
# followed by a capital letter / quote / bracket, so "2010 revenue" or "30 days" don't match.
_NUMBERED = re.compile(r"^[ \t]*(\d{1,2}(?:\.\d{1,2}){0,3})[.)]?[ \t]+(?=[A-Z\"(])(.*)$")
# A short line in ALL CAPS on its own, e.g. "GOVERNING LAW" or "RECITALS".
_CAPS = re.compile(r"^[ \t]*([A-Z][A-Z0-9&,'\-/() ]{3,70})[ \t]*$")

_MIN_HEADINGS = 3     # fewer than this -> treat the document as unstructured


@dataclass
class _Heading:
    pos: int          # char position where the heading line starts
    level: int        # 0 = article, 1 = top section, 2 = sub-section ...
    label: str


def _short_title(rest: str) -> str:
    """'Governing Law. This Agreement is ...' -> 'Governing Law' (max 60 chars)."""
    title = re.split(r"\.(?:\s|$)", rest.strip(), maxsplit=1)[0]
    title = re.sub(r"\s+", " ", title).strip(" .:-")
    return title[:60]


def find_headings(text: str) -> list[_Heading]:
    headings = []
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if m := _ARTICLE.match(stripped):
            label = f"Article {m.group(1)} {_short_title(m.group(2))}".strip()
            headings.append(_Heading(pos, 0, label))
        elif m := _SECTION.match(stripped):
            num = m.group(1)
            headings.append(_Heading(pos, num.count(".") + 1, f"Section {num} {_short_title(m.group(2))}".strip()))
        elif m := _NUMBERED.match(stripped):
            num = m.group(1)
            headings.append(_Heading(pos, num.count(".") + 1, f"{num} {_short_title(m.group(2))}".strip()))
        elif (m := _CAPS.match(stripped)) and sum(c.isalpha() for c in m.group(1)) >= 4:
            headings.append(_Heading(pos, 1, re.sub(r"\s+", " ", m.group(1)).strip()[:60]))
        pos += len(line)
    return headings


class ClauseChunker(Chunker):
    name = "clause"

    def split(self, text: str) -> list[Span]:
        headings = find_headings(text)
        if len(headings) < _MIN_HEADINGS:
            return [Span(a, b) for a, b in split_recursive(text, 0, len(text), self.max_tokens, self.counter)]

        # --- 2 + 3: one segment per heading, with a heading path ------------------
        segments: list[Span] = []
        if headings[0].pos > 0:
            segments.append(Span(0, headings[0].pos, "Preamble"))
        stack: list[_Heading] = []
        for i, h in enumerate(headings):
            while stack and stack[-1].level >= h.level:
                stack.pop()
            stack.append(h)
            end = headings[i + 1].pos if i + 1 < len(headings) else len(text)
            segments.append(Span(h.pos, end, " > ".join(s.label for s in stack)))

        # --- 4: merge heading-only fragments forward --------------------------------
        # Only fragments with (almost) no body are merged: "7. NEW PRODUCTS." or a
        # table-of-contents line. A short but COMPLETE clause such as
        # "6.9 Governing Law. This Agreement is construed under Illinois law." (~30 tokens)
        # must stay on its own — merging it into section 7 would mislabel it.
        min_tokens = self.settings.chunk_min_tokens
        sizes = self.counter.count_batch([text[s.start:s.end] for s in segments])
        merged: list[tuple[Span, int]] = []
        for seg, n in zip(segments, sizes):
            if merged:
                prev, prev_n = merged[-1]
                if prev_n < min_tokens and prev_n + n <= self.max_tokens:
                    # the body's path already contains the parent heading,
                    # e.g. "7 NEW PRODUCTS > 7.1 Right of Option"
                    merged[-1] = (Span(prev.start, seg.end, seg.section_path), prev_n + n)
                    continue
            merged.append((seg, n))
        # a fragment at the very end joins the previous chunk
        if len(merged) > 1 and merged[-1][1] < min_tokens:
            last, last_n = merged.pop()
            prev, prev_n = merged[-1]
            if prev_n + last_n <= self.max_tokens:
                merged[-1] = (Span(prev.start, last.end, prev.section_path), prev_n + last_n)
            else:
                merged.append((last, last_n))

        # --- 5: split oversized clauses, keeping their heading path -----------------
        spans: list[Span] = []
        for seg, n in merged:
            if n <= self.max_tokens:
                spans.append(seg)
            else:
                for a, b in split_recursive(text, seg.start, seg.end, self.max_tokens, self.counter):
                    spans.append(Span(a, b, seg.section_path))
        return spans
