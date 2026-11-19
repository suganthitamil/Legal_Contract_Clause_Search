"""BM25 keyword vectors (the "sparse" half of hybrid search). No model download needed.

WHY: embeddings match MEANING but can blur exact legal terms ("indemnify",
"force majeure", "most favored nation"). Keyword search matches WORDS exactly.
Hybrid search runs both and fuses the rankings (see vectorstore/qdrant_store.py).

HOW (BM25):
  score(doc, query) = sum over query words w of  IDF(w) * TF_part(w, doc)
    IDF(w)      = how RARE the word is across all chunks  -> "indemnify" counts more than "agreement"
    TF_part     = tf*(k1+1) / (tf + k1*(1 - b + b*len/avg_len))
                  -> more occurrences help, with diminishing returns; long chunks are penalized a bit
We compute TF_part here; Qdrant computes IDF itself (sparse vector modifier=IDF).

A sparse vector stores only the non-zero entries: {word_id: weight}. Word ids
come from a stable hash of the word (crc32) so the same word always maps to the same id.
"""

import re
import zlib
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
STOPWORDS = set("""
a an the and or nor but if then else of to in on at by for from with without within into onto upon
as is are was were be been being am do does did doing have has had having this that these those
it its it's he she they them their his her we our you your i me my which who whom whose what when
where why how all any both each few more most other some such no not only own same so than too very
can will just should would could may might must shall hereby herein hereof hereto hereunder thereof
therein thereto such said per via etc
""".split())

# Naive suffix stripping so "terminate / termination / terminated" share one word id.
_SUFFIXES = ("ifications", "ification", "ations", "ation", "ments", "ment", "ities", "ity",
             "ings", "ing", "ions", "ion", "ates", "ated", "ate", "ies", "ied", "ify",
             "ed", "es", "s")


def stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def term_id(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class BM25Encoder:
    def __init__(self, avg_len: float, k1: float = 1.2, b: float = 0.75):
        self.avg_len, self.k1, self.b = max(avg_len, 1.0), k1, b

    @classmethod
    def fit(cls, texts: list[str]) -> "BM25Encoder":
        lengths = [len(tokenize(t)) for t in texts]
        return cls(sum(lengths) / max(len(lengths), 1))

    def encode_document(self, text: str) -> SparseVec:
        terms = tokenize(text)
        norm = self.k1 * (1 - self.b + self.b * len(terms) / self.avg_len)
        weights: dict[int, float] = {}
        for term, tf in Counter(terms).items():
            tid = term_id(term)
            weights[tid] = weights.get(tid, 0.0) + tf * (self.k1 + 1) / (tf + norm)
        return SparseVec(list(weights), list(weights.values()))

    def encode_query(self, text: str) -> SparseVec:
        ids = sorted({term_id(t) for t in tokenize(text)})
        return SparseVec(ids, [1.0] * len(ids))
