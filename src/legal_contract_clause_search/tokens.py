"""Token counting with the embedding model's own tokenizer.

WHY: chunk size limits must be measured in the tokens the EMBEDDING MODEL sees.
bge models read at most 512 tokens; anything longer is silently cut off, so a
"2000 character" limit is meaningless unless you know how many tokens that is.

We load `tokenizer.json` from the cached bge model (no download). If it's missing,
we fall back to a rough estimate (~4 characters per token).
"""

import re
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

_WORD_RE = re.compile(r"\S+")


class TokenCounter:
    def __init__(self, tokenizer_path: Path | None):
        self.exact = tokenizer_path is not None
        self._tok = None
        if tokenizer_path is not None:
            self._tok = Tokenizer.from_file(str(tokenizer_path))
            self._tok.no_truncation()
            self._tok.no_padding()

    def count(self, text: str) -> int:
        if self._tok is None:
            return max(1, len(text) // 4)
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_batch(self, texts: list[str]) -> list[int]:
        if self._tok is None:
            return [self.count(t) for t in texts]
        return [len(e.ids) for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """(char_start, char_end) of every token -> lets us cut text at token boundaries."""
        if self._tok is None:
            # approximation: treat each word as one token
            return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
        return list(self._tok.encode(text, add_special_tokens=False).offsets)


def find_tokenizer(models_dir: Path) -> Path | None:
    candidates = sorted(models_dir.glob("models--*bge-small*/snapshots/*/tokenizer.json"))
    candidates = candidates or sorted(models_dir.glob("**/tokenizer.json"))
    return candidates[0] if candidates else None


@lru_cache
def get_token_counter(models_dir: Path) -> TokenCounter:
    return TokenCounter(find_tokenizer(models_dir))
