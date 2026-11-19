"""A small JSON file recording what each index (collection) was built with.

Search needs to know some ingest-time facts, e.g. the BM25 average chunk length,
and it's useful to see which chunker/embedder/how many chunks an index has
(`uv run lccs stats`).
"""

import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_meta(path: Path, key: str, meta: dict) -> None:
    data = _load(path)
    data[key] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def read_meta(path: Path, key: str) -> dict | None:
    return _load(path).get(key)


def all_meta(path: Path) -> dict:
    return _load(path)
