"""Know whether a Hugging Face model is already downloaded into `models/`.

The office network blocks huggingface.co downloads. fastembed would then retry
for ~40 seconds before failing, so we check the local cache FIRST and give a
clear message ("run `lccs download-models` on another network") instead.
"""

from pathlib import Path


class ModelNotAvailableError(RuntimeError):
    pass


def _hf_repo(model_name: str, kind: str) -> str:
    """fastembed often serves a converted (ONNX) copy from a different HF repo."""
    if kind == "dense":
        from fastembed import TextEmbedding
        models = TextEmbedding.list_supported_models()
    elif kind == "rerank":
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        models = TextCrossEncoder.list_supported_models()
    else:
        raise ValueError(kind)
    for m in models:
        if m["model"] == model_name:
            return (m.get("sources") or {}).get("hf") or model_name
    raise ValueError(f"{model_name!r} is not supported by fastembed ({kind})")


def is_cached(model_name: str, kind: str, models_dir: Path) -> bool:
    folder = models_dir / ("models--" + _hf_repo(model_name, kind).replace("/", "--"))
    return any(folder.glob("snapshots/*/*.onnx")) or any(folder.glob("snapshots/*/**/*.onnx"))


def require_cached(model_name: str, kind: str, models_dir: Path) -> None:
    if not is_cached(model_name, kind, models_dir):
        raise ModelNotAvailableError(
            f"Model {model_name!r} is not downloaded yet. Connect to a network that allows "
            f"huggingface.co and run:  uv run lccs download-models"
        )
