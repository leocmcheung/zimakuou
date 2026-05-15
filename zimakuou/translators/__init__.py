import platform
import sys

from .base import Translator


def get_translator(model: str | None = None) -> Translator:
    """Pick the right backend for the host platform."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        from .mlx_backend import MLXTranslator
        return MLXTranslator(model)
    if sys.platform == "win32":
        from .llama_backend import LlamaTranslator
        return LlamaTranslator(model)
    raise RuntimeError(
        f"Unsupported platform {sys.platform}/{platform.machine()}. "
        "MLX requires Apple Silicon; llama.cpp path is wired for Windows."
    )
