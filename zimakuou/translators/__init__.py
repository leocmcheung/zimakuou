import platform
import sys

from .base import Translator


def get_translator(model: str | None = None, system: str | None = None) -> Translator:
    """Pick the right backend for the host platform. `system` overrides the
    default system prompt — used to inject show context / glossary."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        from .mlx_backend import MLXTranslator
        return MLXTranslator(model, system=system)
    if sys.platform == "win32":
        from .llama_backend import LlamaTranslator
        return LlamaTranslator(model, system=system)
    raise RuntimeError(
        f"Unsupported platform {sys.platform}/{platform.machine()}. "
        "MLX requires Apple Silicon; llama.cpp path is wired for Windows."
    )
