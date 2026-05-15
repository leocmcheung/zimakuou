from __future__ import annotations

import os
import platform
import sys
from typing import TYPE_CHECKING

from .base import Translator
from ._prompt import build_system

if TYPE_CHECKING:
    from ..context import Context

# Project default — Sakura is fine-tuned for JP→ZH light-novel/galge/anime
# dialogue. Resolving the default *here* (not inside a backend) is load-bearing
# so style detection sees the real model name when `--llm` is omitted.
DEFAULT_MODEL = "SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF"


def _is_gguf(model: str) -> bool:
    """Heuristic: GGUF if path ends .gguf, the path exists locally, or repo id
    contains GGUF. MLX repos are mlx-community/* and don't match."""
    if model.endswith(".gguf") or os.path.exists(model):
        return True
    return "gguf" in model.lower()


def detect_style(model: str | None) -> str:
    """Sakura models need their own (Chinese) prompt format — auto-detect
    by name. Anything else is treated as a generic instruction model."""
    if model and "sakura" in model.lower():
        return "sakura"
    return "generic"


def get_translator(
    model: str | None = None,
    *,
    ctx: "Context | None" = None,
) -> Translator:
    """Pick the right backend for `model` + host and wire `ctx` into the
    prompt format the backend expects.

    GGUF models (path or HF GGUF repo) → llama.cpp on both platforms.
    Otherwise → MLX on Apple Silicon. Windows without GGUF is unsupported.

    `ctx` (synopsis / characters / glossary) is threaded into prompts
    differently per style:
    - generic: appended to the English system prompt as a context block.
    - sakura: glossary goes into the user prompt as a gpt_dict (the format
      the model was fine-tuned on); the system prompt is left as the
      verbatim training string and synopsis is dropped.
    """
    from ..context import Context

    ctx = ctx or Context.empty()
    resolved = model or DEFAULT_MODEL
    style = detect_style(resolved)

    if style == "sakura":
        system = None  # backend will use SAKURA_SYSTEM verbatim
        glossary = ctx.glossary
    else:
        system = build_system(ctx.llm_context_block())
        glossary = None  # generic style folds glossary into the system block

    if _is_gguf(resolved):
        from .llama_backend import LlamaTranslator
        return LlamaTranslator(resolved, system=system, style=style, glossary=glossary)

    if sys.platform == "darwin" and platform.machine() == "arm64":
        from .mlx_backend import MLXTranslator
        return MLXTranslator(resolved, system=system, style=style, glossary=glossary)

    raise RuntimeError(
        f"Unsupported platform {sys.platform}/{platform.machine()} for non-GGUF model "
        f"{resolved!r}. Use a GGUF model id/path, or run on Apple Silicon for MLX."
    )
