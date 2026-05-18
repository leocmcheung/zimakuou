from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .base import Translator
from ._prompt import (
    SAKURA_SYSTEM,
    SYSTEM,
    build_prompt,
    build_sakura_prompt,
)

if TYPE_CHECKING:
    from ..context import GlossaryEntry

DEFAULT_REPO = "SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF"
DEFAULT_FILE = "sakura-14b-qwen2.5-v1.0-q4km.gguf"


def _looks_like_hf_repo(model: str) -> bool:
    return "/" in model and not os.path.exists(model)


class LlamaTranslator(Translator):
    def __init__(
        self,
        model: str | None = None,
        system: str | None = None,
        style: str = "generic",
        glossary: list["GlossaryEntry"] | None = None,
        n_gpu_layers: int = -1,
    ):
        from llama_cpp import Llama

        # n_gpu_layers=-1 offloads every layer to Metal (Mac) / CUDA (Windows).
        # Without it llama.cpp runs entirely on CPU even when built with GPU
        # support, which on a 14B model is ~10× slower. On VRAM-constrained
        # GPUs (e.g. 8 GB laptops vs Sakura-14B's 8.4 GB) pass a partial count
        # so the remaining blocks run on CPU instead of OOMing at load.
        kwargs = dict(n_ctx=4096, n_gpu_layers=n_gpu_layers, verbose=False)

        target = model or DEFAULT_REPO
        if _looks_like_hf_repo(target):
            # Repo id — fetch from HF. If the repo has multiple GGUF files,
            # pick by filename; for Sakura's repo we prefer Q4_K_M.
            filename = DEFAULT_FILE if target == DEFAULT_REPO else "*.gguf"
            self.llm = Llama.from_pretrained(
                repo_id=target, filename=filename, **kwargs
            )
        else:
            self.llm = Llama(model_path=target, **kwargs)

        self.style = style
        self.glossary = glossary or []
        self.system = system or (SAKURA_SYSTEM if style == "sakura" else SYSTEM)

    def translate(self, text: str, context: list[str]) -> str:
        if self.style == "sakura":
            user = build_sakura_prompt(text, self.glossary)
            # Sakura's recommended sampling: deterministic-ish, low repetition.
            sampling = dict(temperature=0.1, top_p=0.3, repeat_penalty=1.0)
        else:
            user = build_prompt(text, context)
            sampling = dict(temperature=0.3)

        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": user},
            ],
            max_tokens=512,
            **sampling,
        )
        return out["choices"][0]["message"]["content"].strip()
