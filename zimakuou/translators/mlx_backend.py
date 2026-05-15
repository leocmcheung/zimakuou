from __future__ import annotations

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

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


class MLXTranslator(Translator):
    def __init__(
        self,
        model: str | None = None,
        system: str | None = None,
        style: str = "generic",
        glossary: list["GlossaryEntry"] | None = None,
    ):
        from mlx_lm import generate, load

        self._generate = generate
        self.model, self.tokenizer = load(model or DEFAULT_MODEL)
        self.style = style
        self.glossary = glossary or []
        self.system = system or (SAKURA_SYSTEM if style == "sakura" else SYSTEM)

    def translate(self, text: str, context: list[str]) -> str:
        if self.style == "sakura":
            user = build_sakura_prompt(text, self.glossary)
        else:
            user = build_prompt(text, context)

        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        out = self._generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=False,
        )
        return out.strip()
