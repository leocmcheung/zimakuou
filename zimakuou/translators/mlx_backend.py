from .base import Translator
from ._prompt import SYSTEM, build_prompt

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


class MLXTranslator(Translator):
    def __init__(self, model: str | None = None):
        from mlx_lm import generate, load

        self._generate = generate
        self.model, self.tokenizer = load(model or DEFAULT_MODEL)

    def translate(self, text: str, context: list[str]) -> str:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(text, context)},
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
