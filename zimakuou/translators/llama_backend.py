from .base import Translator
from ._prompt import SYSTEM, build_prompt


class LlamaTranslator(Translator):
    def __init__(self, model: str | None = None, system: str | None = None):
        from llama_cpp import Llama

        if not model:
            raise ValueError(
                "llama.cpp backend needs a GGUF model path — pass --llm <path-to.gguf>"
            )
        self.llm = Llama(model_path=model, n_ctx=4096, verbose=False)
        self.system = system or SYSTEM

    def translate(self, text: str, context: list[str]) -> str:
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": build_prompt(text, context)},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        return out["choices"][0]["message"]["content"].strip()
