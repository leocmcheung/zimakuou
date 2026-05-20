# Two prompt styles are supported:
#
# - "generic": instruction-tuned chat models (Qwen2.5/3, Gemma, etc.) — we
#   write our own system prompt in English specifying Traditional Chinese
#   output, then ask cue-by-cue with a sliding window of prior JP cues.
#
# - "sakura": SakuraLLM's Qwen2.5 fine-tune. Trained on a very specific
#   prompt format from SakuraLLM/utils/consts.py — deviating from it
#   measurably degrades quality. The model is trained to output Simplified
#   Chinese; the OpenCC s2twp post-pass in translate.py converts to
#   Traditional. Glossary terms go into the user prompt as a gpt_dict
#   (jp->zh, optionally `#note`) which Sakura was trained to consume.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import GlossaryEntry


SYSTEM = (
    "You are a professional translator specializing in Japanese anime and pop culture. "
    "Translate Japanese subtitle lines into Traditional Chinese (繁體中文 / zh-TW). "
    "Rules: output ONLY the translated line, no explanations, no quotation marks, "
    "no romaji, no Simplified characters. Preserve speaker tone (casual / formal / "
    "rude / cute) and keep honorifics natural in Chinese. If the line is onomatopoeia "
    "or untranslatable, transliterate naturally."
)

# Verbatim from SakuraLLM/utils/consts.py for the Sakura v1.0 release
# (Sakura-7B and Sakura-14B Qwen2.5 share this exact system prompt).
# Do not edit — the model was trained on this string.
SAKURA_SYSTEM = (
    "你是一个轻小说翻译模型，可以流畅通顺地使用给定的术语表以日本轻小说的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
)


def build_system(context_block: str = "") -> str:
    """Append a context block (synopsis / glossary / characters) to the
    base system prompt. Empty block returns SYSTEM unchanged."""
    if not context_block:
        return SYSTEM
    return f"{SYSTEM}\n\n参考情報:\n{context_block}"


def build_prompt(text: str, context: list[str]) -> str:
    if not context:
        return f"Translate this line to Traditional Chinese:\n{text}"
    window = context[-5:]
    ctx = "\n".join(f"- {c}" for c in window)
    return (
        f"Prior cues (for context only — do NOT translate these):\n{ctx}\n\n"
        f"Translate ONLY this line to Traditional Chinese:\n{text}"
    )


def _format_sakura_gpt_dict(glossary: list["GlossaryEntry"]) -> str:
    if not glossary:
        return ""
    lines: list[str] = []
    for g in glossary:
        line = f"{g.jp}->{g.zh}"
        if g.note:
            line += f" #{g.note}"
        lines.append(line)
    return "\n".join(lines)


def build_sakura_prompt(text: str, glossary: list["GlossaryEntry"] | None) -> str:
    """User-prompt template Sakura v1.0 / v0.10 was trained on."""
    gpt_dict = _format_sakura_gpt_dict(glossary or [])
    return (
        f"根据以下术语表（可以为空）：\n{gpt_dict}\n\n"
        f"将下面的日文文本根据上述术语表的对应关系和备注翻译成中文：{text}"
    )
