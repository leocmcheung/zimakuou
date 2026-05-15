SYSTEM = (
    "You are a professional translator specializing in Japanese anime and pop culture. "
    "Translate Japanese subtitle lines into Traditional Chinese (繁體中文 / zh-TW). "
    "Rules: output ONLY the translated line, no explanations, no quotation marks, "
    "no romaji, no Simplified characters. Preserve speaker tone (casual / formal / "
    "rude / cute) and keep honorifics natural in Chinese. If the line is onomatopoeia "
    "or untranslatable, transliterate naturally."
)


def build_prompt(text: str, context: list[str]) -> str:
    if not context:
        return f"Translate this line to Traditional Chinese:\n{text}"
    window = context[-5:]
    ctx = "\n".join(f"- {c}" for c in window)
    return (
        f"Prior cues (for context only — do NOT translate these):\n{ctx}\n\n"
        f"Translate ONLY this line to Traditional Chinese:\n{text}"
    )
