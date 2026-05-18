from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Whisper's prompt context window is ~224 tokens. JP averages ~1.5 chars/token,
# so ~200 chars is a safe budget that leaves room for special tokens.
WHISPER_PROMPT_CHAR_BUDGET = 200


@dataclass
class GlossaryEntry:
    jp: str
    zh: str
    note: str = ""
    # When False, this term is omitted from whisper's initial_prompt — use for
    # common words that don't need ASR priming but should still be enforced by
    # apply_glossary and the LLM context block.
    whisper: bool = True


@dataclass
class Context:
    title: str = ""
    title_zh_tw: str = ""
    synopsis: str = ""
    glossary: list[GlossaryEntry] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls) -> Context:
        return cls()

    @classmethod
    def load(cls, path: Path) -> Context:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        glossary = [
            GlossaryEntry(
                jp=g["jp"],
                zh=g["zh"],
                note=g.get("note", ""),
                whisper=bool(g.get("whisper", True)),
            )
            for g in (data.get("glossary") or [])
        ]
        return cls(
            title=data.get("title", "") or "",
            title_zh_tw=data.get("title_zh_tw", "") or "",
            synopsis=data.get("synopsis", "") or "",
            glossary=glossary,
            characters=list(data.get("characters") or []),
        )

    def with_sidecar(self, media_path: Path) -> Context:
        """Return a new Context with `<media_path>.description` text appended
        to synopsis, if such a sidecar file exists. Otherwise return self
        unchanged. Caller is responsible for passing a path whose
        `.with_suffix(".description")` resolves to the expected sidecar
        (e.g. a video path, or a stem with no suffix)."""
        sidecar = media_path.with_suffix(".description")
        if not sidecar.is_file():
            return self
        text = sidecar.read_text(encoding="utf-8").strip()
        if not text:
            return self
        merged = f"{self.synopsis}\n{text}".strip() if self.synopsis else text
        return Context(
            title=self.title,
            title_zh_tw=self.title_zh_tw,
            synopsis=merged,
            glossary=list(self.glossary),
            characters=list(self.characters),
        )

    def is_empty(self) -> bool:
        return not (self.title or self.synopsis or self.glossary or self.characters)

    def whisper_initial_prompt(self) -> str:
        """JP names + glossary terms + synopsis to bias whisper's decoder.
        Terms come first (highest signal-per-char); synopsis fills any
        remaining budget so per-episode `.description` content also primes
        the decoder. Returns "" if nothing useful — caller should pass
        `None` to whisper in that case."""
        terms = list(dict.fromkeys(  # dedup, preserve order
            self.characters + [g.jp for g in self.glossary if g.whisper]
        ))
        parts: list[str] = []
        if terms:
            parts.append("、".join(terms))
        if self.synopsis:
            parts.append(self.synopsis)
        if not parts:
            return ""
        return "。".join(parts)[:WHISPER_PROMPT_CHAR_BUDGET]

    def llm_context_block(self) -> str:
        """Markdown-ish context block to append to the LLM system prompt."""
        if self.is_empty():
            return ""
        lines: list[str] = []
        if self.title:
            t = f"{self.title}（{self.title_zh_tw}）" if self.title_zh_tw else self.title
            lines.append(f"作品: {t}")
        if self.synopsis:
            lines.append(f"あらすじ: {self.synopsis}")
        if self.characters:
            lines.append(f"登場人物: {'、'.join(self.characters)}")
        if self.glossary:
            lines.append("用語集 (日本語 → 繁體中文):")
            for g in self.glossary:
                line = f"  - {g.jp} → {g.zh}"
                if g.note:
                    line += f"  ({g.note})"
                lines.append(line)
        return "\n".join(lines)

    def apply_glossary(self, zh_text: str) -> str:
        """Post-pass: if the LLM left a JP glossary term untranslated in the
        Chinese output, replace it with the canonical Chinese rendering."""
        for g in self.glossary:
            if g.jp and g.jp in zh_text:
                zh_text = zh_text.replace(g.jp, g.zh)
        return zh_text
