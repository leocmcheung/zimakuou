from datetime import timedelta
from pathlib import Path

import srt

from .context import Context
from .translators import get_translator
from .translators._prompt import build_system


def translate_subs(
    jp_subs: list[srt.Subtitle],
    llm_model: str | None = None,
    ctx: Context | None = None,
) -> list[srt.Subtitle]:
    """Translate cue-by-cue. Each cue gets a sliding window of prior cues
    as context to keep pronouns / honorifics coherent.

    `ctx` injects show-specific synopsis / characters / glossary into the
    LLM system prompt and runs a post-pass enforcing canonical glossary
    translations."""
    from opencc import OpenCC

    ctx = ctx or Context.empty()
    system = build_system(ctx.llm_context_block())
    translator = get_translator(llm_model, system=system)
    # Safety net: many models leak Simplified characters even when asked for
    # Traditional. s2twp = Simplified → Traditional (Taiwan) with phrase mapping.
    cc = OpenCC("s2twp")

    out: list[srt.Subtitle] = []
    history: list[str] = []
    for sub in jp_subs:
        zh = translator.translate(sub.content, history)
        zh = cc.convert(zh)
        zh = ctx.apply_glossary(zh)
        out.append(
            srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=zh)
        )
        history.append(sub.content)
    return out


if __name__ == "__main__":
    import argparse
    from .srt_writer import write_bilingual, write_srt

    p = argparse.ArgumentParser()
    p.add_argument("jp_srt", type=Path, help="Japanese SRT to translate")
    p.add_argument("--llm", default=None)
    p.add_argument("--context", type=Path, default=None, help="Context YAML")
    args = p.parse_args()

    jp_subs = list(srt.parse(args.jp_srt.read_text(encoding="utf-8")))
    ctx = Context.load(args.context) if args.context else None
    zh_subs = translate_subs(jp_subs, args.llm, ctx=ctx)

    stem = args.jp_srt.with_suffix("").with_suffix("")  # strip .jp.srt
    zh_path = Path(f"{stem}.zh-tw.srt")
    bi_path = Path(f"{stem}.bilingual.srt")
    write_srt(zh_subs, zh_path)
    write_bilingual(jp_subs, zh_subs, bi_path)
    print(f"Wrote {zh_path} and {bi_path}")
