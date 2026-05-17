from datetime import timedelta
from pathlib import Path

import srt
from tqdm import tqdm

from .context import Context
from .translators import get_translator
from .translators.base import Translator


def translate_subs(
    jp_subs: list[srt.Subtitle],
    llm_model: str | None = None,
    ctx: Context | None = None,
    translator: Translator | None = None,
) -> list[srt.Subtitle]:
    """Translate cue-by-cue. Each cue gets a sliding window of prior cues
    as context to keep pronouns / honorifics coherent.

    `ctx` (synopsis / characters / glossary) is threaded into the backend's
    native prompt format inside get_translator. A post-pass with OpenCC
    and a glossary substitution catches Simplified leakage and missed
    glossary terms.

    Pass `translator` to reuse a pre-built backend across multiple files
    (batch mode) — saves the multi-GB model reload between episodes."""
    from opencc import OpenCC

    ctx = ctx or Context.empty()
    if translator is None:
        translator = get_translator(llm_model, ctx=ctx)
    # Safety net: many models leak Simplified characters even when asked for
    # Traditional, and Sakura is *trained* to emit Simplified — OpenCC s2twp
    # normalises everything to Traditional (Taiwan) with phrase mapping.
    cc = OpenCC("s2twp")

    out: list[srt.Subtitle] = []
    history: list[str] = []
    for sub in tqdm(jp_subs, desc="translating", unit="cue", dynamic_ncols=True):
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
    import time
    from ._timing import fmt_duration
    from .audit import write_glossary_draft
    from .srt_writer import write_bilingual, write_srt

    p = argparse.ArgumentParser()
    p.add_argument("jp_srt", type=Path, help="Japanese SRT to translate")
    p.add_argument("--llm", default=None)
    p.add_argument("--context", type=Path, default=None, help="Context YAML")
    args = p.parse_args()

    jp_subs = list(srt.parse(args.jp_srt.read_text(encoding="utf-8")))
    ctx = Context.load(args.context) if args.context else Context.empty()
    t0 = time.perf_counter()
    zh_subs = translate_subs(jp_subs, args.llm, ctx=ctx)
    elapsed = time.perf_counter() - t0

    stem = args.jp_srt.with_suffix("").with_suffix("")  # strip .jp.srt
    zh_path = Path(f"{stem}.zh-tw.srt")
    bi_path = Path(f"{stem}.bilingual.srt")
    write_srt(zh_subs, zh_path)
    write_bilingual(jp_subs, zh_subs, bi_path)
    print(f"Wrote {zh_path} and {bi_path} ({fmt_duration(elapsed)})")

    draft_path = Path(f"{stem}.context.draft.yaml")
    if write_glossary_draft(jp_subs, draft_path, ctx):
        print(
            f"[audit] drafted glossary candidates → {draft_path.name}\n"
            f"        fill in zh:, rename to .context.yaml, re-run with "
            f"--context to improve terminology."
        )
