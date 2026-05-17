import tempfile
import time
from pathlib import Path

from ._timing import fmt_duration
from .audio import extract_audio
from .audit import write_glossary_draft
from .context import Context
from .srt_writer import write_bilingual, write_srt
from .transcribe import DEFAULT_MAX_CUE_DURATION, transcribe
from .translate import translate_subs


def run(
    video: Path,
    asr_model: str | None = None,
    llm_model: str | None = None,
    context_path: Path | None = None,
    max_cue_duration: float | None = DEFAULT_MAX_CUE_DURATION,
) -> tuple[Path, Path, Path]:
    stem = video.with_suffix("")
    jp_path = Path(f"{stem}.jp.srt")
    zh_path = Path(f"{stem}.zh-tw.srt")
    bi_path = Path(f"{stem}.bilingual.srt")

    ctx = Context.load(context_path) if context_path else Context.empty()
    if not ctx.is_empty():
        print(f"[ctx] loaded {context_path.name}: "
              f"{len(ctx.characters)} characters, {len(ctx.glossary)} glossary entries")

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        print(f"[1/3] extracting audio → {wav.name}")
        extract_audio(video, wav)

        print(f"[2/3] transcribing (asr_model={asr_model or 'default'})")
        t0 = time.perf_counter()
        jp_subs = transcribe(
            wav,
            asr_model,
            initial_prompt=ctx.whisper_initial_prompt() or None,
            max_cue_duration=max_cue_duration,
        )
        write_srt(jp_subs, jp_path)
        print(
            f"      wrote {jp_path.name} ({len(jp_subs)} cues, "
            f"{fmt_duration(time.perf_counter() - t0)})"
        )

    print(f"[3/3] translating to Traditional Chinese")
    t0 = time.perf_counter()
    zh_subs = translate_subs(jp_subs, llm_model, ctx=ctx)
    write_srt(zh_subs, zh_path)
    write_bilingual(jp_subs, zh_subs, bi_path)
    print(
        f"      wrote {zh_path.name} and {bi_path.name} "
        f"({fmt_duration(time.perf_counter() - t0)})"
    )

    draft_path = Path(f"{stem}.context.draft.yaml")
    if write_glossary_draft(jp_subs, draft_path, ctx):
        print(
            f"[audit] drafted glossary candidates → {draft_path.name}\n"
            f"        fill in zh:, rename to .context.yaml, re-run translate "
            f"with --context to improve terminology."
        )

    return jp_path, zh_path, bi_path
