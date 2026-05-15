import tempfile
from pathlib import Path

from .audio import extract_audio
from .context import Context
from .srt_writer import write_bilingual, write_srt
from .transcribe import transcribe
from .translate import translate_subs


def run(
    video: Path,
    asr_model: str | None = None,
    llm_model: str | None = None,
    context_path: Path | None = None,
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
        jp_subs = transcribe(
            wav, asr_model, initial_prompt=ctx.whisper_initial_prompt() or None
        )
        write_srt(jp_subs, jp_path)
        print(f"      wrote {jp_path.name} ({len(jp_subs)} cues)")

    print(f"[3/3] translating to Traditional Chinese")
    zh_subs = translate_subs(jp_subs, llm_model, ctx=ctx)
    write_srt(zh_subs, zh_path)
    write_bilingual(jp_subs, zh_subs, bi_path)
    print(f"      wrote {zh_path.name} and {bi_path.name}")

    return jp_path, zh_path, bi_path
