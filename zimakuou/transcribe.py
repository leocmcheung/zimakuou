import platform
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import srt

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_CT2_MODEL = "Systran/faster-whisper-large-v3"


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def is_runaway(text: str, duration: float) -> bool:
    """Detect classic Whisper failure modes worth dropping:

    1. **Stuck-loop cue** — a long timestamp window with very little text
       (e.g. 30s of audio rendered as one 8-char string). Real Japanese
       speech runs ~3-15 chars/sec, so anything below ~1 char/sec over
       a multi-second window is the model getting stuck.
    2. **Repeated-character noise** — e.g. "ほほほほほ..." or "ピピピピ...".
       If a single character makes up >50% of a long string, it's noise.
    """
    text = text.strip()
    if not text:
        return True
    if duration > 5.0 and len(text) / duration < 1.0:
        return True
    if len(text) > 20:
        most_common_count = Counter(text).most_common(1)[0][1]
        if most_common_count / len(text) > 0.5:
            return True
    return False


def transcribe(
    audio: Path,
    model_id: str | None = None,
    initial_prompt: str | None = None,
) -> list[srt.Subtitle]:
    """Transcribe Japanese audio. Picks MLX (Metal GPU) on Apple Silicon,
    faster-whisper (CT2) elsewhere. `initial_prompt` biases the decoder
    toward show-specific names / terms."""
    if _is_apple_silicon():
        segs = _transcribe_mlx(audio, model_id or DEFAULT_MLX_MODEL, initial_prompt)
    else:
        segs = _transcribe_ct2(audio, model_id or DEFAULT_CT2_MODEL, initial_prompt)

    return [
        srt.Subtitle(
            index=i,
            start=timedelta(seconds=start),
            end=timedelta(seconds=end),
            content=text,
        )
        for i, (start, end, text) in enumerate(segs, start=1)
    ]


def _transcribe_mlx(
    audio: Path, model_id: str, initial_prompt: str | None
) -> list[tuple[float, float, str]]:
    import mlx_whisper

    # mlx-whisper has an internal tqdm frame-progress bar, but it's only
    # shown when verbose is explicitly False (verbose=None → no bar AND
    # no text; verbose=True → text segments instead of a bar).
    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=model_id,
        language="ja",
        word_timestamps=False,
        initial_prompt=initial_prompt or None,
        verbose=False,
    )
    out = []
    for seg in result["segments"]:
        start, end = float(seg["start"]), float(seg["end"])
        text = seg["text"].strip()
        if not is_runaway(text, end - start):
            out.append((start, end, text))
    return out


def _transcribe_ct2(
    audio: Path, model_id: str, initial_prompt: str | None
) -> list[tuple[float, float, str]]:
    from faster_whisper import WhisperModel
    from tqdm import tqdm

    device, compute_type = _ct2_device()
    model = WhisperModel(model_id, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio),
        language="ja",
        beam_size=5,
        vad_filter=True,
        initial_prompt=initial_prompt or None,
    )
    # faster-whisper returns a generator over segments. Drive it through
    # tqdm with the audio duration as the total so the bar reports how
    # much of the file has been decoded so far (not segment count, which
    # is unknown until the generator is exhausted).
    out = []
    with tqdm(total=round(info.duration), unit="s", desc="transcribing", dynamic_ncols=True) as pbar:
        last_end = 0.0
        for seg in segments:
            start, end = float(seg.start), float(seg.end)
            pbar.update(round(end - last_end))
            last_end = end
            text = seg.text.strip()
            if not is_runaway(text, end - start):
                out.append((start, end, text))
        # Snap to 100% in case the last segment ends slightly before info.duration.
        pbar.update(max(0, round(info.duration) - round(last_end)))
    return out


def _ct2_device() -> tuple[str, str]:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


if __name__ == "__main__":
    import argparse
    import time
    from ._timing import fmt_duration
    from .srt_writer import write_srt

    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path)
    p.add_argument("--model", default=None, help="MLX or CT2 HF repo / local path")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--context", type=Path, default=None, help="Context YAML for initial_prompt")
    args = p.parse_args()
    out = args.out or args.audio.with_suffix(".jp.srt")
    prompt = None
    if args.context:
        from .context import Context
        prompt = Context.load(args.context).whisper_initial_prompt() or None
    t0 = time.perf_counter()
    subs = transcribe(args.audio, args.model, initial_prompt=prompt)
    elapsed = time.perf_counter() - t0
    write_srt(subs, out)
    print(f"Wrote {out} ({len(subs)} cues, {fmt_duration(elapsed)})")
