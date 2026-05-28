import platform
import re
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import srt

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_CT2_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"

# Whisper can emit cues spanning a full conversation turn (20-30s) when speech
# is continuous. We post-split anything longer than this at the largest inter-
# word silence so the SRT is readable on screen. 12s is in the middle of the
# 10-15s sweet spot for subtitle line duration.
DEFAULT_MAX_CUE_DURATION = 10.0


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


_SENTENCE_BOUNDARY = re.compile(r"[。．！？]")


def _find_boundary_split(words: list[tuple[float, float, str]]) -> int | None:
    """Return the index of the last word that ends a sentence/clause."""
    for i in range(len(words) - 1):
        if i < len(words) - 1 and _SENTENCE_BOUNDARY.search(words[i][2]):
            return i
    return None


def _merge_short_cues(
    cues: list[tuple[float, float, str]],
    max_dur: float,
) -> list[tuple[float, float, str]]:
    if not cues:
        return cues
    result = [cues[0]]
    for start, end, text in cues[1:]:
        prev_start, prev_end, prev_text = result[-1]
        if (end - start < 1.0 or len(text) <= 2) and end - prev_start <= max_dur:
            result[-1] = (prev_start, end, prev_text + text)
        else:
            result.append((start, end, text))
    return result


def split_long_cue(
    start: float,
    end: float,
    text: str,
    words: list[tuple[float, float, str]],
    max_dur: float,
) -> list[tuple[float, float, str]]:
    """Recursively split a cue, preferring sentence/clause boundaries,
    falling back to the largest silence gap.

    `words` is a list of (start, end, text) tuples from whisper's
    word_timestamps=True output.

    Falls back to leaving the cue intact if no clean split point exists."""
    if len(words) < 2:
        return [(start, end, text)]

    punct_idx = _find_boundary_split(words)
    if punct_idx is not None:
        left_words = words[: punct_idx + 1]
        right_words = words[punct_idx + 1 :]
        left_text = "".join(w[2] for w in left_words).strip()
        right_text = "".join(w[2] for w in right_words).strip()
        return (
            split_long_cue(left_words[0][0], left_words[-1][1], left_text, left_words, max_dur)
            + split_long_cue(right_words[0][0], right_words[-1][1], right_text, right_words, max_dur)
        )

    gaps = [(words[i + 1][0] - words[i][1], i) for i in range(len(words) - 1)]
    max_gap, split_idx = max(gaps)
    if max_gap < 0.05:
        return [(start, end, text)]
    left_words = words[: split_idx + 1]
    right_words = words[split_idx + 1 :]
    left_text = "".join(w[2] for w in left_words).strip()
    right_text = "".join(w[2] for w in right_words).strip()
    return (
        split_long_cue(left_words[0][0], left_words[-1][1], left_text, left_words, max_dur)
        + split_long_cue(right_words[0][0], right_words[-1][1], right_text, right_words, max_dur)
    )


def _is_repeating_pattern(text: str) -> bool:
    """True if `text` is mostly a short unit repeated many times,
    e.g. "自動自動自動..." or "ほほほほ..."."""
    n = len(text)
    for unit_len in range(1, 5):
        unit = text[:unit_len]
        repeats = n // unit_len
        if repeats < 3:
            continue
        reconstructed = unit * repeats
        if sum(a == b for a, b in zip(text, reconstructed)) / n > 0.8:
            return True
    return False


def is_runaway(text: str, duration: float) -> bool:
    """Detect classic Whisper failure modes worth dropping:

    1. **Stuck-loop cue** — a long timestamp window with very little text
       (e.g. 30s of audio rendered as one 8-char string). Real Japanese
       speech runs ~3-15 chars/sec, so anything below ~1 char/sec over
       a multi-second window is the model getting stuck.
    2. **Repeated-character noise** — e.g. "ほほほほほ..." or "ピピピピ...".
       If a single character makes up >50% of a long string, it's noise.
    3. **Repeated n-gram pattern** — e.g. "自動自動自動..." where no single
       char exceeds 50% but a short unit tiles the entire string.
    4. **Micro-cue spam** — ≤2 chars in <0.3s, the decoder sputtering
       fragments after a stuck loop. Real speech doesn't produce these.
    """
    text = text.strip()
    if not text:
        return True
    if len(text) <= 2 and duration < 0.3:
        return True
    if duration > 5.0 and len(text) / duration < 1.0:
        return True
    if len(text) > 20:
        most_common_count = Counter(text).most_common(1)[0][1]
        if most_common_count / len(text) > 0.5:
            return True
    if len(text) > 8 and _is_repeating_pattern(text):
        return True
    return False


def transcribe(
    audio: Path,
    model_id: str | None = None,
    initial_prompt: str | None = None,
    max_cue_duration: float | None = DEFAULT_MAX_CUE_DURATION,
) -> list[srt.Subtitle]:
    """Transcribe Japanese audio. Picks MLX (Metal GPU) on Apple Silicon,
    faster-whisper (CT2) elsewhere. `initial_prompt` biases the decoder
    toward show-specific names / terms.

    `max_cue_duration` post-splits cues longer than this (seconds) at the
    largest internal silence using word-level timestamps. Pass None to
    disable splitting (also skips the word_timestamps decode pass)."""
    if _is_apple_silicon():
        segs = _transcribe_mlx(audio, model_id or DEFAULT_MLX_MODEL, initial_prompt, max_cue_duration)
    else:
        segs = _transcribe_ct2(audio, model_id or DEFAULT_CT2_MODEL, initial_prompt, max_cue_duration)

    segs = _merge_short_cues(segs, max_cue_duration or DEFAULT_MAX_CUE_DURATION)

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
    audio: Path,
    model_id: str,
    initial_prompt: str | None,
    max_cue_duration: float | None,
) -> list[tuple[float, float, str]]:
    import mlx_whisper

    # mlx-whisper has an internal tqdm frame-progress bar, but it's only
    # shown when verbose is explicitly False (verbose=None → no bar AND
    # no text; verbose=True → text segments instead of a bar).
    #
    # condition_on_previous_text=False breaks the cue-to-cue context chain
    # that lets one bad cue poison the next — the classic "ピピピピ" /
    # "ほほほほ" stuck-loop pattern. We re-enforce show vocab via the
    # glossary post-pass anyway, so we don't need whisper's auto-context.
    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=model_id,
        language="ja",
        word_timestamps=max_cue_duration is not None,
        initial_prompt=initial_prompt or None,
        condition_on_previous_text=False,
        verbose=False,
    )
    out = []
    for seg in result["segments"]:
        start, end = float(seg["start"]), float(seg["end"])
        text = seg["text"].strip()
        if is_runaway(text, end - start):
            continue
        if max_cue_duration is None:
            out.append((start, end, text))
            continue
        words = [
            (float(w["start"]), float(w["end"]), w["word"])
            for w in seg.get("words", [])
        ]
        out.extend(split_long_cue(start, end, text, words, max_cue_duration))
    return out


def _transcribe_ct2(
    audio: Path,
    model_id: str,
    initial_prompt: str | None,
    max_cue_duration: float | None,
) -> list[tuple[float, float, str]]:
    from faster_whisper import WhisperModel
    from tqdm import tqdm

    device, compute_type = _ct2_device()
    model = WhisperModel(model_id, device=device, compute_type=compute_type)
    # Tighter Silero VAD than the defaults: anime/news mixes have dense BGM
    # that the default threshold (0.5) decodes as speech. 0.6 raises the
    # bar; 700ms min silence means we trust short pauses less.
    # condition_on_previous_text=False — see _transcribe_mlx for the why.
    segments, info = model.transcribe(
        str(audio),
        language="ja",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, threshold=0.6),
        initial_prompt=initial_prompt or None,
        condition_on_previous_text=False,
        word_timestamps=max_cue_duration is not None,
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
            if is_runaway(text, end - start):
                continue
            if max_cue_duration is None:
                out.append((start, end, text))
                continue
            words = [
                (float(w.start), float(w.end), w.word) for w in (seg.words or [])
            ]
            out.extend(split_long_cue(start, end, text, words, max_cue_duration))
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
    p.add_argument(
        "--max-cue-duration",
        type=float,
        default=DEFAULT_MAX_CUE_DURATION,
        help=(
            f"Post-split cues longer than this many seconds at the largest "
            f"internal silence (default: {DEFAULT_MAX_CUE_DURATION}). "
            f"Use 0 to disable splitting."
        ),
    )
    args = p.parse_args()
    out = args.out or args.audio.with_suffix(".jp.srt")
    prompt = None
    if args.context:
        from .context import Context
        prompt = Context.load(args.context).whisper_initial_prompt() or None
    max_dur = args.max_cue_duration if args.max_cue_duration > 0 else None
    t0 = time.perf_counter()
    subs = transcribe(args.audio, args.model, initial_prompt=prompt, max_cue_duration=max_dur)
    elapsed = time.perf_counter() - t0
    write_srt(subs, out)
    print(f"Wrote {out} ({len(subs)} cues, {fmt_duration(elapsed)})")
