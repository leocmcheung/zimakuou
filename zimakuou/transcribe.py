import platform
import sys
from datetime import timedelta
from pathlib import Path

import srt

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_CT2_MODEL = "Systran/faster-whisper-large-v3"


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


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

    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=model_id,
        language="ja",
        word_timestamps=False,
        initial_prompt=initial_prompt or None,
    )
    out = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if text:
            out.append((float(seg["start"]), float(seg["end"]), text))
    return out


def _transcribe_ct2(
    audio: Path, model_id: str, initial_prompt: str | None
) -> list[tuple[float, float, str]]:
    from faster_whisper import WhisperModel

    device, compute_type = _ct2_device()
    model = WhisperModel(model_id, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(audio),
        language="ja",
        beam_size=5,
        vad_filter=True,
        initial_prompt=initial_prompt or None,
    )
    out = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            out.append((float(seg.start), float(seg.end), text))
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
    subs = transcribe(args.audio, args.model, initial_prompt=prompt)
    write_srt(subs, out)
    print(f"Wrote {out} ({len(subs)} cues)")
