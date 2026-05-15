from datetime import timedelta
from pathlib import Path

import srt


def transcribe(audio: Path, model_id: str = "litagin/anime-whisper") -> list[srt.Subtitle]:
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=30,
        return_timestamps=True,
    )
    result = pipe(str(audio), return_timestamps=True, generate_kwargs={"language": "ja"})

    subs: list[srt.Subtitle] = []
    for i, chunk in enumerate(result["chunks"], start=1):
        start, end = chunk["timestamp"]
        if start is None or end is None:
            continue
        text = chunk["text"].strip()
        if not text:
            continue
        subs.append(
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=start),
                end=timedelta(seconds=end),
                content=text,
            )
        )
    return subs


if __name__ == "__main__":
    import argparse
    from .srt_writer import write_srt

    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path)
    p.add_argument("--model", default="litagin/anime-whisper")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or args.audio.with_suffix(".jp.srt")
    subs = transcribe(args.audio, args.model)
    write_srt(subs, out)
    print(f"Wrote {out} ({len(subs)} cues)")
