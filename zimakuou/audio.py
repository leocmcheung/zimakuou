import subprocess
from pathlib import Path


def extract_audio(video: Path, out_wav: Path) -> Path:
    """Extract mono 16kHz WAV — the format anime-whisper expects."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            str(out_wav),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return out_wav


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    extract_audio(args.video, args.out)
    print(f"Wrote {args.out}")
