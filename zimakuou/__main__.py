import argparse
from pathlib import Path

from .pipeline import run


def main():
    parser = argparse.ArgumentParser(
        prog="zimakuou",
        description="Extract Japanese audio from anime video and translate subtitles to Traditional Chinese.",
    )
    parser.add_argument("video", type=Path, help="Input video file (mp4/mkv)")
    parser.add_argument(
        "--asr-model",
        default="litagin/anime-whisper",
        help="HuggingFace ASR model id",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help="LLM model id (MLX) or GGUF path (llama.cpp). Defaults per platform.",
    )
    args = parser.parse_args()
    run(args.video, asr_model=args.asr_model, llm_model=args.llm)


if __name__ == "__main__":
    main()
