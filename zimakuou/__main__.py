import argparse
from pathlib import Path

from .pipeline import run
from .transcribe import DEFAULT_MAX_CUE_DURATION


def main():
    parser = argparse.ArgumentParser(
        prog="zimakuou",
        description="Extract Japanese audio from anime video and translate subtitles to Traditional Chinese.",
    )
    parser.add_argument("video", type=Path, help="Input video file (mp4/mkv)")
    parser.add_argument(
        "--asr-model",
        default=None,
        help="ASR model id (defaults to MLX whisper-large-v3 on Mac, CT2 on Windows)",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help=(
            "Translator: GGUF file path, HF GGUF repo id, or an MLX repo id. "
            "Defaults to SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF (auto-downloads)."
        ),
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=None,
        help="Optional YAML with synopsis, characters, glossary (see CLAUDE.md)",
    )
    parser.add_argument(
        "--max-cue-duration",
        type=float,
        default=DEFAULT_MAX_CUE_DURATION,
        help=(
            f"Post-split cues longer than this many seconds at the largest "
            f"internal silence (default: {DEFAULT_MAX_CUE_DURATION}). "
            f"Use 0 to disable splitting."
        ),
    )
    args = parser.parse_args()
    run(
        args.video,
        asr_model=args.asr_model,
        llm_model=args.llm,
        context_path=args.context,
        max_cue_duration=args.max_cue_duration if args.max_cue_duration > 0 else None,
    )


if __name__ == "__main__":
    main()
