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
            "Defaults to SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF (auto-downloads). "
            "Pass SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF for higher quality at "
            "~4-5× slower throughput on VRAM-constrained GPUs."
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
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help=(
            "How many transformer blocks to offload to GPU (llama.cpp/GGUF only; "
            "ignored by MLX). -1 = all (default) — fits the default Sakura-7B "
            "IQ4_XS on 8 GB VRAM. Drop to e.g. 35 if you switch --llm to "
            "Sakura-14B Q4_K_M (~8.4 GB) on a VRAM-constrained card."
        ),
    )
    args = parser.parse_args()
    run(
        args.video,
        asr_model=args.asr_model,
        llm_model=args.llm,
        context_path=args.context,
        max_cue_duration=args.max_cue_duration if args.max_cue_duration > 0 else None,
        n_gpu_layers=args.n_gpu_layers,
    )


if __name__ == "__main__":
    main()
