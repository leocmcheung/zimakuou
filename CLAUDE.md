# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Greenfield. The repo is currently empty; this document describes the intended architecture so the first implementation lands consistently.

## Goal

Pipeline that takes an anime video file and produces Japanese + Traditional Chinese subtitles:

```
video (mp4/mkv) → ffmpeg → audio (wav 16kHz mono)
                              ↓
                       anime-whisper (HF: litagin/anime-whisper)
                              ↓
                       Japanese SRT cues
                              ↓
                       local LLM (MLX on macOS / llama.cpp on Windows)
                              ↓
                       Traditional Chinese SRT cues
                              ↓
            jp.srt  +  zh-tw.srt  +  bilingual.srt
```

## Stack & conventions

- **Python**, dependencies pinned in `requirements.txt` (pip, not uv/poetry).
- **ASR model**: `litagin/anime-whisper` from Hugging Face — a whisper-large-v3 finetune on anime dialogue. Load via `transformers` `pipeline("automatic-speech-recognition", ...)` with `return_timestamps=True` so SRT cues get accurate boundaries.
- **Audio extraction**: shell out to `ffmpeg` (must be on PATH). Target 16kHz mono WAV — anime-whisper expects whisper-format input.
- **Translation LLM** is platform-split — keep this behind a single `Translator` interface so the pipeline doesn't branch:
  - **macOS (Apple Silicon)**: `mlx-lm` with an MLX-quantized model (e.g. a Qwen2.5 or similar JP→ZH-capable instruct model from `mlx-community`).
  - **Windows**: `llama-cpp-python` loading a GGUF quant of the same model family.
  - Detect platform at startup (`sys.platform` + `platform.machine()`) and instantiate the right backend. Don't import both unconditionally — both have heavy native deps that fail to install on the wrong OS.
- **Translation prompt**: translate Japanese → Traditional Chinese (zh-TW / 繁體中文, *not* Simplified). Translate cue-by-cue but pass a sliding window of prior cues as context so pronouns and honorifics stay coherent. Preserve cue indices and timestamps verbatim — the LLM must never reorder or merge cues.
- **Output**: emit three files alongside the input video:
  - `<name>.jp.srt` — raw whisper output
  - `<name>.zh-tw.srt` — translation only
  - `<name>.bilingual.srt` — Japanese line on top, Traditional Chinese below, same cue timing

## Commands

```bash
# install (also ensure ffmpeg is on PATH: `brew install ffmpeg` / winget)
pip install -r requirements.txt

# run end-to-end
python -m zimakuou path/to/video.mkv

# stage-by-stage (each stage is independently runnable)
python -m zimakuou.transcribe path/to/audio.wav      # → jp.srt
python -m zimakuou.translate  path/to/file.jp.srt    # → zh-tw.srt + bilingual.srt
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest                              # all smoke tests
pytest tests/test_translate.py -v   # one file
pytest -k bilingual                 # by name
```

The smoke suite avoids loading the real ASR / LLM models. `tests/test_translate.py` patches `get_translator` with a stub; `tests/test_audio.py` synthesizes a silent video with ffmpeg and skips itself if ffmpeg isn't installed. The full pipeline is **not** smoke-tested end-to-end — verify that manually with a short clip.

## Things that will bite

- **Hugging Face model download**: `litagin/anime-whisper` is multi-GB. Cache it (`HF_HOME` / `~/.cache/huggingface`) and don't re-download on every run.
- **Whisper hallucinations**: anime-whisper is better than vanilla whisper on anime but still hallucinates on silence/music. Filter cues with suspiciously long durations or repeated tokens before translating.
- **MLX vs llama.cpp output drift**: same model, different quantization → slightly different translations. Acceptable, but don't write tests that assert exact translation strings.
- **SRT timestamp format**: `HH:MM:SS,mmm` with a comma, not a period. Use a library (`srt`, `pysrt`) rather than hand-rolling.
- **Traditional vs Simplified**: many open models default to Simplified Chinese even when asked for Traditional. Pin this in the system prompt and consider post-processing with OpenCC (`opencc-python-reimplemented`, `s2twp.json` config) as a safety net.
