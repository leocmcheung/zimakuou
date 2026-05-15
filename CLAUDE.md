# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

Pipeline that takes an anime video file and produces Japanese + Traditional Chinese subtitles:

```
video (mp4/mkv) → ffmpeg → audio (wav 16kHz mono)
                              ↓
                       whisper (MLX on macOS / faster-whisper CT2 on Windows)
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

- **Python**, dependencies pinned in `requirements.txt` (pip, not uv/poetry). Project venv is a pyenv-virtualenv called `zimakuou` (pinned in `.python-version`).
- **ASR is platform-split** in `zimakuou/transcribe.py`, same pattern as the translator:
  - **macOS (Apple Silicon)**: `mlx-whisper` — Metal GPU, fastest on this hardware. Default model `mlx-community/whisper-large-v3-mlx`.
  - **Windows / non-Apple**: `faster-whisper` (CTranslate2). Default `Systran/faster-whisper-large-v3`. CT2 has no Metal backend, so on Mac it'd be CPU-only — that's why we don't use it here.
  - Lazy import inside the chosen branch so the wrong-platform package never has to be installed.
  - We previously tried `litagin/anime-whisper` via `transformers` + MPS. It had two problems: (1) `chunk_length_s=30` returns one mega-chunk with `(None, None)` timestamps for seq2seq Whisper, and (2) anime-whisper hallucinated badly on BGM-heavy mixes. Both fixed by moving to MLX whisper with VAD-friendly defaults.
- **Audio extraction**: shell out to `ffmpeg` (must be on PATH). Target 16kHz mono WAV.
- **Translation LLM** is also platform-split, behind a single `Translator` interface in `zimakuou/translators/`:
  - **macOS (Apple Silicon)**: `mlx-lm` with an MLX-quantized model (default `mlx-community/Qwen2.5-7B-Instruct-4bit`).
  - **Windows**: `llama-cpp-python` loading a GGUF quant — must pass `--llm <path-to.gguf>`, no auto-download.
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

# with show-specific context (recommended — improves both ASR and translation)
python -m zimakuou path/to/video.mkv --context path/to/show.context.yaml

# stage-by-stage (each stage is independently runnable, both accept --context)
python -m zimakuou.transcribe path/to/audio.wav      # → jp.srt
python -m zimakuou.translate  path/to/file.jp.srt    # → zh-tw.srt + bilingual.srt
```

### Context YAML schema

All fields optional. Used in three places: whisper `initial_prompt` (biases ASR toward show vocab), LLM system prompt (synopsis + glossary), and a post-translation find/replace enforcing canonical glossary terms.

```yaml
title: 悟の恋
title_zh_tw: 慧的愛戀
synopsis: |
  A boy with mind-reading powers ("スペック") falls for a girl who can read his thoughts back.
characters: [悟, 美咲, ミミ, 水島]
glossary:
  - { jp: スペック,   zh: 讀心能力, note: protagonist's ability }
  - { jp: サトリマス, zh: 我懂了 }
  - { jp: ミミ,        zh: 咪咪 }
```

Whisper's `initial_prompt` budget is ~224 tokens — only the JP side of `characters` + `glossary` is sent, capped at 200 chars. The full block (synopsis + glossary table) goes into the LLM system prompt, which has no practical cap.

### Tests

```bash
pip install -r requirements-dev.txt
pytest                              # all smoke tests
pytest tests/test_translate.py -v   # one file
pytest -k bilingual                 # by name
```

The smoke suite avoids loading the real ASR / LLM models. `tests/test_translate.py` patches `get_translator` with a stub; `tests/test_audio.py` synthesizes a silent video with ffmpeg and skips itself if ffmpeg isn't installed. The full pipeline is **not** smoke-tested end-to-end — verify that manually with a short clip.

## Things that will bite

- **HF model downloads** are multi-GB and only happen on first use, then cache to `~/.cache/huggingface`.
- **Whisper hallucinations on BGM**: even with VAD enabled, dense music tracks can produce repeated phrases ("ピピピピピ") or collapsed timestamps. If you see a single cue with a wall of text and a sub-second duration, the model lost timestamp prediction.
- **MLX whisper rounds timestamps to whole seconds** by default — fine for subtitles but don't assume word-level precision.
- **No `chunk_length_s` with seq2seq Whisper** — it returns one chunk with `(None, None)` timestamps. Whisper's built-in long-form handles >30s audio.
- **MLX vs llama.cpp output drift**: same LLM weights, different quantization → slightly different translations. Don't write tests that assert exact translation strings.
- **Traditional vs Simplified**: many open models leak Simplified Chinese even when asked for Traditional. Pin this in the system prompt + post-process with OpenCC (`s2twp` config) as a safety net.
- **SRT timestamp format**: `HH:MM:SS,mmm` with a comma, not a period. Use the `srt` library — never hand-roll.
