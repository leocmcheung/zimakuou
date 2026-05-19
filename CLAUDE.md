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
  - **Optional Parakeet backend (CUDA only)**: pass `--asr-model nvidia/parakeet-tdt_ctc-0.6b-ja` (or any `nvidia/parakeet*` repo) to route through NVIDIA NeMo instead of faster-whisper. Transducer models tend to skip non-speech rather than hallucinate over it, so this can help with BGM-dense mixes. NeMo (~1-2 GB of deps) is bundled on Windows via `requirements.txt`; install manually on Linux/WSL with `pip install nemo_toolkit[asr]`. Parakeet has no inference-time text prior, so `initial_prompt` biasing is silently dropped (glossary is still enforced post-translation). Refused on Apple Silicon — no Metal/MLX path.
  - We previously tried `litagin/anime-whisper` via `transformers` + MPS. It had two problems: (1) `chunk_length_s=30` returns one mega-chunk with `(None, None)` timestamps for seq2seq Whisper, and (2) anime-whisper hallucinated badly on BGM-heavy mixes. Both fixed by moving to MLX whisper with VAD-friendly defaults.
- **Audio extraction**: shell out to `ffmpeg` (must be on PATH). Target 16kHz mono WAV.
- **Translation LLM** is the [SakuraLLM Sakura-14B-Qwen2.5-v1.0](https://huggingface.co/SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF) GGUF (Q4_K_M, ~8.4 GB), served via `llama-cpp-python` with Metal on macOS and CUDA/CPU on Windows. Sakura is fine-tuned on light-novel / galge / anime JP→ZH and uses a specific Chinese prompt format with a `gpt_dict` glossary block in the user message — that wiring lives in `zimakuou/translators/_prompt.py` (`SAKURA_SYSTEM`, `build_sakura_prompt`). Sakura is trained to emit Simplified Chinese; the OpenCC `s2twp` post-pass in `translate.py` normalises to Traditional, and `Context.apply_glossary` enforces canonical glossary terms after that.
  - Backend selection lives in `zimakuou/translators/__init__.py`. If `--llm` points at a GGUF (file path or HF GGUF repo) or no model is given (default), it uses `LlamaTranslator`. Otherwise on Apple Silicon it falls back to `MLXTranslator` (e.g. `--llm mlx-community/Qwen3-8B-4bit` if you want to A/B). Lazy imports inside each branch, so the wrong-platform package never has to install.
  - Style auto-detects from the model name: anything containing `sakura` (case-insensitive) is routed to Sakura's trained prompt; anything else uses the generic English system prompt.
  - The Sakura GGUF auto-downloads via `Llama.from_pretrained` on first use and caches to `~/.cache/huggingface` (same as MLX models). Pass `--llm <path-to.gguf>` to use a local file instead.
- **Translation prompt**: translate Japanese → Traditional Chinese (zh-TW / 繁體中文, *not* Simplified). Translate cue-by-cue but pass a sliding window of prior cues as context so pronouns and honorifics stay coherent. Preserve cue indices and timestamps verbatim — the LLM must never reorder or merge cues. For Sakura, sliding-window history is *not* passed to the model (it isn't trained on it); coherence comes from the glossary + s2twp + post-replace passes.
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

# batch mode for many videos, designed for NAS-hosted sources that may
# disconnect: phase 1 extracts ALL audio up front (NAS-dependent), phase 2
# transcribes each .wav locally then deletes it, phase 3 loads the LLM ONCE
# and translates every .jp.srt. Outputs land in cwd. Resumable — each phase
# skips files whose outputs already exist.
cd ~/some/local/work-dir
python -m zimakuou.batch \
    /Volumes/NAS/show/ep01.mp4 /Volumes/NAS/show/ep02.mp4 ... \
    --context /Volumes/NAS/show/show.context.yaml

# merge per-episode draft glossaries into a show-wide master (keeps terms
# with ≥3 total mentions across all drafts; preserves existing comments)
python -m zimakuou.merge_drafts /path/to/show-folder
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
  - { jp: 食材,        zh: 食材, whisper: false }   # post-pass only, no ASR bias
```

Whisper's `initial_prompt` budget is ~224 tokens — JP `characters` + glossary entries (those with default `whisper: true`) + a synopsis tail are sent, capped at 200 chars. The full glossary (regardless of `whisper:`) goes into the LLM system prompt and the post-translation `apply_glossary` pass; the flag only filters the ASR prompt.

If a `<video>.description` sidecar sits next to the input video, its text is auto-merged into `synopsis` — use this for per-episode plot/recipe details while keeping the YAML show-wide.

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
