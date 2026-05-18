"""Disconnect-resilient batch pipeline for many videos.

Designed for the case where the source videos live on a NAS that may
disconnect after ~an hour. We split the work into three phases:

  Phase 1 — extract: ffmpeg all videos to .wav in the local cwd. This is
            the only phase that touches the NAS. Get it done up front
            before the share drops.
  Phase 2 — transcribe: whisper each .wav → .jp.srt locally, then delete
            the .wav. NAS-independent. Resumable.
  Phase 3 — translate: load the LLM once, then translate each .jp.srt →
            .zh-tw.srt + .bilingual.srt locally. NAS-independent.

Outputs all land in the current working directory next to the cwd, named
after the video stem. Subtitle files are saved locally — move them to
the NAS manually when the run is done.

Resumable: each phase skips a file when its output already exists, so a
killed run can be restarted with the same command. Pass --force to redo
everything.

Run:
    cd ~/some/local/work-dir
    python -m zimakuou.batch \\
        /Volumes/NAS/show/ep01.mp4 /Volumes/NAS/show/ep02.mp4 ... \\
        --context /Volumes/NAS/show/show.context.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import srt

from ._timing import fmt_duration
from .audio import extract_audio
from .audit import write_glossary_draft
from .context import Context
from .srt_writer import write_bilingual, write_srt
from .transcribe import DEFAULT_MAX_CUE_DURATION, transcribe
from .translate import translate_subs
from .translators import get_translator


def _local_paths(video: Path, out_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Map a (possibly-NAS) video path to its local artifacts."""
    stem = video.stem  # filename without extension, NOT path
    return (
        out_dir / f"{stem}.wav",
        out_dir / f"{stem}.jp.srt",
        out_dir / f"{stem}.zh-tw.srt",
        out_dir / f"{stem}.bilingual.srt",
        out_dir / f"{stem}.context.draft.yaml",
    )


# Subtitle suffixes the pipeline produces. If any of these exists for a given
# video stem we treat the video as "already processed" and skip it in batch
# runs unless --force is passed.
_PIPELINE_SUB_SUFFIXES = (".jp.srt", ".zh-tw.srt", ".bilingual.srt")


def _existing_subtitle(video: Path, out_dir: Path) -> Path | None:
    """Return the first pipeline subtitle file we find for `video`, or None.
    Checks two locations: (1) the local out_dir (previous batch run in the
    same cwd) and (2) next to the video on whatever storage it lives on,
    so a video whose .srt files were already moved back to the NAS is
    recognised on the next batch run."""
    stem = video.stem
    for parent in (out_dir, video.parent):
        for suffix in _PIPELINE_SUB_SUFFIXES:
            candidate = parent / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _partition_by_existing_subs(
    videos: list[Path], out_dir: Path
) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """Split `videos` into (to_process, already_done) where the second
    list pairs each skipped video with the existing subtitle that
    triggered the skip."""
    to_process: list[Path] = []
    already_done: list[tuple[Path, Path]] = []
    for v in videos:
        existing = _existing_subtitle(v, out_dir)
        if existing is None:
            to_process.append(v)
        else:
            already_done.append((v, existing))
    return to_process, already_done


def _extract_all(videos: list[Path], out_dir: Path, force: bool) -> None:
    print(f"[1/3] extracting audio for {len(videos)} videos → {out_dir}")
    for i, video in enumerate(videos, start=1):
        wav, *_ = _local_paths(video, out_dir)
        if wav.exists() and not force:
            print(f"  [{i}/{len(videos)}] {video.name} → {wav.name} (skip, exists)")
            continue
        if not video.exists():
            print(f"  [{i}/{len(videos)}] {video.name} → MISSING, skipping", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        print(f"  [{i}/{len(videos)}] {video.name} → {wav.name}")
        extract_audio(video, wav)
        print(f"        done ({fmt_duration(time.perf_counter() - t0)})")


def _transcribe_all(
    videos: list[Path],
    out_dir: Path,
    asr_model: str | None,
    ctx: Context,
    force: bool,
    max_cue_duration: float | None = DEFAULT_MAX_CUE_DURATION,
) -> None:
    print(f"[2/3] transcribing (asr_model={asr_model or 'default'})")
    for i, video in enumerate(videos, start=1):
        wav, jp_srt, *_ = _local_paths(video, out_dir)
        if jp_srt.exists() and not force:
            print(f"  [{i}/{len(videos)}] {jp_srt.name} (skip, exists)")
            # Cleanup any stale wav left from a previous run.
            if wav.exists():
                wav.unlink()
            continue
        if not wav.exists():
            print(
                f"  [{i}/{len(videos)}] {wav.name} missing — re-run phase 1 first",
                file=sys.stderr,
            )
            continue
        # Per-video ctx: merge the show-wide YAML with this episode's
        # `<video>.description` sidecar if one exists.
        per_ctx = ctx.with_sidecar(video)
        initial_prompt = per_ctx.whisper_initial_prompt() or None
        t0 = time.perf_counter()
        print(f"  [{i}/{len(videos)}] {wav.name} → {jp_srt.name}")
        jp_subs = transcribe(
            wav, asr_model, initial_prompt=initial_prompt, max_cue_duration=max_cue_duration
        )
        write_srt(jp_subs, jp_srt)
        print(
            f"        wrote {jp_srt.name} ({len(jp_subs)} cues, "
            f"{fmt_duration(time.perf_counter() - t0)})"
        )
        # Delete the wav once we have the jp.srt — these are ~80 MB/episode.
        wav.unlink()


def _translate_all(
    videos: list[Path],
    out_dir: Path,
    llm_model: str | None,
    ctx: Context,
    force: bool,
    n_gpu_layers: int = -1,
) -> None:
    print(f"[3/3] translating to Traditional Chinese")
    # Build the translator once — Sakura is 8.4 GB to load; doing it per
    # episode would cost ~30s × N episodes in pointless reloads.
    translator = None
    for i, video in enumerate(videos, start=1):
        _wav, jp_srt, zh_srt, bi_srt, draft = _local_paths(video, out_dir)
        if zh_srt.exists() and bi_srt.exists() and not force:
            print(f"  [{i}/{len(videos)}] {zh_srt.name} (skip, exists)")
            continue
        if not jp_srt.exists():
            print(
                f"  [{i}/{len(videos)}] {jp_srt.name} missing — re-run phase 2 first",
                file=sys.stderr,
            )
            continue
        # Per-video ctx so per-episode `.description` flows into the
        # glossary post-pass and (for non-Sakura backends) any future
        # synopsis-aware behaviour. The cached translator's system prompt
        # is fixed at construction — fine for Sakura (no synopsis used)
        # and an acceptable trade-off vs reloading 8 GB per episode.
        per_ctx = ctx.with_sidecar(video)
        if translator is None:
            print(f"        loading translator (one-time)…")
            t_load = time.perf_counter()
            translator = get_translator(llm_model, ctx=per_ctx, n_gpu_layers=n_gpu_layers)
            print(f"        ready ({fmt_duration(time.perf_counter() - t_load)})")

        jp_subs = list(srt.parse(jp_srt.read_text(encoding="utf-8")))
        t0 = time.perf_counter()
        print(f"  [{i}/{len(videos)}] {jp_srt.name} → {zh_srt.name}")
        zh_subs = translate_subs(jp_subs, ctx=per_ctx, translator=translator)
        write_srt(zh_subs, zh_srt)
        write_bilingual(jp_subs, zh_subs, bi_srt)
        print(
            f"        wrote {zh_srt.name} + {bi_srt.name} "
            f"({fmt_duration(time.perf_counter() - t0)})"
        )
        if write_glossary_draft(jp_subs, draft, per_ctx):
            print(f"        [audit] drafted glossary candidates → {draft.name}")


def run_batch(
    videos: list[Path],
    out_dir: Path,
    asr_model: str | None = None,
    llm_model: str | None = None,
    ctx: Context | None = None,
    force: bool = False,
    max_cue_duration: float | None = DEFAULT_MAX_CUE_DURATION,
    n_gpu_layers: int = -1,
) -> None:
    ctx = ctx or Context.empty()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight: skip videos that already have pipeline subtitles, either
    # locally (previous batch) or next to the video (e.g. files moved back
    # to the NAS after an earlier run). Saves NAS bandwidth in phase 1.
    if force:
        to_process = videos
    else:
        to_process, already_done = _partition_by_existing_subs(videos, out_dir)
        if already_done:
            print(
                f"[batch] {len(already_done)} of {len(videos)} videos already have "
                f"subtitles, skipping (--force to override):"
            )
            for video, existing in already_done:
                print(f"  - {video.name}  ← {existing.name}")
        if not to_process:
            print(f"[batch] nothing to do.")
            return
        print(f"[batch] processing {len(to_process)} videos")

    _extract_all(to_process, out_dir, force)
    _transcribe_all(to_process, out_dir, asr_model, ctx, force, max_cue_duration)
    _translate_all(to_process, out_dir, llm_model, ctx, force, n_gpu_layers=n_gpu_layers)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="zimakuou.batch",
        description="Disconnect-resilient batch pipeline. Extracts audio for "
                    "all videos up front (NAS phase), then transcribes and "
                    "translates locally. Outputs land in the cwd.",
    )
    parser.add_argument("videos", type=Path, nargs="+", help="Video files (mp4/mkv)")
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--llm", default=None)
    parser.add_argument(
        "--context", type=Path, default=None, help="Context YAML (see CLAUDE.md)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.cwd(),
        help="Where to write .wav / .srt files (default: cwd)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redo all phases even when output files already exist",
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
            "ignored by MLX). -1 = all (default). Drop to a partial count on "
            "VRAM-constrained GPUs, e.g. 35 for Sakura-14B Q4_K_M on an 8 GB card."
        ),
    )
    args = parser.parse_args()

    ctx = Context.load(args.context) if args.context else Context.empty()
    if not ctx.is_empty():
        print(
            f"[ctx] loaded {args.context.name}: "
            f"{len(ctx.characters)} characters, {len(ctx.glossary)} glossary entries"
        )

    run_batch(
        args.videos,
        out_dir=args.out_dir,
        asr_model=args.asr_model,
        llm_model=args.llm,
        ctx=ctx,
        force=args.force,
        max_cue_duration=args.max_cue_duration if args.max_cue_duration > 0 else None,
        n_gpu_layers=args.n_gpu_layers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
