"""Merge per-episode `<stem>.context.draft.yaml` files into a show-wide master.

For a show with many episodes, every translate run drops a draft glossary
next to the video. This script aggregates those drafts:

- Glossary terms are summed across all drafts and kept when their total
  mention count is ≥ MIN_MENTIONS (default 10).
- Character names are re-counted by scanning the matching `<stem>.jp.srt`
  files (drafts don't preserve character counts), and the same threshold
  applies — so anchors who recur weekly survive, but one-off interviewees
  drop out.
- The master file is edited in place: new entries are appended to the
  existing `characters:` and `glossary:` blocks without re-rendering the
  rest of the file. Existing lines, trailing `# comments`, and ordering
  are preserved byte-for-byte.
- Entries already present in the master are skipped (zh values are never
  overwritten).
- Drafts are left in place — they'll be regenerated on the next run.

Run:
    python -m zimakuou.merge_drafts <folder> [--master <path-to-master-yaml>]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import srt
import yaml

from .audit import find_candidates
from .context import Context

MIN_MENTIONS = 10

# Note format in drafts: `note: "21×"`. Captures the integer.
_COUNT_RE = re.compile(r"(\d+)\s*[×x]")


def _parse_count(note: str) -> int:
    if not note:
        return 0
    m = _COUNT_RE.search(note)
    return int(m.group(1)) if m else 0


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _draft_stem(draft_path: Path) -> str:
    """`X.context.draft.yaml` → `X` (strip both suffixes)."""
    return draft_path.name[: -len(".context.draft.yaml")]


def _sum_glossary_counts(drafts: list[Path]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for draft in drafts:
        data = _load_yaml(draft)
        for entry in data.get("glossary") or []:
            jp = (entry.get("jp") or "").strip()
            if jp:
                totals[jp] += _parse_count(entry.get("note", ""))
    return totals


def _sum_character_counts(folder: Path, drafts: list[Path]) -> Counter[str]:
    """Re-derive character counts by re-scanning each draft's sibling jp.srt.
    Drafts only emit the deduped name list, not per-name counts."""
    totals: Counter[str] = Counter()
    for draft in drafts:
        jp_srt = folder / f"{_draft_stem(draft)}.jp.srt"
        if not jp_srt.exists():
            continue
        subs = list(srt.parse(jp_srt.read_text(encoding="utf-8")))
        names, _kata = find_candidates(subs, Context.empty())
        totals.update(names)
    return totals


def _find_default_master(folder: Path) -> Path | None:
    """The master is the one `*.context.yaml` that isn't a `.draft`."""
    candidates = [
        p for p in folder.glob("*.context.yaml")
        if not p.name.endswith(".context.draft.yaml")
    ]
    return candidates[0] if len(candidates) == 1 else None


def _append_under_key(
    text: str, key: str, new_lines: list[str]
) -> str:
    """Append `new_lines` to the existing `key:` block in `text`, preserving
    every other byte of the file. If the key doesn't exist, append a new
    block at the end.

    A block is defined as the `key:` line plus every following indented
    line, stopping at the first non-indented non-blank line (i.e. the next
    top-level key) or EOF. Trailing blank lines inside the block are kept
    where they are; new entries are inserted *before* them so the visual
    grouping stays intact.
    """
    if not new_lines:
        return text

    lines = text.splitlines(keepends=True)
    key_re = re.compile(rf"^{re.escape(key)}\s*:\s*$")
    key_idx = next(
        (i for i, line in enumerate(lines) if key_re.match(line.rstrip("\n"))),
        None,
    )
    if key_idx is None:
        # No such key — append a new block at end of file.
        prefix = "" if text.endswith("\n") else "\n"
        block = f"\n{key}:\n" + "\n".join(new_lines) + "\n"
        return text + prefix + block

    # Find the end of the block: first line at column 0 that isn't blank
    # and isn't the key line itself.
    end_idx = len(lines)
    for i in range(key_idx + 1, len(lines)):
        stripped = lines[i].rstrip("\n")
        if stripped and not stripped.startswith((" ", "\t")):
            end_idx = i
            break

    # Walk back past trailing blank lines inside the block so new entries
    # land flush with existing ones, not after the visual gap.
    insert_at = end_idx
    while insert_at > key_idx + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1

    insertion = [line + "\n" for line in new_lines]
    return "".join(lines[:insert_at] + insertion + lines[insert_at:])


def _format_character_line(name: str) -> str:
    return f"  - {name}"


def _format_glossary_line(jp: str, count: int) -> str:
    # zh is left blank by design — these are candidates awaiting translation.
    return f'  - {{ jp: {jp}, zh: "", note: "{count}×" }}'


def merge(
    folder: Path, master_path: Path, min_mentions: int = MIN_MENTIONS
) -> tuple[list[str], list[str], list[str]]:
    """Edit `master_path` in place to append new characters / glossary
    entries that pass the threshold. Returns (new_chars, new_jp_terms,
    all_jp_terms_missing_zh)."""
    drafts = sorted(folder.glob("*.context.draft.yaml"))
    if not drafts:
        raise SystemExit(f"No *.context.draft.yaml files in {folder}")

    existing = Context.load(master_path) if master_path.exists() else Context.empty()
    existing_chars = set(existing.characters)
    existing_jp = {g.jp for g in existing.glossary}

    glossary_counts = _sum_glossary_counts(drafts)
    character_counts = _sum_character_counts(folder, drafts)

    new_chars = [
        n for n, _c in sorted(
            character_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )
        if character_counts[n] >= min_mentions and n not in existing_chars
    ]
    new_glossary: list[tuple[str, int]] = [
        (jp, total)
        for jp, total in sorted(
            glossary_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )
        if total >= min_mentions and jp not in existing_jp
    ]

    text = master_path.read_text(encoding="utf-8") if master_path.exists() else ""
    text = _append_under_key(
        text, "characters", [_format_character_line(n) for n in new_chars]
    )
    text = _append_under_key(
        text,
        "glossary",
        [_format_glossary_line(jp, c) for jp, c in new_glossary],
    )
    master_path.write_text(text, encoding="utf-8")

    # Re-parse to discover everything (existing + new) still lacking zh.
    final = Context.load(master_path)
    missing_zh = [g.jp for g in final.glossary if not g.zh]
    return new_chars, [jp for jp, _ in new_glossary], missing_zh


def main() -> int:
    p = argparse.ArgumentParser(prog="zimakuou.merge_drafts", description=__doc__)
    p.add_argument("folder", type=Path, help="Folder containing draft YAMLs")
    p.add_argument(
        "--master",
        type=Path,
        default=None,
        help="Path to the show's master context.yaml (default: auto-detect)",
    )
    p.add_argument(
        "--min-mentions",
        type=int,
        default=MIN_MENTIONS,
        help=f"Minimum total mentions to keep a candidate (default: {MIN_MENTIONS})",
    )
    args = p.parse_args()

    if not args.folder.is_dir():
        print(f"error: {args.folder} is not a directory", file=sys.stderr)
        return 2

    master = args.master or _find_default_master(args.folder)
    if master is None:
        print(
            "error: could not auto-detect master .context.yaml — pass --master",
            file=sys.stderr,
        )
        return 2

    new_chars, new_jp_terms, missing_zh = merge(
        args.folder, master, min_mentions=args.min_mentions
    )

    print(f"Updated {master}")
    print(f"  +{len(new_chars)} new characters")
    print(f"  +{len(new_jp_terms)} new glossary terms")
    if missing_zh:
        print()
        print(f"Glossary entries still missing a zh translation ({len(missing_zh)}):")
        for jp in missing_zh:
            print(f"  - {jp}")
        print()
        print(f"Edit {master.name} to fill in the zh: values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
