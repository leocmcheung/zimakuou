from pathlib import Path

import srt


def write_srt(subs: list[srt.Subtitle], path: Path) -> None:
    path.write_text(srt.compose(subs), encoding="utf-8")


def write_bilingual(
    jp_subs: list[srt.Subtitle], zh_subs: list[srt.Subtitle], path: Path
) -> None:
    """Stack Japanese on top, Traditional Chinese below, sharing each cue's timing."""
    merged = [
        srt.Subtitle(
            index=jp.index,
            start=jp.start,
            end=jp.end,
            content=f"{jp.content}\n{zh.content}",
        )
        for jp, zh in zip(jp_subs, zh_subs)
    ]
    path.write_text(srt.compose(merged), encoding="utf-8")
