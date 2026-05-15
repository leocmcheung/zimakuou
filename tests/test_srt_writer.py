from datetime import timedelta
from pathlib import Path

import srt

from zimakuou.srt_writer import write_bilingual, write_srt


def _sub(i: int, start: float, end: float, text: str) -> srt.Subtitle:
    return srt.Subtitle(
        index=i,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        content=text,
    )


def test_write_srt_roundtrips(tmp_path: Path):
    subs = [_sub(1, 0.0, 1.0, "こんにちは"), _sub(2, 1.0, 2.0, "さようなら")]
    out = tmp_path / "x.srt"
    write_srt(subs, out)

    parsed = list(srt.parse(out.read_text("utf-8")))
    assert [s.content for s in parsed] == ["こんにちは", "さようなら"]
    assert parsed[0].start == timedelta(seconds=0)
    assert parsed[1].end == timedelta(seconds=2)


def test_bilingual_stacks_jp_over_zh(tmp_path: Path):
    jp = [_sub(1, 0.0, 1.0, "こんにちは")]
    zh = [_sub(1, 0.0, 1.0, "你好")]
    out = tmp_path / "bi.srt"
    write_bilingual(jp, zh, out)

    parsed = list(srt.parse(out.read_text("utf-8")))
    assert parsed[0].content == "こんにちは\n你好"
    assert parsed[0].start == timedelta(seconds=0)
    assert parsed[0].end == timedelta(seconds=1)
