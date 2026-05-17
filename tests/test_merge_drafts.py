from datetime import timedelta

import srt

from zimakuou.context import Context
from zimakuou.merge_drafts import _append_under_key, merge


def _write_draft(path, characters: list[str], glossary: list[tuple[str, int]]):
    lines = ["characters:"] + [f"  - {n}" for n in characters] + ["", "glossary:"] + [
        f'  - {{ jp: {jp}, zh: "", note: "{count}×" }}' for jp, count in glossary
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_jp_srt(path, lines: list[str]):
    subs = [
        srt.Subtitle(
            index=i, start=timedelta(seconds=i), end=timedelta(seconds=i + 1),
            content=line,
        )
        for i, line in enumerate(lines, start=1)
    ]
    path.write_text(srt.compose(subs), encoding="utf-8")


def test_append_under_key_inserts_before_trailing_blanks():
    """New entries should land flush with existing ones, not after the blank
    line that separates blocks — otherwise we accumulate stray blanks."""
    text = "characters:\n  - 山田\n  - 鈴木\n\nglossary:\n  - foo\n"
    out = _append_under_key(text, "characters", ["  - 佐藤"])
    assert out == "characters:\n  - 山田\n  - 鈴木\n  - 佐藤\n\nglossary:\n  - foo\n"


def test_append_under_key_appends_block_when_key_missing():
    text = "title: foo\n"
    out = _append_under_key(text, "glossary", ['  - { jp: x, zh: "" }'])
    assert "glossary:" in out
    assert '  - { jp: x, zh: "" }' in out


def test_append_under_key_no_op_when_no_new_lines():
    text = "characters:\n  - 山田\n"
    assert _append_under_key(text, "characters", []) == text


def test_merge_preserves_existing_comments_byte_for_byte(tmp_path):
    """The whole point of in-place editing: hand-written # comments on
    existing glossary lines, file headers, and ordering must survive."""
    master = tmp_path / "show.context.yaml"
    master.write_text(
        "# show-wide notes\n"
        "title: テスト番組\n"
        "\n"
        "characters:\n"
        "  - 山田\n"
        "\n"
        "glossary:\n"
        '  - { jp: 既存, zh: "已存在", note: "5×" } # hand-curated note\n',
        encoding="utf-8",
    )

    # Two drafts agree on a 12× total for an unseen term.
    _write_draft(
        tmp_path / "ep1.context.draft.yaml",
        characters=["山田"],   # already in master, must be skipped
        glossary=[("新語", 7)],
    )
    _write_draft(
        tmp_path / "ep2.context.draft.yaml",
        characters=["新人"],
        glossary=[("新語", 5)],
    )
    # No matching .jp.srt files → character threshold can't fire, which is fine.

    new_chars, new_jp, missing = merge(tmp_path, master, min_mentions=10)

    text = master.read_text(encoding="utf-8")
    # Existing content preserved verbatim:
    assert "# show-wide notes" in text
    assert "title: テスト番組" in text
    assert '{ jp: 既存, zh: "已存在", note: "5×" } # hand-curated note' in text
    # New term appended:
    assert '{ jp: 新語, zh: "", note: "12×" }' in text
    assert new_jp == ["新語"]
    # Character threshold couldn't fire without jp.srt files:
    assert new_chars == []
    # Missing-zh checklist includes only the new term:
    assert "新語" in missing
    assert "既存" not in missing


def test_merge_thresholds_characters_via_jp_srt_scan(tmp_path):
    """Character counts aren't in the draft; we re-scan jp.srt files."""
    master = tmp_path / "show.context.yaml"
    master.write_text("characters:\n\nglossary:\n", encoding="utf-8")

    # One draft, but the jp.srt mentions 山田先生 enough times to clear the bar.
    _write_draft(
        tmp_path / "ep1.context.draft.yaml",
        characters=["山田"],
        glossary=[],
    )
    _write_jp_srt(tmp_path / "ep1.jp.srt", ["山田先生"] * 12)

    new_chars, _new_jp, _missing = merge(tmp_path, master, min_mentions=10)
    assert new_chars == ["山田"]
    assert "  - 山田" in master.read_text(encoding="utf-8")


def test_merge_skips_terms_already_in_master(tmp_path):
    master = tmp_path / "show.context.yaml"
    master.write_text(
        'glossary:\n  - { jp: 既存, zh: "已存在", note: "20×" }\n',
        encoding="utf-8",
    )
    _write_draft(
        tmp_path / "ep1.context.draft.yaml",
        characters=[],
        glossary=[("既存", 50)],  # huge count, but already known
    )
    new_chars, new_jp, missing = merge(tmp_path, master, min_mentions=10)
    assert new_jp == []
    assert missing == []  # existing 既存 has zh, no new terms added
