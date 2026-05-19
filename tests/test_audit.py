from datetime import timedelta

import srt
import yaml

from zimakuou.audit import (
    _KATAKANA_STOPWORDS,
    _load_jlpt_stopwords,
    find_candidates,
    write_glossary_draft,
)
from zimakuou.context import Context, GlossaryEntry


def _sub(i: int, text: str) -> srt.Subtitle:
    return srt.Subtitle(
        index=i, start=timedelta(seconds=i), end=timedelta(seconds=i + 1), content=text
    )


def test_finds_long_katakana_terms_at_or_above_threshold():
    # KATAKANA_MIN_COUNT = 5 — need at least five mentions per episode.
    subs = [_sub(i, "今日はインティマシーコーディネーターの話") for i in range(1, 6)]
    subs.append(_sub(6, "バー"))  # short katakana — ignored regardless of count
    _names, kata = find_candidates(subs, Context.empty())
    assert "インティマシーコーディネーター" in kata
    assert kata["インティマシーコーディネーター"] == 5
    assert "バー" not in kata


def test_skips_below_threshold_katakana():
    # Four occurrences — just below the ≥5 threshold.
    subs = [_sub(i, "プロフェッショナル") for i in range(1, 5)]
    _names, kata = find_candidates(subs, Context.empty())
    assert "プロフェッショナル" not in kata


def test_skips_common_katakana_in_stopword_list():
    """Common loanwords like アメリカ (in JLPT) and ロボット (gap-fill)
    pass the regex + count threshold but aren't worth glossarying."""
    subs = [_sub(i, "アメリカに行った") for i in range(1, 6)]
    subs += [_sub(i, "ロボットの話") for i in range(6, 11)]
    _names, kata = find_candidates(subs, Context.empty())
    assert "アメリカ" not in kata  # via JLPT file
    assert "ロボット" not in kata  # via inline gap-fill


def test_finds_names_with_honorifics():
    # NAME_MIN_COUNT = 2 — need at least two mentions per episode.
    subs = [
        _sub(1, "西山桃子先生です"),
        _sub(2, "西山桃子先生と話した"),
        _sub(3, "美咲ちゃんと出かけた"),
        _sub(4, "美咲ちゃんが来た"),
        _sub(5, "MEGUMIさんの番組"),
        _sub(6, "MEGUMIさんに会った"),
    ]
    names, _kata = find_candidates(subs, Context.empty())
    assert "西山桃子" in names
    assert "美咲" in names
    assert "MEGUMI" in names


def test_drops_pronouns_misread_as_names():
    # "あなたさん" / "あなた様" shows up in the wild; stopwords protect us
    # even when they pass the count threshold.
    subs = [
        _sub(1, "あなた様、わたしさん"),
        _sub(2, "あなた様、わたしさん"),
    ]
    names, _kata = find_candidates(subs, Context.empty())
    assert "あなた" not in names
    assert "わたし" not in names


def test_skips_terms_already_in_existing_context():
    existing = Context(
        characters=["西山桃子"],
        glossary=[GlossaryEntry(jp="インティマシーコーディネーター", zh="親密戲協調員")],
    )
    subs = [_sub(i, "インティマシーコーディネーター") for i in range(1, 6)]
    subs += [_sub(6, "西山桃子先生"), _sub(7, "西山桃子先生")]
    subs += [_sub(8, "美咲ちゃん"), _sub(9, "美咲ちゃん")]  # this one is new
    names, kata = find_candidates(subs, existing)
    assert "インティマシーコーディネーター" not in kata  # already glossaried
    assert "西山桃子" not in names                       # already a character
    assert "美咲" in names                                # new — surface it


def test_jlpt_stopwords_loaded_and_cover_common_loanwords():
    jlpt = _load_jlpt_stopwords()
    # The bundled file is non-trivial — sanity check the size hasn't
    # regressed to empty (which would silently disable the filter).
    assert len(jlpt) > 300
    # A few words from each level should be present.
    assert "カメラ" in jlpt          # N5
    assert "エネルギー" in jlpt      # N3
    assert "ダイヤモンド" in jlpt    # N2


def test_gapfill_covers_common_loanwords_missing_from_jlpt():
    """JLPT vocabulary is curriculum-scoped, not frequency-ranked, so a few
    very common loanwords (スピード, プロジェクト) aren't in any JLPT level.
    The gap-fill set picks those up — the combined set must filter all of
    `ダイヤモンド`, `スピード`, `プロジェクト` from the original discussion."""
    jlpt = _load_jlpt_stopwords()
    # スピード and プロジェクト aren't in JLPT — confirms the gap-fill is
    # load-bearing for them.
    assert "スピード" not in jlpt
    assert "プロジェクト" not in jlpt
    # But ALL three should be in the merged set used by the audit.
    for w in ("ダイヤモンド", "スピード", "プロジェクト"):
        assert w in _KATAKANA_STOPWORDS, f"{w} should be filtered"


def test_write_returns_none_when_nothing_to_surface(tmp_path):
    out = tmp_path / "draft.yaml"
    result = write_glossary_draft([_sub(1, "はい")], out, Context.empty())
    assert result is None
    assert not out.exists()


def test_emitted_yaml_parses_back_into_a_context(tmp_path):
    # ≥5 katakana mentions, ≥2 name mentions to clear thresholds.
    subs = [
        _sub(i, "インティマシーコーディネーターの西山桃子先生")
        for i in range(1, 6)
    ]
    subs += [_sub(6, "美咲ちゃん"), _sub(7, "美咲ちゃんが来た")]
    out = tmp_path / "ep.context.draft.yaml"
    assert write_glossary_draft(subs, out, Context.empty()) == out

    # The draft should round-trip through Context.load — proving the YAML is
    # both valid and shaped like a real context file.
    ctx = Context.load(out)
    assert "西山桃子" in ctx.characters
    assert "美咲" in ctx.characters
    jp_terms = [g.jp for g in ctx.glossary]
    assert "インティマシーコーディネーター" in jp_terms
    # And the zh side is intentionally blank — user has to fill it in.
    for g in ctx.glossary:
        assert g.zh == ""


def test_emitted_yaml_includes_human_readable_header(tmp_path):
    subs = [_sub(1, "西山桃子先生"), _sub(2, "西山桃子先生")]
    out = tmp_path / "ep.context.draft.yaml"
    write_glossary_draft(subs, out, Context.empty())

    raw = out.read_text(encoding="utf-8")
    # Header comments tell the user what to do next.
    assert raw.startswith("# Draft glossary")
    assert "--context" in raw
