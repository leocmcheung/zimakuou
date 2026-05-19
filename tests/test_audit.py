from datetime import timedelta

import srt
import yaml

from zimakuou.audit import find_candidates, write_glossary_draft
from zimakuou.context import Context, GlossaryEntry


def _sub(i: int, text: str) -> srt.Subtitle:
    return srt.Subtitle(
        index=i, start=timedelta(seconds=i), end=timedelta(seconds=i + 1), content=text
    )


def test_finds_long_katakana_terms_at_or_above_threshold():
    subs = [
        _sub(1, "今日はインティマシーコーディネーターと話す"),
        _sub(2, "インティマシーコーディネーターって何"),
        _sub(3, "インティマシーコーディネーターの仕事"),
        _sub(4, "バー"),  # short katakana — ignored
    ]
    _names, kata = find_candidates(subs, Context.empty())
    assert "インティマシーコーディネーター" in kata
    assert kata["インティマシーコーディネーター"] == 3
    assert "バー" not in kata


def test_skips_below_threshold_katakana():
    # ≥3 occurrences threshold — anything rarer is too noisy to surface.
    subs = [_sub(1, "プロフェッショナル"), _sub(2, "プロフェッショナル")]
    _names, kata = find_candidates(subs, Context.empty())
    assert "プロフェッショナル" not in kata


def test_skips_common_katakana_in_stopword_list():
    """Common loanwords like アメリカ pass the regex + count threshold but
    aren't worth glossarying — every audience knows them."""
    subs = [
        _sub(1, "アメリカに行った"),
        _sub(2, "アメリカは広い"),
        _sub(3, "アメリカの文化"),
        _sub(4, "ニュースを見た"),
        _sub(5, "ニュースで知った"),
        _sub(6, "今日のニュース"),
    ]
    _names, kata = find_candidates(subs, Context.empty())
    assert "アメリカ" not in kata
    assert "ニュース" not in kata


def test_finds_names_with_honorifics():
    subs = [
        _sub(1, "西山桃子先生です"),
        _sub(2, "美咲ちゃんと話した"),
        _sub(3, "MEGUMIさんの番組"),
    ]
    names, _kata = find_candidates(subs, Context.empty())
    assert "西山桃子" in names
    assert "美咲" in names
    assert "MEGUMI" in names


def test_drops_pronouns_misread_as_names():
    # "あなたさん" / "あなた様" shows up in the wild; stopwords protect us.
    subs = [_sub(1, "あなた様、わたしさん")]
    names, _kata = find_candidates(subs, Context.empty())
    assert "あなた" not in names
    assert "わたし" not in names


def test_skips_terms_already_in_existing_context():
    existing = Context(
        characters=["西山桃子"],
        glossary=[GlossaryEntry(jp="インティマシーコーディネーター", zh="親密戲協調員")],
    )
    subs = [
        _sub(1, "インティマシーコーディネーター"),
        _sub(2, "インティマシーコーディネーター"),
        _sub(3, "インティマシーコーディネーター"),
        _sub(4, "西山桃子先生"),
        _sub(5, "美咲ちゃん"),  # this one is new
    ]
    names, kata = find_candidates(subs, existing)
    assert "インティマシーコーディネーター" not in kata  # already glossaried
    assert "西山桃子" not in names                       # already a character
    assert "美咲" in names                                # new — surface it


def test_write_returns_none_when_nothing_to_surface(tmp_path):
    out = tmp_path / "draft.yaml"
    result = write_glossary_draft([_sub(1, "はい")], out, Context.empty())
    assert result is None
    assert not out.exists()


def test_emitted_yaml_parses_back_into_a_context(tmp_path):
    subs = [
        _sub(1, "インティマシーコーディネーターの西山桃子先生"),
        _sub(2, "インティマシーコーディネーターの仕事"),
        _sub(3, "インティマシーコーディネーターと話す"),
        _sub(4, "美咲ちゃん"),
    ]
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
    subs = [_sub(1, "西山桃子先生")]
    out = tmp_path / "ep.context.draft.yaml"
    write_glossary_draft(subs, out, Context.empty())

    raw = out.read_text(encoding="utf-8")
    # Header comments tell the user what to do next.
    assert raw.startswith("# Draft glossary")
    assert "--context" in raw
