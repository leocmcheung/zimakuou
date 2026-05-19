"""Heuristic glossary candidate extraction from a JP SRT.

After every translation run we scan the .jp.srt for terms that probably
need to live in a context glossary — long katakana loanwords/names and
kanji/hiragana names with honorifics — and emit a draft `<stem>.context.draft.yaml`
next to the video. The user fills in the `zh:` side, drops the `.draft`
from the filename, and re-runs the translate stage with `--context`.

We deliberately do *not* ask the LLM to propose the `zh:` side. The terms
Sakura misses in translation are the same ones it would mistranslate in
glossary suggestions, so heuristic-only is more honest about what it
doesn't know.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import srt as srt_mod

    from .context import Context


# Long-katakana run, ≥4 chars (゠–ヿ is the katakana block).
# Catches loanwords/foreign names like "インティマシーコーディネーター".
# Short katakana (バー, アニメ) is too generic to be glossary-worthy.
_KATAKANA_RE = re.compile(r"[゠-ヿ]{4,}")

# Name + honorific. Three guards keep the noise floor low:
#   1) The captured name must be single-script (kanji, katakana, or ASCII)
#      — mixing scripts makes the regex grab particles and extension marks
#      before a real name (e.g. matching "ーの西山桃子" out of
#      "コーディネーターの西山桃子").
#   2) A negative-lookbehind requires the previous char to NOT be a name
#      character, so we get the full name run, not a 6-char tail of a
#      longer word (e.g. avoiding "ディネーター" out of "コーディネーターさん").
#   3) Hiragana-only candidates are dropped entirely — hiragana names exist
#      (あおい, さくら) but the false-positive rate from sentence endings
#      ("そうですよ先生", "してるさん") is much higher than the signal.
_HONORIFIC_RE = re.compile(
    # Exclude kanji/katakana/ASCII before the match (so "ディネーター" out of
    # "コーディネーターさん" doesn't match) but NOT hiragana — particles
    # like の/が/を legitimately precede names ("の西山桃子先生").
    r"(?<![一-鿿゠-ヿA-Za-z])"
    r"("
    r"[一-鿿]{2,4}"
    r"|[゠-ヿ]{2,5}"
    r"|[A-Za-z]{2,8}"
    r")(先生|さん|ちゃん|くん|様|さま)"
)

# Pronouns / particles that the honorific regex picks up but aren't names.
_NAME_STOPWORDS = {
    "あなた", "あんた", "おまえ", "きみ", "わたし", "あたし", "ぼく",
    "おれ", "うち", "みんな", "ども", "皆",
}

# Common katakana loanwords that pass the ≥4-char regex but aren't worth
# glossarying — every Japanese audience knows them, and listing them just
# wastes prompt budget. Edit per-project if a show legitimately centres
# on one of these (e.g. an America-focused travel show wanting アメリカ
# tracked as a recurring topic).
_KATAKANA_STOPWORDS = {
    # Countries / regions
    "アメリカ", "ヨーロッパ", "アフリカ", "イギリス", "フランス", "イタリア",
    "スペイン", "オランダ", "ベルギー", "ポルトガル", "ギリシャ", "メキシコ",
    "ブラジル", "アルゼンチン", "オーストラリア", "ニュージーランド",
    "インドネシア", "マレーシア", "フィリピン", "ベトナム", "シンガポール",
    "イスラエル",
    # Common concepts
    "ニュース", "スポーツ", "インターネット", "シリーズ", "バランス",
    "パターン", "メッセージ", "システム", "スタイル", "イメージ",
    "スケジュール", "アドバイス", "ストレス", "グループ", "メンバー",
    "パーティー", "レストラン", "スーパー", "デパート", "マンション",
    # Products / tech / loanwords
    "パソコン", "スマートフォン", "シンプル", "ナチュラル", "スマート",
    "メディア", "アイドル",
}

KATAKANA_MIN_COUNT = 3  # need ≥3 mentions in an episode to suggest it
NAME_MIN_COUNT = 1      # names matter even when mentioned once


def find_candidates(
    jp_subs: list["srt_mod.Subtitle"],
    existing: "Context",
) -> tuple[Counter[str], Counter[str]]:
    """Return (name_counter, katakana_counter). Drops anything already in
    `existing` (characters list or glossary jp side) and noise stopwords."""
    text = "\n".join(s.content for s in jp_subs)

    katakana = Counter(_KATAKANA_RE.findall(text))
    names = Counter(m.group(1) for m in _HONORIFIC_RE.finditer(text))

    covered = {g.jp for g in existing.glossary} | set(existing.characters)

    katakana = Counter({
        k: v for k, v in katakana.items()
        if v >= KATAKANA_MIN_COUNT and k not in covered and k not in _KATAKANA_STOPWORDS
    })
    names = Counter({
        n: v for n, v in names.items()
        if v >= NAME_MIN_COUNT and n not in covered and n not in _NAME_STOPWORDS
    })
    return names, katakana


def _render_yaml(names: Counter[str], katakana: Counter[str]) -> str:
    lines = [
        "# Draft glossary — auto-generated from the .jp.srt heuristic audit.",
        "# Fill in `zh:` for the entries that matter, delete the rest, rename",
        "# this file to <stem>.context.yaml (drop `.draft`), then re-run:",
        "#   python -m zimakuou.translate <jp.srt> --context <yaml>",
        "",
    ]
    if names:
        lines.append("characters:")
        for n, _ in sorted(names.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  - {n}")
        lines.append("")
    if katakana:
        lines.append("glossary:")
        for term, count in sorted(katakana.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f'  - {{ jp: {term}, zh: "", note: "{count}×" }}')
        lines.append("")
    return "\n".join(lines)


def write_glossary_draft(
    jp_subs: list["srt_mod.Subtitle"],
    out_path: Path,
    existing: "Context",
) -> Path | None:
    """Write a draft context YAML to `out_path` if there are candidates not
    already covered by `existing`. Returns the path written, or None when
    everything is already covered (in which case nothing is emitted)."""
    names, katakana = find_candidates(jp_subs, existing)
    if not names and not katakana:
        return None
    out_path.write_text(_render_yaml(names, katakana), encoding="utf-8")
    return out_path
