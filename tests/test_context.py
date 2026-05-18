from pathlib import Path

from zimakuou.context import Context, GlossaryEntry


def _write(p: Path, body: str) -> Path:
    p.write_text(body, encoding="utf-8")
    return p


def test_load_minimal_yaml(tmp_path: Path):
    p = _write(
        tmp_path / "ctx.yaml",
        "title: 悟の恋\n"
        "title_zh_tw: 慧的愛戀\n"
        "characters: [悟, ミミ]\n"
        "glossary:\n"
        "  - {jp: スペック, zh: 讀心能力, note: protagonist's ability}\n"
        "  - {jp: サトリ, zh: 悟}\n",
    )
    ctx = Context.load(p)

    assert ctx.title == "悟の恋"
    assert ctx.title_zh_tw == "慧的愛戀"
    assert ctx.characters == ["悟", "ミミ"]
    assert len(ctx.glossary) == 2
    assert ctx.glossary[0] == GlossaryEntry(
        jp="スペック", zh="讀心能力", note="protagonist's ability"
    )


def test_load_empty_yaml_returns_empty(tmp_path: Path):
    p = _write(tmp_path / "empty.yaml", "")
    ctx = Context.load(p)
    assert ctx.is_empty()


def test_whisper_initial_prompt_dedupes_and_caps_length():
    ctx = Context(
        characters=["悟", "ミミ", "悟"],  # duplicate
        glossary=[
            GlossaryEntry(jp="スペック", zh="讀心"),
            GlossaryEntry(jp="サトリ", zh="悟"),
        ],
    )
    prompt = ctx.whisper_initial_prompt()
    assert "悟" in prompt
    assert "ミミ" in prompt
    assert "スペック" in prompt
    assert prompt.count("悟") == 1  # deduped between characters and glossary.zh? no — JP only
    assert len(prompt) <= 200


def test_whisper_initial_prompt_empty_when_no_data():
    assert Context.empty().whisper_initial_prompt() == ""


def test_whisper_initial_prompt_includes_synopsis():
    """Synopsis content (e.g. an auto-merged `.description` sidecar) should
    flow into the whisper prompt so per-episode vocabulary primes ASR."""
    ctx = Context(synopsis="セロリのきんぴら、春野菜のうま煮を紹介する。")
    prompt = ctx.whisper_initial_prompt()
    assert "きんぴら" in prompt
    assert "うま煮" in prompt
    assert len(prompt) <= 200


def test_whisper_initial_prompt_omits_entries_marked_whisper_false():
    """`whisper: false` entries stay in apply_glossary + llm_context_block
    but get excluded from the whisper initial_prompt budget."""
    ctx = Context(
        glossary=[
            GlossaryEntry(jp="きんぴら", zh="金平"),                       # primes whisper
            GlossaryEntry(jp="ほうれん草", zh="菠菜", whisper=False),       # post-pass only
        ],
    )
    prompt = ctx.whisper_initial_prompt()
    assert "きんぴら" in prompt
    assert "ほうれん草" not in prompt
    # The LLM context block ignores the flag — full glossary goes there
    assert "ほうれん草" in ctx.llm_context_block()
    # And the post-pass still substitutes — whisper:false doesn't disable that
    assert ctx.apply_glossary("我喜歡ほうれん草") == "我喜歡菠菜"


def test_load_yaml_with_whisper_flag(tmp_path: Path):
    p = _write(
        tmp_path / "ctx.yaml",
        "glossary:\n"
        "  - {jp: きんぴら, zh: 金平}\n"
        "  - {jp: 食材, zh: 食材, whisper: false}\n",
    )
    ctx = Context.load(p)
    assert ctx.glossary[0].whisper is True   # default
    assert ctx.glossary[1].whisper is False  # explicit opt-out


def test_whisper_initial_prompt_terms_first_synopsis_fills_remainder():
    ctx = Context(
        characters=["悟"],
        glossary=[GlossaryEntry(jp="スペック", zh="讀心")],
        synopsis="あらすじテキスト",
    )
    prompt = ctx.whisper_initial_prompt()
    assert prompt.index("悟") < prompt.index("あらすじ")


def test_llm_context_block_includes_glossary_and_synopsis():
    ctx = Context(
        title="悟の恋",
        synopsis="A boy with mind-reading powers.",
        characters=["悟", "ミミ"],
        glossary=[GlossaryEntry(jp="サトリ", zh="悟", note="protagonist")],
    )
    block = ctx.llm_context_block()
    assert "悟の恋" in block
    assert "mind-reading" in block
    assert "サトリ" in block
    assert "protagonist" in block
    assert "用語集" in block


def test_llm_context_block_empty_for_empty_context():
    assert Context.empty().llm_context_block() == ""


def test_apply_glossary_replaces_jp_terms_left_in_zh_output():
    ctx = Context(
        glossary=[
            GlossaryEntry(jp="ミミ", zh="咪咪"),
            GlossaryEntry(jp="スペック", zh="讀心能力"),
        ]
    )
    # LLM left "ミミ" untranslated in the Chinese output — glossary fixes it
    assert ctx.apply_glossary("我喜歡ミミ") == "我喜歡咪咪"
    # Multiple replacements in one string
    assert ctx.apply_glossary("ミミとスペック") == "咪咪と讀心能力"
    # No-op if no glossary terms present
    assert ctx.apply_glossary("這沒有任何術語") == "這沒有任何術語"


def test_apply_glossary_empty_context_is_identity():
    assert Context.empty().apply_glossary("不變") == "不變"


def test_with_sidecar_appends_description_to_synopsis(tmp_path: Path):
    video = tmp_path / "ep01.mp4"
    video.touch()
    _write(tmp_path / "ep01.description", "今日は副菜のレシピを紹介する。")
    ctx = Context(synopsis="show-wide synopsis")

    merged = ctx.with_sidecar(video)

    assert merged is not ctx
    assert "show-wide synopsis" in merged.synopsis
    assert "副菜のレシピ" in merged.synopsis


def test_with_sidecar_uses_description_when_synopsis_empty(tmp_path: Path):
    video = tmp_path / "ep01.mp4"
    video.touch()
    _write(tmp_path / "ep01.description", "episode-only synopsis")
    ctx = Context.empty()

    merged = ctx.with_sidecar(video)

    assert merged.synopsis == "episode-only synopsis"


def test_with_sidecar_returns_self_when_no_sidecar(tmp_path: Path):
    video = tmp_path / "ep01.mp4"
    video.touch()
    ctx = Context(synopsis="show-wide")

    merged = ctx.with_sidecar(video)

    assert merged is ctx


def test_with_sidecar_ignores_empty_sidecar(tmp_path: Path):
    video = tmp_path / "ep01.mp4"
    video.touch()
    _write(tmp_path / "ep01.description", "   \n  ")
    ctx = Context(synopsis="kept")

    merged = ctx.with_sidecar(video)

    assert merged is ctx
