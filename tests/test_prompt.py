from zimakuou.context import GlossaryEntry
from zimakuou.translators import detect_style
from zimakuou.translators._prompt import (
    SAKURA_SYSTEM,
    SYSTEM,
    build_prompt,
    build_sakura_prompt,
)


def test_system_pins_traditional_chinese():
    assert "繁體中文" in SYSTEM


def test_build_prompt_with_no_context():
    p = build_prompt("こんにちは", [])
    assert "こんにちは" in p
    assert "Prior" not in p


def test_build_prompt_uses_sliding_window_of_last_5():
    history = [f"line{i}" for i in range(10)]
    p = build_prompt("now", history)

    assert "now" in p
    assert "line9" in p
    assert "line5" in p
    # older cues are dropped — 5-cue window
    assert "line4" not in p
    assert "line0" not in p


def test_sakura_system_is_the_trained_chinese_string():
    # Sakura was fine-tuned on this exact string; drift here will silently
    # degrade quality, so pin a few load-bearing fragments.
    assert "轻小说翻译模型" in SAKURA_SYSTEM
    assert "简体中文" in SAKURA_SYSTEM


def test_sakura_prompt_embeds_glossary_as_gpt_dict():
    glossary = [
        GlossaryEntry(jp="スペック", zh="讀心能力", note="protagonist's ability"),
        GlossaryEntry(jp="ミミ", zh="咪咪"),
    ]
    p = build_sakura_prompt("こんにちは", glossary)

    # Trained user-prompt scaffolding
    assert "根据以下术语表" in p
    assert "将下面的日文文本" in p
    # gpt_dict format: jp->zh, optionally `#note`
    assert "スペック->讀心能力 #protagonist's ability" in p
    assert "ミミ->咪咪" in p
    # source text appears verbatim at the tail
    assert p.endswith("こんにちは")


def test_sakura_prompt_with_empty_glossary_still_includes_scaffolding():
    # Empty gpt_dict is the trained-on case — the scaffolding must remain.
    p = build_sakura_prompt("こんにちは", [])
    assert "根据以下术语表" in p
    assert "将下面的日文文本" in p
    assert "こんにちは" in p


def test_detect_style_picks_sakura_from_repo_name():
    assert detect_style("SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF") == "sakura"
    assert detect_style("/path/to/sakura-14b-qwen2.5-v1.0-q4km.gguf") == "sakura"
    assert detect_style("mlx-community/Qwen2.5-7B-Instruct-4bit") == "generic"
    assert detect_style(None) == "generic"
