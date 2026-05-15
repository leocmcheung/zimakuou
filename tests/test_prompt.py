from zimakuou.translators._prompt import SYSTEM, build_prompt


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
