from zimakuou.transcribe import is_runaway


def test_normal_cue_kept():
    # Real anime/news pace: ~10 chars over 3s ≈ 3 chars/sec
    assert not is_runaway("こんにちは、お元気ですか", 3.0)


def test_short_interjection_kept():
    # "うん" (yeah) — 2 chars over a brief moment is normal
    assert not is_runaway("うん", 1.0)
    assert not is_runaway("はい", 0.5)


def test_stuck_loop_dropped():
    # Real failure observed on WBS clip: cue 18 spanned 30s for "最も効果的で"
    assert is_runaway("最も効果的で", 30.0)
    # Also catches modest stretches
    assert is_runaway("短い", 10.0)


def test_short_text_in_short_window_kept():
    # 8 chars in 4s is fine — the threshold needs duration > 5s
    assert not is_runaway("最も効果的で", 4.0)


def test_repeated_character_noise_dropped():
    # Real failure observed: "ほほほほほ..." × hundreds of chars
    assert is_runaway("ほ" * 200, 0.3)
    # Also "ピピピピピ" pattern
    assert is_runaway("ピ" * 50, 1.0)


def test_repeated_character_threshold_is_majority():
    # Mixed text isn't dropped just because one char appears often
    assert not is_runaway("ありがとうございます、ありがとうございます", 4.0)


def test_empty_text_dropped():
    assert is_runaway("", 1.0)
    assert is_runaway("   ", 1.0)
