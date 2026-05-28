from zimakuou.transcribe import is_runaway, split_long_cue


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


def test_repeated_bigram_pattern_dropped():
    # Real failure: "自動自動自動..." — neither 自 nor 動 exceeds 50%
    # individually, but the 2-char unit tiles the whole string.
    assert is_runaway("自動" * 30, 15.0)
    # Also catches 3-char and 4-char repeating units
    assert is_runaway("おはよ" * 15, 5.0)
    assert is_runaway("ありがと" * 12, 5.0)


def test_repeating_pattern_needs_enough_text():
    # Short text that happens to repeat shouldn't be flagged
    assert not is_runaway("自動自動", 2.0)


def test_micro_cue_spam_dropped():
    # Single-char cues in <0.3s — decoder sputtering after a stuck loop
    assert is_runaway("自", 0.04)
    assert is_runaway("動", 0.1)
    assert is_runaway("自", 0.29)


def test_short_interjection_not_micro_cue():
    # "うん" at 0.5s is a real interjection, not spam
    assert not is_runaway("うん", 0.5)
    # Even very short real speech: "え" at 0.4s
    assert not is_runaway("え", 0.4)


def test_split_short_cue_at_silence_gaps():
    words = [(0.0, 1.0, "こん"), (1.2, 2.0, "にちは"), (3.0, 5.0, "皆さん")]
    out = split_long_cue(0.0, 5.0, "こんにちは皆さん", words, max_dur=12.0)
    assert out == [(0.0, 1.0, "こん"), (1.2, 2.0, "にちは"), (3.0, 5.0, "皆さん")]


def test_split_long_cue_at_largest_gap():
    # 20s cue with a big silence in the middle — should split there.
    words = [
        (0.0, 2.0, "おはよう"),
        (2.0, 4.0, "ございます"),     # tight cluster, gap of 0.0 after
        (10.0, 12.0, "今日は"),       # big 6s gap before this word
        (12.0, 14.0, "いい天気"),
        (14.0, 16.0, "ですね"),
    ]
    out = split_long_cue(0.0, 16.0, "おはようございます今日はいい天気ですね", words, max_dur=12.0)
    assert len(out) == 2
    left, right = out
    assert left == (0.0, 4.0, "おはようございます")
    assert right == (10.0, 16.0, "今日はいい天気ですね")


def test_split_recurses_until_under_budget():
    # 30s cue. First split should yield two ~15s halves; the larger
    # half still exceeds 12s and should be split again.
    words = [
        (0.0, 3.0, "A"),
        (3.0, 6.0, "B"),
        (10.0, 13.0, "C"),  # 4s gap — biggest, primary split point
        (13.0, 16.0, "D"),
        (18.0, 21.0, "E"),  # 2s gap — secondary split inside right half
        (21.0, 24.0, "F"),
    ]
    out = split_long_cue(0.0, 24.0, "ABCDEF", words, max_dur=10.0)
    # Three pieces, each ≤10s and in time order.
    assert len(out) == 3
    assert all(end - start <= 10.0 for start, end, _ in out)
    starts = [s for s, _, _ in out]
    assert starts == sorted(starts)


def test_split_no_word_timestamps_leaves_cue_alone():
    # No word_timestamps → can't split safely. Better one long cue than
    # an arbitrary mid-string chop.
    out = split_long_cue(0.0, 30.0, "テキスト", words=[], max_dur=10.0)
    assert out == [(0.0, 30.0, "テキスト")]


def test_split_no_real_gap_keeps_cue_when_not_egregious():
    # Tightly packed words and only moderately over budget — splitting
    # would produce an awkward mid-sentence chop. Leave it.
    words = [(i * 1.0, (i + 1) * 1.0, "x") for i in range(13)]
    out = split_long_cue(0.0, 13.0, "x" * 13, words, max_dur=12.0)
    assert out == [(0.0, 13.0, "x" * 13)]


