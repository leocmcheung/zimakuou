from datetime import timedelta

import srt

from zimakuou import translate as translate_mod


class StubTranslator:
    """Returns a Simplified-character translation so we also verify the
    OpenCC s2twp safety net converts to Traditional."""

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def translate(self, text: str, context: list[str]) -> str:
        self.calls.append((text, list(context)))
        # 译 is Simplified; OpenCC s2twp should map it to Traditional 譯
        return f"译[{text}]"


def _sub(i: int, start: float, end: float, text: str) -> srt.Subtitle:
    return srt.Subtitle(
        index=i,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        content=text,
    )


def test_translate_preserves_indices_and_timing(monkeypatch):
    stub = StubTranslator()
    monkeypatch.setattr(translate_mod, "get_translator", lambda _llm: stub)

    jp = [_sub(1, 0.0, 1.0, "あ"), _sub(2, 1.0, 2.0, "い")]
    zh = translate_mod.translate_subs(jp)

    assert [s.index for s in zh] == [1, 2]
    assert [s.start for s in zh] == [s.start for s in jp]
    assert [s.end for s in zh] == [s.end for s in jp]


def test_opencc_safety_net_converts_simplified_to_traditional(monkeypatch):
    stub = StubTranslator()
    monkeypatch.setattr(translate_mod, "get_translator", lambda _llm: stub)

    jp = [_sub(1, 0.0, 1.0, "あ")]
    zh = translate_mod.translate_subs(jp)

    # Stub emits Simplified 译 — OpenCC s2twp must convert it to Traditional 譯
    assert "譯" in zh[0].content
    assert "译" not in zh[0].content


def test_translator_receives_sliding_history(monkeypatch):
    stub = StubTranslator()
    monkeypatch.setattr(translate_mod, "get_translator", lambda _llm: stub)

    jp = [_sub(i + 1, float(i), float(i + 1), f"line{i}") for i in range(3)]
    translate_mod.translate_subs(jp)

    assert stub.calls[0][1] == []
    assert stub.calls[1][1] == ["line0"]
    assert stub.calls[2][1] == ["line0", "line1"]
