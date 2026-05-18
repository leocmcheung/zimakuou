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
    monkeypatch.setattr(translate_mod, "get_translator", lambda _llm, *, ctx=None, **_: stub)

    jp = [_sub(1, 0.0, 1.0, "あ"), _sub(2, 1.0, 2.0, "い")]
    zh = translate_mod.translate_subs(jp)

    assert [s.index for s in zh] == [1, 2]
    assert [s.start for s in zh] == [s.start for s in jp]
    assert [s.end for s in zh] == [s.end for s in jp]


def test_opencc_safety_net_converts_simplified_to_traditional(monkeypatch):
    stub = StubTranslator()
    monkeypatch.setattr(translate_mod, "get_translator", lambda _llm, *, ctx=None, **_: stub)

    jp = [_sub(1, 0.0, 1.0, "あ")]
    zh = translate_mod.translate_subs(jp)

    # Stub emits Simplified 译 — OpenCC s2twp must convert it to Traditional 譯
    assert "譯" in zh[0].content
    assert "译" not in zh[0].content


def test_context_glossary_is_applied_post_translation(monkeypatch):
    from zimakuou.context import Context, GlossaryEntry

    class LeakyTranslator:
        """Pretends to be a real LLM that left a JP glossary term in the output."""
        def translate(self, text, context):
            return f"我喜歡ミミ"  # leaks "ミミ" untranslated

    monkeypatch.setattr(
        translate_mod, "get_translator", lambda _llm, *, ctx=None, **_: LeakyTranslator()
    )

    ctx = Context(glossary=[GlossaryEntry(jp="ミミ", zh="咪咪")])
    jp = [_sub(1, 0.0, 1.0, "あ")]
    zh = translate_mod.translate_subs(jp, ctx=ctx)

    # Glossary post-pass should have replaced ミミ → 咪咪
    assert "咪咪" in zh[0].content
    assert "ミミ" not in zh[0].content


def test_context_is_threaded_to_translator(monkeypatch):
    """translate_subs must pass the Context through to get_translator so the
    backend can render it into its native prompt format (generic system block
    vs Sakura gpt_dict)."""
    from zimakuou.context import Context, GlossaryEntry

    captured: dict[str, object] = {}

    def fake_get_translator(_llm, *, ctx=None, **_):
        captured["ctx"] = ctx
        return StubTranslator()

    monkeypatch.setattr(translate_mod, "get_translator", fake_get_translator)

    ctx = Context(
        synopsis="A boy reads minds.",
        glossary=[GlossaryEntry(jp="サトリ", zh="悟")],
    )
    translate_mod.translate_subs([_sub(1, 0.0, 1.0, "あ")], ctx=ctx)

    passed = captured["ctx"]
    assert passed.synopsis == "A boy reads minds."
    assert passed.glossary[0].jp == "サトリ"


def test_translator_receives_sliding_history(monkeypatch):
    stub = StubTranslator()
    monkeypatch.setattr(translate_mod, "get_translator", lambda _llm, *, ctx=None, **_: stub)

    jp = [_sub(i + 1, float(i), float(i + 1), f"line{i}") for i in range(3)]
    translate_mod.translate_subs(jp)

    assert stub.calls[0][1] == []
    assert stub.calls[1][1] == ["line0"]
    assert stub.calls[2][1] == ["line0", "line1"]
