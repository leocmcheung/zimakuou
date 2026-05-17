from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import srt

from zimakuou import batch as batch_mod
from zimakuou.context import Context


def _sub(i: int, text: str) -> srt.Subtitle:
    return srt.Subtitle(
        index=i, start=timedelta(seconds=i), end=timedelta(seconds=i + 1), content=text
    )


def _stub_video(path: Path) -> Path:
    """Real videos are huge — for the batch unit tests we just want a path
    that .exists(), since extract_audio is mocked."""
    path.write_bytes(b"fake-mp4")
    return path


def test_phase1_extracts_each_video_once(tmp_path, monkeypatch):
    v1 = _stub_video(tmp_path / "ep1.mp4")
    v2 = _stub_video(tmp_path / "ep2.mp4")
    extracted = []

    def fake_extract(video, wav):
        extracted.append((video.name, wav.name))
        wav.write_bytes(b"fake-wav")
        return wav

    monkeypatch.setattr(batch_mod, "extract_audio", fake_extract)

    batch_mod._extract_all([v1, v2], tmp_path, force=False)
    assert extracted == [("ep1.mp4", "ep1.wav"), ("ep2.mp4", "ep2.wav")]


def test_phase1_skips_existing_wavs(tmp_path, monkeypatch):
    """Resumability: if .wav exists, don't re-extract."""
    v = _stub_video(tmp_path / "ep1.mp4")
    (tmp_path / "ep1.wav").write_bytes(b"already there")
    calls = []
    monkeypatch.setattr(batch_mod, "extract_audio", lambda *a: calls.append(a))

    batch_mod._extract_all([v], tmp_path, force=False)
    assert calls == []


def test_phase1_force_redoes_existing(tmp_path, monkeypatch):
    v = _stub_video(tmp_path / "ep1.mp4")
    (tmp_path / "ep1.wav").write_bytes(b"stale")
    called = []
    monkeypatch.setattr(
        batch_mod,
        "extract_audio",
        lambda video, wav: (called.append(wav), wav.write_bytes(b"fresh"))[1],
    )

    batch_mod._extract_all([v], tmp_path, force=True)
    assert len(called) == 1


def test_phase2_deletes_wav_after_transcribe(tmp_path, monkeypatch):
    v = _stub_video(tmp_path / "ep1.mp4")
    wav = tmp_path / "ep1.wav"
    wav.write_bytes(b"audio")

    monkeypatch.setattr(
        batch_mod, "transcribe", lambda *a, **kw: [_sub(1, "こんにちは")]
    )

    batch_mod._transcribe_all([v], tmp_path, None, Context.empty(), force=False)

    assert (tmp_path / "ep1.jp.srt").exists()
    # WAVs are intermediate — phase 2 must delete them once .jp.srt is written
    # so we don't leave ~80 MB/episode lying around.
    assert not wav.exists()


def test_phase2_skips_when_jp_srt_exists_and_cleans_stale_wav(tmp_path, monkeypatch):
    v = _stub_video(tmp_path / "ep1.mp4")
    (tmp_path / "ep1.jp.srt").write_text(srt.compose([_sub(1, "x")]), encoding="utf-8")
    wav = tmp_path / "ep1.wav"
    wav.write_bytes(b"stale leftover from killed earlier run")

    called = []
    monkeypatch.setattr(
        batch_mod, "transcribe", lambda *a, **kw: called.append(a) or []
    )

    batch_mod._transcribe_all([v], tmp_path, None, Context.empty(), force=False)
    assert called == []  # didn't re-transcribe
    assert not wav.exists()  # but did clean up the stale wav


def test_phase3_loads_translator_once_for_all_files(tmp_path, monkeypatch):
    """The whole point of batch mode: avoid reloading the LLM per episode."""
    v1 = _stub_video(tmp_path / "ep1.mp4")
    v2 = _stub_video(tmp_path / "ep2.mp4")
    (tmp_path / "ep1.jp.srt").write_text(
        srt.compose([_sub(1, "a")]), encoding="utf-8"
    )
    (tmp_path / "ep2.jp.srt").write_text(
        srt.compose([_sub(1, "b")]), encoding="utf-8"
    )

    get_translator_calls = []

    def fake_get_translator(*args, **kwargs):
        get_translator_calls.append(kwargs)
        t = MagicMock()
        t.translate.return_value = "譯文"
        return t

    monkeypatch.setattr(batch_mod, "get_translator", fake_get_translator)

    batch_mod._translate_all([v1, v2], tmp_path, None, Context.empty(), force=False)

    assert len(get_translator_calls) == 1, "translator must be built once, not per file"
    assert (tmp_path / "ep1.zh-tw.srt").exists()
    assert (tmp_path / "ep2.zh-tw.srt").exists()


def test_phase3_skips_when_outputs_exist(tmp_path, monkeypatch):
    v = _stub_video(tmp_path / "ep1.mp4")
    (tmp_path / "ep1.jp.srt").write_text(
        srt.compose([_sub(1, "a")]), encoding="utf-8"
    )
    (tmp_path / "ep1.zh-tw.srt").write_text("existing", encoding="utf-8")
    (tmp_path / "ep1.bilingual.srt").write_text("existing", encoding="utf-8")

    called = []
    monkeypatch.setattr(
        batch_mod,
        "get_translator",
        lambda *a, **kw: called.append("loaded") or MagicMock(),
    )

    batch_mod._translate_all([v], tmp_path, None, Context.empty(), force=False)
    # Translator should never be loaded if nothing needs translating —
    # 8.4 GB of pointless disk I/O otherwise.
    assert called == []


def test_paths_are_derived_from_video_stem_not_full_path(tmp_path):
    """Videos may live on NAS paths with deep nesting; outputs must use
    just the stem so they all land in out_dir at the top level."""
    v = Path("/Volumes/NAS/some/deep/path/My Show S01E01.mp4")
    wav, jp, zh, bi, draft = batch_mod._local_paths(v, tmp_path)
    assert wav == tmp_path / "My Show S01E01.wav"
    assert jp == tmp_path / "My Show S01E01.jp.srt"
    assert zh == tmp_path / "My Show S01E01.zh-tw.srt"
    assert bi == tmp_path / "My Show S01E01.bilingual.srt"
    assert draft == tmp_path / "My Show S01E01.context.draft.yaml"
