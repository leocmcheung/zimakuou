import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from zimakuou.audio import extract_audio

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def _make_silent_video(path: Path, seconds: int = 1) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={seconds}",
            "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono",
            "-shortest",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_extract_audio_writes_16khz_mono_wav(tmp_path: Path):
    video = tmp_path / "silent.mp4"
    _make_silent_video(video, seconds=1)

    out = tmp_path / "out.wav"
    extract_audio(video, out)

    assert out.exists()
    assert out.stat().st_size > 0
    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
