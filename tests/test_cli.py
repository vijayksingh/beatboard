"""Feature slices: the beatboard command, run end to end on a synthetic project whose every fact is
known (examples/click-track: 128 BPM, a half-second pickup, drums built up on the one, a bassline
from bar 9). Each test drives the real CLI on a fresh copy and reads what it wrote."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "click-track"


def bb(project: Path, *args: str, ok: bool = True) -> str:
    run = subprocess.run([sys.executable, "-m", "beatboard", "-C", str(project), *args], capture_output=True, text=True)
    if ok and run.returncode != 0:
        raise AssertionError(f"beatboard {' '.join(args)} failed:\n{run.stdout}\n{run.stderr}")
    return run.stdout + run.stderr


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    """A fresh copy of the example with its track synthesised, analysed and cut."""
    root = tmp_path_factory.mktemp("click-track")
    for f in EXAMPLE.iterdir():
        if f.is_file():
            shutil.copy(f, root / f.name)
    subprocess.run([sys.executable, str(root / "make_track.py"), str(root / "out")], check=True, capture_output=True)
    bb(root, "analyze")
    bb(root, "edit")
    return root


def read(project: Path, rel: str) -> dict:
    return json.loads((project / rel).read_text())


def test_analyze_finds_the_tempo_and_bar_one(project):
    a = read(project, "analysis.json")
    assert abs(a["bpm"] - 128) < 0.05
    assert abs(a["firstDownbeat"] - 0.5) < 0.02  # the pickup is skipped: bar 1 is the first downbeat
    assert max(abs(d["ms"]) for d in a["driftMs"]) <= 10


def test_analyze_reads_the_drums_on_the_grid(project):
    bars = read(project, "analysis.json")["bars"]
    assert bars[0]["drums"]["kick"][::4] == "xxxx"  # the kick on every beat
    # the bass comes in at bar 9: the sub band rises there
    assert bars[8]["sub"] - bars[7]["sub"] > 6


def test_edit_cuts_on_bar_lines_and_exports_cues(project):
    cues = read(project, "generated/cues.json")
    assert len(cues["bars"]) == 8  # two four-bar runs
    assert len(cues["splices"]) == 1
    splice, bar5 = cues["splices"][0]["t"], cues["bars"][4]["t"]
    assert abs(splice - bar5) < 0.001  # the splice is bar 5's line
    meters = read(project, "generated/meters.json")
    assert meters["fps"] == 60 and meters["frames"] == -(-cues["seconds"] * 60 // 1)


def test_edit_finds_the_named_sounds(project):
    hats = read(project, "generated/events.json")["hats"]
    assert 12 <= len(hats) <= 16  # eighth-note hats over bars 3-4 of the song: 16 of them
    assert all(3 <= h["bar"] <= 4 for h in hats)


def test_check_proves_the_edit(project):
    out = bb(project, "check")
    assert "all checks pass" in out
    assert "every splice falls on a bar line" in out


def test_storyboard_covers_the_edit(project):
    out = bb(project, "storyboard", "--out", "generated/storyboard.json")
    assert "2 shots cover all 8 bars once" in out
    shots = read(project, "generated/storyboard.json")["shots"]
    assert shots[1]["sections"] == ["drums-and-bass"]
    assert shots[1]["frames"][0] == round(read(project, "generated/cues.json")["bars"][4]["t"] * 60)


def test_storyboard_fails_on_a_gap(project, tmp_path):
    board = {"shots": [{"id": "a", "bars": [1, 3]}, {"id": "b", "bars": [5, 8]}]}
    (project / "gappy.json").write_text(json.dumps(board))
    out = bb(project, "storyboard", "gappy.json", ok=False)
    assert "bar 4 is in no shot" in out


def test_master_proves_the_sync(project):
    seconds = read(project, "generated/cues.json")["seconds"]
    silent = project / "out" / "silent.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=white:s=320x180:r=60:d={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)], check=True)
    out = bb(project, "master", str(silent), str(project / "out" / "film.mp4"))
    assert "soundtrack offset +0.00 ms" in out
