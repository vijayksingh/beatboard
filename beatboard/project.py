"""A project: one track, its authored inputs, and where the generated files go.

A project is a directory holding beatboard.json. Every path in it is relative to that directory,
and any it leaves out takes the default below, so the smallest project is `{}` beside a song.json
and an edit.json. A film that keeps its music in a subfolder and its cues next to its code says so:

    {
      "fps": 60,
      "paths": { "song": "music/song.json", "edit": "music/edit.json", "cues": "src/cues.generated.json" }
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MANIFEST = "beatboard.json"

DEFAULTS = {
    # authored
    "song": "song.json",  # the source (a URL for fetch) and its section labels by bar
    "edit": "edit.json",  # the cut: bar runs, loops, transitions, gain, mastering
    "listening": "listening.json",  # what a person heard at each splice
    "events": "events.json",  # sounds the picture answers one by one
    # generated
    "analysis": "analysis.json",
    "work": "out",  # source and edit WAVs, spectrograms
    "audio": "out/edit.m4a",  # the finished edit for the picture to play
    "cues": "generated/cues.json",
    "meters": "generated/meters.json",
    "eventsOut": "generated/events.json",
}
FPS = 60  # the picture's frame rate: meters are sampled per frame


def load_project(where: Path) -> tuple[Path, dict, int]:
    """The project's root, its paths (defaults filled in) and its frame rate."""
    root = where.resolve()
    manifest = root / MANIFEST
    if not manifest.exists():
        sys.exit(f"no {MANIFEST} in {root}; create one (it can be just {{}}) or pass --project")
    spec = json.loads(manifest.read_text() or "{}")
    unknown = set(spec.get("paths", {})) - set(DEFAULTS)
    if unknown:
        sys.exit(f"{MANIFEST}: unknown paths {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    return root, {**DEFAULTS, **spec.get("paths", {})}, int(spec.get("fps", FPS))
