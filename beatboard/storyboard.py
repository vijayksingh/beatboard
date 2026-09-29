"""Storyboard against the sound.

A storyboard is the picture's plan in musical time: shots that each own a run of the edit's bars.
This checks it covers the edit (every bar in exactly one shot, in order, none past the end) and
prints the sheet a person or a model plans motion from: for each shot, where it sits in seconds and
frames, the song's sections under it, the splices it has to carry, the named sounds that fire in it
(from events.json), how loud it plays, and the drums of its first bar.

storyboard.json:

    { "shots": [ { "id": "opener", "bars": [1, 4], "title": "The hook", "picture": "...", "sync": "..." }, ... ] }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import pipeline as P


def load_storyboard(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"no storyboard at {path}; write one (see beatboard/storyboard.py)")
    shots = json.loads(path.read_text()).get("shots", [])
    for s in shots:
        if not (isinstance(s.get("id"), str) and isinstance(s.get("bars"), list) and len(s["bars"]) == 2):
            sys.exit(f"storyboard: every shot needs an id and bars [first, last]; got {s}")
    return shots


def coverage(shots: list[dict], n_bars: int) -> list[str]:
    """What is wrong with how the shots cover the edit's bars: gaps, overlaps, order, overruns."""
    wrong: list[str] = []
    owners: dict[int, list[str]] = {}
    for s in shots:
        a, b = s["bars"]
        if a > b:
            wrong.append(f"{s['id']} runs backwards: bars {a}-{b}")
        if b > n_bars:
            wrong.append(f"{s['id']} runs to bar {b}; the edit has {n_bars}")
        for bar in range(a, b + 1):
            owners.setdefault(bar, []).append(s["id"])
    for bar in range(1, n_bars + 1):
        who = owners.get(bar, [])
        if not who:
            wrong.append(f"bar {bar} is in no shot")
        elif len(who) > 1:
            wrong.append(f"bar {bar} is in {len(who)} shots: {', '.join(who)}")
    starts = [s["bars"][0] for s in shots]
    if starts != sorted(starts):
        wrong.append("shots are out of order")
    return wrong


def sheet(shots: list[dict], cues: dict, events: dict | None, fps: int) -> list[dict]:
    """Each shot against the sound, as data."""
    bars = cues["bars"]
    end_of = lambda b: bars[b]["t"] if b < len(bars) else cues["seconds"]  # noqa: E731 (bar b is 1-based; bars[b] is the next one)
    rows = []
    for s in shots:
        a, b = s["bars"]
        t0 = 0.0 if a == 1 else bars[a - 1]["t"]
        t1 = end_of(b)
        under = bars[a - 1:b]
        sections: list[str] = []
        for bar in under:
            if bar.get("section") and bar["section"] not in sections:
                sections.append(bar["section"])
        sounds = {}
        for name, hits in (events or {}).items():
            if isinstance(hits, list):
                k = sum(1 for h in hits if isinstance(h, dict) and a <= h.get("bar", 0) <= b)
                if k:
                    sounds[name] = k
        rows.append({
            "id": s["id"], "bars": [a, b], "title": s.get("title", ""),
            "seconds": [round(t0, 3), round(t1, 3)], "frames": [round(t0 * fps), round(t1 * fps)],
            "sections": sections,
            "splices": [round(x["t"], 3) for x in cues.get("splices", []) if t0 <= x["t"] < t1],
            "sounds": sounds,
            "loud": round(sum(bar.get("loud", 0) for bar in under) / max(1, len(under)), 1),
            "drums": under[0].get("drums", {}) if under else {},
        })
    return rows


def cmd_storyboard(args) -> None:
    cues = P.load_json(P.CUES)
    events = P.load_json(P.EVENTS) if P.EVENTS.exists() else None
    shots = load_storyboard(P.APP / args.file)
    wrong = coverage(shots, len(cues["bars"]))
    rows = sheet(shots, cues, events, P.METER_FPS)
    for r in rows:
        print(f"{r['id']:<16} bars {r['bars'][0]:>2}-{r['bars'][1]:<2}  {r['seconds'][0]:6.2f}-{r['seconds'][1]:6.2f}s  "
              f"frames {r['frames'][0]:>5}-{r['frames'][1]:<5}  loud {r['loud']:5.1f}  {', '.join(r['sections'])}")
        if r["title"]:
            print(f"{'':<18}{r['title']}")
        if r["splices"]:
            print(f"{'':<18}splices at {', '.join(f'{t:.2f}s' for t in r['splices'])}")
        if r["sounds"]:
            print(f"{'':<18}sounds: {', '.join(f'{k} ×{v}' for k, v in r['sounds'].items())}")
        for drum, grid in r["drums"].items():
            print(f"{'':<18}{drum:<6}{grid}")
    if args.out:
        out = P.APP / args.out
        P.write_json(out, {"$generated": "beatboard storyboard. Do not edit.", "fps": P.METER_FPS, "shots": rows})
        print(f"-> {out.relative_to(P.APP)}")
    if wrong:
        print("\n".join(f"  FAIL  {w}" for w in wrong))
        sys.exit(1)
    print(f"storyboard: {len(shots)} shots cover all {len(cues['bars'])} bars once")
