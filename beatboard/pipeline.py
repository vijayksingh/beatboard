"""beatboard: read a track into facts a picture can run on, cut an edit of it, and prove both.

A person can't hand a track to a film by ear, so this reads it into numbers:

  fetch     download the source track (song.json "source") into the work dir and decode it to WAV
  analyze   one fixed tempo grid, bar 1, per-bar loudness and bands, 16-step drum grids,
            the key, and where the texture changes -> analysis.json
  edit      cut the bar runs in edit.json into the edit -> the audio file, cues (bars, beats,
            splices), meters (per-frame loudness, bands and drum onsets, for meters in the picture)
            and events (each sound named in events.json, onset by onset)
  check     re-read the finished edit and prove it: one tempo straight through the splices,
            no clicks, under the length cap, builds that build, payoffs that land, cues fresh
  contour   how loud the edit sounds beat by beat between two edit bars (K-weighted, like LUFS)
  master    lay the edit under a silent render and prove the sync to the millisecond
  spectrogram   images of the track and the edit with bar lines and section names, for a
            reviewer (person or model) to read what the numbers can't say
  storyboard    check a storyboard (shots owning bars) covers the edit, and print each shot
            against the sound: seconds, frames, sections, splices, named sounds, loudness, drums

Authored inputs: song.json (source and section labels), edit.json (the cut), listening.json (what a
person heard at each splice), events.json (sounds the picture answers one by one). Where each lives,
and where the generated files go, is the project's beatboard.json (see beatboard/project.py).
Everything generated is never hand-edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Where a project's files live. Set by configure() from the project's beatboard.json before any
# command runs; see beatboard/project.py for the manifest and its defaults.
APP: Path  # the project root, which printed paths are relative to
OUT: Path
SOURCE_WAV: Path
EDIT_WAV: Path  # before mastering: sample-exact against the source, for checks
MASTER_WAV: Path  # what is heard: loudness-normalised and limited
ANALYSIS: Path
SONG: Path
EDIT: Path
LISTENING: Path
PUBLIC_AUDIO: Path
CUES: Path
METERS: Path
EVENTS_SPEC: Path
EVENTS: Path


def configure(root: Path, paths: dict, fps: int) -> None:
    """Point every command at one project: its authored inputs and where the generated files go."""
    global APP, OUT, SOURCE_WAV, EDIT_WAV, MASTER_WAV, ANALYSIS, SONG, EDIT, LISTENING, PUBLIC_AUDIO, CUES, METERS, EVENTS_SPEC, EVENTS, METER_FPS
    APP = root
    OUT = root / paths["work"]
    SOURCE_WAV = OUT / "source.wav"
    EDIT_WAV = OUT / "edit.wav"
    MASTER_WAV = OUT / "edit-master.wav"
    ANALYSIS = root / paths["analysis"]
    SONG = root / paths["song"]
    EDIT = root / paths["edit"]
    LISTENING = root / paths["listening"]
    PUBLIC_AUDIO = root / paths["audio"]
    CUES = root / paths["cues"]
    METERS = root / paths["meters"]
    EVENTS_SPEC = root / paths["events"]
    EVENTS = root / paths["eventsOut"]
    METER_FPS = fps

SR = 22050  # analysis rate; the edit is cut from the source at its own rate
BANDS = {"sub": (20, 120), "low": (120, 500), "mid": (500, 2500), "high": (2500, 11000)}
DRUMS = {"kick": (30, 120), "snare": (1200, 5000), "hat": (7000, 11000)}
METER_FPS = 60  # the frame rate meters are sampled at: the picture's; set by configure()
NOTES = "C C# D D# E F F# G G# A A# B".split()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict, indent: int | None = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def db(x):
    return 20 * np.log10(np.asarray(x) + 1e-7)


def load_mono(path: Path):
    import librosa

    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


# ---------------------------------------------------------------- fetch


def cmd_fetch(_args) -> None:
    song = load_json(SONG)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("source.*"):
        old.unlink()
    # YouTube serves some clients SABR-only streams; walk the clients until one hands over a file.
    for client in ["default", "tv", "web_safari", "ios", "mweb"]:
        extra = [] if client == "default" else ["--extractor-args", f"youtube:player_client={client}"]
        run = subprocess.run(
            ["yt-dlp", "-q", "--no-playlist", "-f", "bestaudio/best", *extra, "-o", str(OUT / "source.%(ext)s"), song["source"]],
            capture_output=True, text=True,
        )
        got = [p for p in OUT.glob("source.*") if p.suffix != ".wav"]
        if got:
            print(f"fetched with the {client} client: {got[0].name}")
            break
        print(f"  {client}: {run.stderr.strip().splitlines()[-1] if run.stderr.strip() else 'no file'}")
    else:
        sys.exit("fetch failed with every client; update yt-dlp or drop a file at <work>/source.<ext>")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(got[0]), "-vn", "-ac", "2", "-ar", "44100", str(SOURCE_WAV)], check=True)
    print(f"decoded -> {SOURCE_WAV.relative_to(APP)}")


# ---------------------------------------------------------------- analysis


def fit_grid(y, bpm_hint: float | None = None) -> dict:
    """One fixed tempo for the whole track: the BPM and phase whose beats sit on the most onset energy.

    Step-sequenced and click-tracked music never changes tempo, so a single grid beats a
    beat tracker, which slips wherever off-beat accents are loud. The drift report says how well it fits.
    """
    import librosa

    hop = 64
    fr = SR / hop
    oe = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    dur = len(y) / SR
    prior = bpm_hint or float(np.atleast_1d(librosa.feature.tempo(onset_envelope=oe, sr=SR, hop_length=hop))[0])

    def scores(bpm, phases):
        per = 60 / bpm
        idx = ((phases[:, None] + per * np.arange(int(dur / per))[None, :]) * fr).astype(int)
        return oe[np.clip(idx, 0, len(oe) - 1)].mean(1)

    best = (-1.0, prior, 0.0)
    for bpm in np.arange(prior - 4, prior + 4, 0.02):
        ph = np.arange(0, 60 / bpm, 0.004)
        s = scores(bpm, ph)
        if s.max() > best[0]:
            best = (float(s.max()), float(bpm), float(ph[s.argmax()]))
    _, bpm0, ph0 = best
    for bpm in np.arange(bpm0 - 0.03, bpm0 + 0.03, 0.002):
        ph = np.arange(ph0 - 0.01, ph0 + 0.01, 0.0005)
        s = scores(bpm, ph)
        if s.max() > best[0]:
            best = (float(s.max()), float(bpm), float(ph[s.argmax()]))
    score, bpm, phase = best
    per = 60 / bpm
    phase %= per

    # Drift: the best local nudge for each 32-beat window. Small everywhere means one grid really fits.
    drift = []
    offs = np.arange(-0.06, 0.0605, 0.002)
    for w in range(0, int(dur / per), 32):
        ts = phase + per * np.arange(w, w + 32)
        ts = ts[ts < dur - 0.1]
        if len(ts) < 8 or oe[(ts * fr).astype(int)].mean() < oe.mean() * 0.5:
            continue  # a tail or a silence has no beats to fit
        sc = [oe[np.clip(((ts + o) * fr).astype(int), 0, len(oe) - 1)].mean() for o in offs]
        drift.append({"t": round(float(ts[0]), 2), "ms": round(float(offs[int(np.argmax(sc))] * 1000))})
    return {"bpm": round(bpm, 3), "beat": per, "phase": phase, "score": round(float(score / oe.mean()), 2), "drift": drift}


def band_frames(y, hop: int):
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    return {k: np.sqrt((S[(f >= lo) & (f < hi)] ** 2).mean(0)) for k, (lo, hi) in BANDS.items()}, S, f


def drum_onsets(y, hop: int):
    """Per-frame onset strength for kick, snare and hat, from the percussive half of the mix."""
    import librosa

    D = librosa.stft(y, n_fft=2048, hop_length=hop)
    _, P = librosa.decompose.hpss(D, margin=2.0)
    Pm = np.abs(P)
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    env = {}
    for k, (lo, hi) in DRUMS.items():
        e = librosa.onset.onset_strength(S=librosa.amplitude_to_db(Pm[(f >= lo) & (f < hi)]), sr=SR, hop_length=hop)
        env[k] = e / (np.percentile(e, 97) * 0.55 + 1e-9)  # 1.0 = a clear hit
    return env


def pick_downbeat(y, beat: float, phase: float) -> int:
    """Which beat of four starts the bar: the phase where bar boundaries line up with the most change.

    Arrangements change on the one, so bars cut at the true downbeat differ most from their neighbours,
    and differ sharply: the same change cut a beat off is smeared across two bar lines.
    """
    hop = 256
    fr = SR / hop
    B, _, _ = band_frames(y, hop)
    raw = np.stack([db(v) for v in B.values()])
    # Silence is quiet, not infinitely different: floor each band 60 dB under its loudest, so a
    # pickup or a silent tail can't outvote the arrangement's changes on the one.
    M = np.maximum(raw, raw.max(1, keepdims=True) - 60)
    n = int((len(y) / SR - phase) / beat)
    beat_of = lambda X, i: X[:, int((phase + i * beat) * fr): int((phase + (i + 1) * beat) * fr)].mean(1)  # noqa: E731
    per_beat = np.stack([beat_of(M, i) for i in range(n)], 1)
    # A beat of digital silence (a pickup, a tail): a bar holding one is left out.
    level = np.array([beat_of(raw, i).mean() for i in range(n)])
    silent = level < level.max() - 80
    novelty = []
    for p in range(4):
        k = (n - p) // 4
        bars = per_beat[:, p: p + 4 * k].reshape(len(B), k, 4).mean(2)
        quiet = silent[p: p + 4 * k].reshape(k, 4).any(1)
        steps = np.linalg.norm(np.diff(bars, axis=1), axis=0)
        steps = steps[~(quiet[:-1] | quiet[1:])]
        # Squared, so change concentrated on a bar line beats the same change smeared across two.
        novelty.append(float((steps ** 2).sum()))
    return int(np.argmax(novelty))


def key_of(y) -> dict:
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=SR).mean(1)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    cands = [(np.corrcoef(np.roll(major, i), chroma)[0, 1], f"{NOTES[i]} major") for i in range(12)]
    cands += [(np.corrcoef(np.roll(minor, i), chroma)[0, 1], f"{NOTES[i]} minor") for i in range(12)]
    cands.sort(reverse=True)
    return {"best": cands[0][1], "confidence": round(float(cands[0][0]), 2), "runnersUp": [c[1] for c in cands[1:3]]}


def cmd_analyze(_args) -> None:
    song = load_json(SONG)
    y = load_mono(SOURCE_WAV)
    dur = len(y) / SR
    grid = fit_grid(y, song.get("bpmHint"))
    beat = grid["beat"]
    bar = 4 * beat
    shift = pick_downbeat(y, beat, grid["phase"])
    t0 = grid["phase"] + shift * beat

    hop = 128
    fr = SR / hop
    B, _, _ = band_frames(y, hop)
    loud = np.sqrt(sum(v ** 2 for v in B.values()) / len(B))
    drums = drum_onsets(y, hop)

    # Bar 1 is the first downbeat whose bar is audible.
    while t0 - bar >= 0:
        t0 -= bar
    ceiling = db(loud).max()

    def bar_stats(a, b):
        ia, ib = int(a * fr), max(int(a * fr) + 1, int(b * fr))
        return {"loud": round(float(db(loud[ia:ib].mean())), 1), **{k: round(float(db(v[ia:ib].mean())), 1) for k, v in B.items()}}

    starts = list(np.arange(t0, dur, bar))
    while starts and bar_stats(starts[0], starts[0] + bar)["loud"] < ceiling - 30:
        starts.pop(0)
    first = float(starts[0])

    def steps(env, a):
        v = []
        for k in range(16):
            c = int((a + k * beat / 4) * fr)
            v.append(float(env[max(0, c - 2): c + 4].max()) if c < len(env) else 0.0)
        return "".join("x" if s > 1 else ("·" if s > 0.6 else "-") for s in v)

    bars = []
    for i, a in enumerate(starts):
        b = min(a + bar, dur)
        bars.append({"n": i + 1, "t": round(float(a), 3), **bar_stats(a, b), "drums": {k: steps(e, a) for k, e in drums.items()}})

    # Where the texture changes: the band profile of the two bars after a line against the two before.
    vec = np.array([[r[k] for k in BANDS] for r in bars])
    change = [0.0] + [float(np.linalg.norm(vec[i: i + 2].mean(0) - vec[max(0, i - 2): i].mean(0))) for i in range(1, len(bars))]
    order = sorted(range(len(bars)), key=lambda i: -change[i])
    changes = sorted(bars[i]["n"] for i in order[:18])

    analysis = {
        "$generated": "beatboard analyze. Do not edit.",
        "source": {"file": SOURCE_WAV.name, "sha": sha(SOURCE_WAV), "seconds": round(dur, 3)},
        "bpm": grid["bpm"],
        "beatSeconds": round(beat, 6),
        "barSeconds": round(bar, 6),
        "firstDownbeat": round(first, 4),
        "gridScore": grid["score"],
        "driftMs": grid["drift"],
        "key": key_of(y),
        "textureChanges": changes,
        "bars": bars,
    }
    write_json(ANALYSIS, analysis)
    worst = max(abs(d["ms"]) for d in grid["drift"])
    print(f"{grid['bpm']} BPM, beat {beat * 1000:.2f} ms, bar 1 at {first:.3f}s, {len(bars)} bars, drift within ±{worst} ms, key {analysis['key']['best']}")
    print(f"texture changes at bars {changes}")


# ---------------------------------------------------------------- edit


def song_bar_start(an: dict, n: int) -> float:
    return an["firstDownbeat"] + (n - 1) * an["barSeconds"]


def section_of(song: dict, n: int) -> str | None:
    for s in song["sections"]:
        if s["bars"][0] <= n <= s["bars"][1]:
            return s["id"]
    return None


def cmd_edit(_args) -> None:
    import soundfile as sf

    an = load_json(ANALYSIS)
    song = load_json(SONG)
    plan = load_json(EDIT)
    audio, sr = sf.read(SOURCE_WAV, always_2d=True)
    # Headroom: the track is mastered to full scale, so anything laid over it needs room.
    audio = audio.T * 10 ** (plan.get("gainDb", 0) / 20)
    total = audio.shape[1] / sr
    xf = int(plan["crossfadeMs"] / 1000 * sr)
    bar = an["barSeconds"]

    # Counted in whole samples, so every run sits exactly where the cues say it does.
    pieces, runs = [], []
    at = 0
    for i, p in enumerate(plan["pieces"]):
        a, b = p["bars"]
        s = 0 if (i == 0 and a == 1) else round(song_bar_start(an, a) * sr)
        e = min(audio.shape[1], round((song_bar_start(an, a) + plan["tailSeconds"]) * sr)) if b == "end" else round(song_bar_start(an, b + 1) * sr)
        times = p.get("times", 1)
        run = audio[:, s: e + xf].copy()
        for _ in range(times - 1):
            # The run again from its start, crossfaded on the bar line: a loop, for a build.
            again = audio[:, s: e + xf]
            ramp = np.linspace(0, 1, xf)
            run[:, -xf:] = run[:, -xf:] * (1 - ramp) + again[:, :xf] * ramp
            run = np.concatenate([run, again[:, xf:]], axis=1)
        e = s + (e - s) * times
        processed = bool(p.get("fx") or p.get("layers"))
        if processed:
            from .transitions import apply

            run = apply(run, sr, an["beatSeconds"], p, audio, lambda n: song_bar_start(an, n))
            print(f"  transition {a}-{b}: the limiter turned it down at most {apply.last_reduction_db:.1f} dB")
        pieces.append(run)
        runs.append({"songBars": [a, b], "times": times, "why": p.get("why"), "section": p.get("section"), "processed": processed,
                     "videoStart": round(at / sr, 6), "songStart": round(s / sr, 6),
                     "seconds": round((e - s) / sr, 6), "videoStartSample": at, "songStartSample": s, "samples": e - s})
        at += e - s
    t_video = at / sr

    out = pieces[0]
    for p in pieces[1:]:
        ramp = np.linspace(0, 1, xf)
        out[:, -xf:] = out[:, -xf:] * (1 - ramp) + p[:, :xf] * ramp
        out = np.concatenate([out, p[:, xf:]], axis=1)
    out = out[:, :at]
    peak = float(np.abs(out).max())
    if peak > 0.999:
        sys.exit(f"the edit clips (peak {peak:.3f}): turn a transition's layers down in edit.json")
    fade = int(plan["tailFadeMs"] / 1000 * sr)
    out[:, -fade:] *= np.linspace(1, 0, fade)
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(EDIT_WAV, out.T, sr)
    mastered = master(out, sr, plan.get("master"))
    sf.write(MASTER_WAV, mastered.T, sr)
    PUBLIC_AUDIO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(MASTER_WAV), "-c:a", "aac", "-b:a", "192k", str(PUBLIC_AUDIO)], check=True)
    duration = out.shape[1] / sr

    # Video bars: the first run keeps its pickup, so video bar 1 sits where song bar 1 did.
    pickup = an["firstDownbeat"] if (plan["pieces"][0]["bars"][0] == 1) else 0.0
    by_n = {r["n"]: r for r in an["bars"]}
    vbars = []
    for run in runs:
        a, b = run["songBars"]
        last = an["bars"][-1]["n"] if b == "end" else b
        for sn in list(range(a, last + 1)) * run.get("times", 1):
            t = pickup + len(vbars) * bar
            if t >= duration:
                break
            src = by_n[sn]
            vbars.append({"n": len(vbars) + 1, "t": round(t, 4), "songBar": sn, "section": run["section"] or section_of(song, sn), "loud": src["loud"], "drums": src["drums"]})
    # A splice is a jump in the song. Runs that simply continue (bar 16 then bar 17) aren't one.
    splices = [
        {"t": r["videoStart"], "sample": r["videoStartSample"], "from": runs[i - 1]["songBars"], "to": r["songBars"], "songLanding": r["songStart"]}
        for i, r in enumerate(runs)
        if i > 0 and r["songBars"][0] != runs[i - 1]["songBars"][1] + 1
    ]
    # A looped run jumps back to its own start: that's a splice too, to hear and to check for clicks.
    for r in runs:
        for k in range(1, r["times"]):
            at_s = r["videoStartSample"] + k * r["samples"] // r["times"]
            splices.append({"t": round(at_s / sr, 6), "sample": at_s, "from": r["songBars"], "to": r["songBars"], "songLanding": r["songStart"], "loop": True})
    splices.sort(key=lambda s: s["t"])

    cues = {
        "$generated": "beatboard edit. Do not edit; change edit.json and re-run.",
        "inputs": {"source": an["source"]["sha"], "edit": sha(EDIT), "song": sha(SONG)},
        "audio": PUBLIC_AUDIO.name,
        "sampleRate": sr,
        "seconds": round(duration, 4),
        "bpm": an["bpm"],
        "beatSeconds": an["beatSeconds"],
        "barSeconds": an["barSeconds"],
        "firstDownbeat": round(pickup, 4),
        "rule": "time of (bar, beat, step) = firstDownbeat + (bar-1)*barSeconds + (beat-1)*beatSeconds + step*beatSeconds/4. Round each event to a frame on its own; never add frame counts.",
        "runs": runs,
        "splices": splices,
        "bars": vbars,
    }
    write_json(CUES, cues)
    write_json(METERS, meters(mastered.mean(0), sr, duration), indent=None)
    if EVENTS_SPEC.exists():
        write_json(EVENTS, find_events(mastered.mean(0), sr, cues), indent=1)
    print(f"edit {duration:.2f}s, {len(vbars)} bars, {len(splices)} splices -> {PUBLIC_AUDIO.relative_to(APP)}, {CUES.relative_to(APP)}, {METERS.relative_to(APP)}")


def find_events(mono, sr: int, cues: dict) -> dict:
    """Each sound in events.json, found in the finished edit: its onsets in video time, the sixteenth
    each sits nearest (and how far off it is), its pitch and its strength (0..1 within the event)."""
    import librosa

    spec = load_json(EVENTS_SPEC)
    y = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=SR)
    hop = 128
    fr = SR / hop
    beat = cues["beatSeconds"]
    out = {"$generated": "beatboard edit, from the project's events spec. Do not edit.", "inputs": sha(EVENTS_SPEC)}
    for ev in spec["events"]:
        a, b = ev["bars"]
        t0 = cues["bars"][a - 1]["t"]
        t1 = cues["bars"][b]["t"] if b < len(cues["bars"]) else cues["seconds"]
        seg = y[int(t0 * SR): int(t1 * SR)]
        H, P = librosa.effects.hpss(seg, margin=2.0)
        part = H if ev.get("part") == "harmonic" else P if ev.get("part") == "percussive" else seg
        S = np.abs(librosa.stft(part, n_fft=1024, hop_length=hop))
        f = librosa.fft_frequencies(sr=SR, n_fft=1024)
        lo, hi = ev["band"]
        on = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[(f >= lo) & (f < hi)]), sr=SR, hop_length=hop)
        peaks = librosa.util.peak_pick(on, pre_max=3, post_max=3, pre_avg=8, post_avg=8, delta=float(on.std()) * ev.get("sensitivity", 0.8), wait=4)
        top = float(on[peaks].max()) if len(peaks) else 1.0
        hits = []
        for k in peaks:
            t = t0 + k / fr
            col = S[:, k: k + 6].mean(1)
            m = (f >= lo * 0.5) & (f < hi)
            pitch = float(f[m][col[m].argmax()])
            steps = (t - cues["firstDownbeat"]) / (beat / 4)
            hits.append({"t": round(t, 4), "bar": int(steps // 16) + 1, "step": int(round(steps)) % 16,
                         "offMs": round((steps - round(steps)) * beat / 4 * 1000, 1), "hz": round(pitch, 1),
                         "note": librosa.hz_to_note(pitch), "strength": round(float(on[k]) / top, 3)})
        out[ev["name"]] = hits
        print(f"  events: {ev['name']}: {len(hits)} in bars {a}-{b}")
    return out


def meters(mono, sr: int, duration: float) -> dict:
    """Per-frame readings of the finished edit, 0..1, for needles, scopes, LEDs and faders in the picture."""
    import librosa

    y = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=SR)
    hop = 128
    fr = SR / hop
    B, _, _ = band_frames(y, hop)
    B["loud"] = np.sqrt(sum(v ** 2 for v in B.values()) / 4)
    drums = drum_onsets(y, hop)
    frames = int(np.ceil(duration * METER_FPS))
    out = {"$generated": "beatboard edit. Do not edit.", "fps": METER_FPS, "frames": frames}
    for k, v in {**B, **drums}.items():
        is_drum = k in DRUMS
        x = v if is_drum else db(v)
        lo, hi = (0.0, 1.5) if is_drum else (np.percentile(x, 2), np.percentile(x, 99.5))
        samples = [x[min(len(x) - 1, int(f / METER_FPS * fr)): min(len(x), int((f + 1) / METER_FPS * fr) + 1)].max() for f in range(frames)]
        out[k] = [round(float(np.clip((s - lo) / (hi - lo), 0, 1)), 3) for s in samples]
    return out


def integrated_lufs(x, sr: int) -> float:
    """Integrated loudness per ITU BS.1770: K-weighted 400 ms blocks (75% overlap), gated at -70 LUFS
    and then 10 LU under the ungated mean."""
    kw = np.stack([k_weight(c, sr) for c in np.atleast_2d(x)])
    size, hop = int(0.4 * sr), int(0.1 * sr)
    z = np.array([(kw[:, i: i + size] ** 2).mean(1).sum() for i in range(0, kw.shape[1] - size, hop)])
    lk = -0.691 + 10 * np.log10(z + 1e-12)
    z = z[lk > -70]
    rel = -0.691 + 10 * np.log10(z.mean()) - 10
    z = z[-0.691 + 10 * np.log10(z) > rel]
    return float(-0.691 + 10 * np.log10(z.mean()))


def master(x, sr: int, spec: dict | None):
    """The last stage: bring the whole edit to a loudness target and hold its peaks under a ceiling.
    The design (the lift of a climax over what came before) lives in the edit; this only sets the
    level it is played at."""
    if not spec:
        return x
    from .transitions import limit

    gain = spec["lufs"] - integrated_lufs(x, sr)
    y, reduced = limit(x * 10 ** (gain / 20), sr, ceiling_db=spec["ceilingDb"], lookahead_ms=5, release_ms=120)
    print(f"  master: {gain:+.1f} dB to {spec['lufs']} LUFS, the limiter turned it down at most {reduced:.1f} dB (ceiling {spec['ceilingDb']} dBFS)")
    return y


# ---------------------------------------------------------------- check


def cmd_check(_args) -> None:
    import soundfile as sf

    an = load_json(ANALYSIS)
    plan = load_json(EDIT)
    cues = load_json(CUES)
    fails = []

    def expect(ok: bool, what: str) -> None:
        print(("  ok    " if ok else "  FAIL  ") + what)
        if not ok:
            fails.append(what)

    worst = max(abs(d["ms"]) for d in an["driftMs"])
    expect(worst <= 15, f"one tempo fits the source: drift within ±{worst} ms (limit 15)")
    expect(cues["inputs"] == {"source": an["source"]["sha"], "edit": sha(EDIT), "song": sha(SONG)}, "cues are fresh for source, edit.json and song.json")
    expect(cues["seconds"] <= plan["maxSeconds"], f"edit is {cues['seconds']:.2f}s (cap {plan['maxSeconds']}s)")

    # The edit, re-read cold: if the splices kept time, one grid still fits it and lands on the cue grid.
    y = load_mono(EDIT_WAV)
    grid = fit_grid(y, an["bpm"])
    # Kept time, exactly: after a splice the edit is the song's own samples, so the first and last
    # second of every run must equal the source at the place the cues say (clear of crossfades and
    # the tail fade). Onset statistics can't prove this: a downbeat after a silence reads earlier
    # than the same downbeat after a fill.
    edit_full, esr = sf.read(EDIT_WAV, always_2d=True)
    src_full, _ = sf.read(SOURCE_WAV, always_2d=True)
    src_full = src_full * 10 ** (plan.get("gainDb", 0) / 20)
    fade_from = cues["seconds"] - plan["tailFadeMs"] / 1000
    worst, where = 0.0, ""
    for r in cues["runs"]:
        if r.get("processed"):
            continue  # a built transition is meant to differ from the song; its place is checked below
        for at in (0.05, r["seconds"] - 1.05):
            end = r["videoStart"] + at + 1
            if at < 0 or end > fade_from:
                continue
            k = round(at * esr)
            a, b = r["videoStartSample"] + k, r["songStartSample"] + k
            probe, ref = edit_full[a: a + esr], src_full[b: b + esr]
            err = float(np.sqrt(((probe - ref) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-9))
            if err > worst:
                worst, where = err, f" (run {r['songBars']}, {at:.2f}s in)"
    expect(worst <= 1e-3, f"every run is the song's own samples, exactly where the cues say (worst residual {worst:.1e}{where})")
    splits = [abs(((s["t"] - cues["firstDownbeat"]) / cues["barSeconds"] + 0.5) % 1 - 0.5) * cues["barSeconds"] * 1000 for s in cues["splices"]]
    expect(max(splits, default=0) <= 1, f"every splice falls on a bar line (worst {max(splits, default=0):.2f} ms off)")
    beat = cues["beatSeconds"]
    off = ((grid["phase"] - cues["firstDownbeat"]) / beat) % 1 * beat
    off = min(off, beat - off) * 1000
    expect(off <= 10, f"the edit's beats sit on the cue grid (off by {off:.1f} ms)")

    # A click is a sample jump the music doesn't have: compare each splice with the source's own
    # jump where the edit lands (a drop's hit is loud in both; a bad cut is loud only in the edit).
    edit, sr = sf.read(EDIT_WAV, always_2d=True)
    source, _ = sf.read(SOURCE_WAV, always_2d=True)
    w = int(0.005 * sr)

    def jump(audio, t):
        i = int(t * sr)
        return float(np.abs(np.diff(audio[max(0, i - w): i + w].mean(1))).max())

    for s in cues["splices"]:
        j, natural = jump(edit, s["t"]), jump(source, s["songLanding"])
        expect(j <= natural * 1.25 + 0.01, f"no click at the {s['from']}→{s['to']} splice ({s['t']:.2f}s): jump {j:.3f}, the song's own {natural:.3f}")

    # A built transition keeps its declared shape, measured the way the ear hears (K-weighted):
    # a dip below where it started, then a climb that rises and arrives near the bar it lands on.
    y_edit, _ = sf.read(EDIT_WAV)
    y_edit = y_edit.mean(1)
    for i, r in enumerate(cues["runs"]):
        exp = next((p.get("expect") for p in plan["pieces"] if p["bars"] == r["songBars"] and p.get("expect")), None)
        if not exp:
            continue
        beats = round(r["seconds"] / cues["beatSeconds"])
        lk = beat_loudness(y_edit, esr, r["videoStart"], cues["beatSeconds"], beats)
        name = f"transition {r['songBars'][0]}-{r['songBars'][1]}"
        if "dip" in exp:
            a, b = exp["dip"]["beats"]
            depth = float(np.mean(lk[a:b])) - lk[0]
            expect(depth <= exp["dip"]["below"], f"{name} dips {depth:+.1f} LU below where it starts (wants {exp['dip']['below']})")
        if "climb" in exp:
            a, b = exp["climb"]["beats"]
            rise = lk[b - 1] - min(lk[a:a + 2])
            expect(rise >= exp["climb"]["rise"], f"{name} climbs {rise:+.1f} LU through its build (wants +{exp['climb']['rise']})")

    # A gap needs a payoff. A run that ends in silence builds anticipation; the run after it must
    # declare a payoff and meet it. What makes a drop hit, measured:
    #   weight  the low end (30-250 Hz) crashes back: the landing's first beat against the build's
    #           last two beats. A build that keeps its bass leaves the drop nothing to bring.
    #   slam    the drop's first beat against the build's last sounding beat, K-weighted: what a
    #           laptop or a phone plays. (A bass return alone doesn't reach small speakers.)
    #   build   the rising layers run at least four bars: anticipation takes time
    #   lift    the landing section sits louder than the drop it answers, K-weighted.
    from scipy import signal as _sig

    low = _sig.sosfilt(_sig.butter(4, [30, 250], btype="band", fs=esr, output="sos"), y_edit)
    kw = k_weight(y_edit, esr)

    def level(x, t: float, dur: float) -> float:
        s = x[int(t * esr): int((t + dur) * esr)]
        return float(10 * np.log10((s ** 2).mean() + 1e-12))

    def run_of(bars):
        return next((r for r in cues["runs"] if r["songBars"] == bars), None)

    bs = cues["beatSeconds"]
    for i, r in enumerate(cues["runs"]):
        piece = next(p for p in plan["pieces"] if p["bars"] == r["songBars"])
        beats = round(r["seconds"] / bs)
        gap = any(f["type"] == "mute" and f["beats"][1] >= beats for f in piece.get("fx", []))
        if not gap or i + 1 >= len(cues["runs"]):
            continue
        land = cues["runs"][i + 1]
        pay = next((p.get("expect", {}).get("payoff") for p in plan["pieces"] if p["bars"] == land["songBars"]), None)
        name = f"the landing after the {r['songBars'][0]}-{r['songBars'][1]} gap"
        expect(pay is not None, f"{name} declares a payoff (a gap needs one)")
        if not pay:
            continue
        t0 = land["videoStart"]
        weight = level(low, t0, bs) - level(low, t0 - 3 * bs, 2 * bs)
        expect(weight >= pay["weight"], f"{name} brings the weight back {weight:+.1f} dB over the end of the build (wants +{pay['weight']})")
        # Slam: what a laptop or a phone plays (K-weighted, which drops the sub): the drop's first beat
        # against the build's last sounding beat. A build that ends as loud as the drop has none.
        gap_beats = sum(f["beats"][1] - f["beats"][0] for f in piece.get("fx", []) if f["type"] == "mute")
        before = beat_loudness(y_edit, esr, t0 - (gap_beats + 1) * bs, bs, 1)[0]
        slam = beat_loudness(y_edit, esr, t0, bs, 1)[0] - before
        expect(slam >= pay["slam"], f"{name} slams {slam:+.1f} LU over the build's last beat (wants +{pay['slam']})")
        # Anticipation takes time: the build's rising layers must run at least four bars into the gap.
        rising = [l for l in piece.get("layers", []) if l["type"] in ("riser", "roll", "sweep") and "beats" in l]
        span = (max(l["beats"][1] for l in rising) - min(l["beats"][0] for l in rising)) if rising else 0
        expect(span >= 16, f"{name} is anticipated by a {span / 4:g}-bar build (wants 4 or more)")
        over = run_of(pay["lift"]["over"])
        if over:
            mine = np.mean(beat_loudness(y_edit, esr, t0, bs, 16))
            theirs = np.mean(beat_loudness(y_edit, esr, over["videoStart"], bs, 16))
            expect(mine - theirs >= pay["lift"]["by"], f"{name} sits {mine - theirs:+.1f} LU over the drop it answers (wants +{pay['lift']['by']})")

    # The ear: whether a splice lands musically is a person's call (a peak cut straight into a riser
    # measures smooth and sounds wrong). listening.json holds what was heard.
    verdicts = load_json(LISTENING)["verdicts"] if LISTENING.exists() else []
    now = sha(EDIT)
    unheard = []
    for s in cues["splices"]:
        key = f"{s['from'][1]}→{s['to'][0]}"
        mine = [v for v in verdicts if v["splice"] == key]
        if not mine:
            unheard.append(f"{key} at {s['t']:.2f}s")
            continue
        if any(v["verdict"] == "ok" and v.get("edit") == now for v in mine):
            mine = [v for v in mine if v.get("edit") == now]  # heard as ok in this edit: older verdicts are history
        for v in mine:
            bad = v["verdict"] != "ok"
            if bad and v.get("edit") in (None, now):
                expect(False, f"the {key} splice was heard as {v['verdict']} in this very edit ({v['by']}, {v['date']})")
            elif bad:
                unheard.append(f"{key} at {s['t']:.2f}s (heard as {v['verdict']} before the last change)")
            else:
                print(f"  ok    the {key} splice was heard as ok ({v['by']}, {v['date']})")
    if unheard:
        print(f"  listen  not yet heard by a person: {', '.join(unheard)}; log a verdict in listening.json")

    if fails:
        sys.exit(f"{len(fails)} check(s) failed")
    print("all checks pass")


# ---------------------------------------------------------------- master


def decode(path: Path, rate: int = 8000):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def av_offset_ms(video: Path) -> float:
    """How late the video's soundtrack runs against the edit, by cross-correlation (0.125 ms steps)."""
    import scipy.signal as sig

    a, b = decode(MASTER_WAV), decode(video)
    n = min(len(a), len(b))
    c = sig.correlate(b[:n], a[:n], mode="full", method="fft")
    return (int(c.argmax()) - (n - 1)) / 8.0


def cmd_master(args) -> None:
    """Lay the edit under a silent render. Remotion's own AAC mux runs 2048 samples late (two
    encoder-priming frames it doesn't record), so the picture renders muted and ffmpeg adds the
    sound, writing the priming into the file where players skip it. Then prove the sync."""
    video, out = Path(args.video).resolve(), Path(args.out).resolve()
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-i", str(MASTER_WAV), "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(out)],
        check=True,
    )
    off = av_offset_ms(out)
    print(f"{out.relative_to(APP)}: soundtrack offset {off:+.2f} ms")
    if abs(off) > 1:
        sys.exit("the soundtrack is out of sync by more than 1 ms")


# ---------------------------------------------------------------- contour


def k_weight(y, sr: int):
    """ITU BS.1770 K-weighting (the stage before LUFS): a high shelf of about +4 dB above 1.5 kHz and a
    high-pass near 38 Hz, so loudness reads the way the ear hears it rather than how much sub it has."""
    from scipy import signal

    # Pre-filter (high shelf), designed at the given rate from the standard's analog prototype.
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh, vb = 10 ** (g / 20), 10 ** (g / 20) ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    b1 = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    a1 = [1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    # RLB (high-pass).
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    a2 = [1, 2 * (k * k - 1) / (1 + k / q + k * k), (1 - k / q + k * k) / (1 + k / q + k * k)]
    return signal.lfilter([1, -2, 1], a2, signal.lfilter(b1, a1, y))


def beat_loudness(y, sr: int, t0: float, beat: float, beats: int) -> list[float]:
    """K-weighted loudness of each beat from t0, in LU-like dB (mean square, -0.691 offset)."""
    kw = k_weight(y, sr)
    out = []
    for i in range(beats):
        seg = kw[int((t0 + i * beat) * sr): int((t0 + (i + 1) * beat) * sr)]
        out.append(round(float(-0.691 + 10 * np.log10((seg ** 2).mean() + 1e-12)), 1))
    return out


def cmd_contour(args) -> None:
    """How loud the edit sounds, beat by beat, between two video bars: for shaping transitions."""
    import soundfile as sf

    cues = load_json(CUES)
    y, sr = sf.read(EDIT_WAV)
    y = y.mean(1) if y.ndim > 1 else y
    first = cues["bars"][args.first - 1]
    beats = (args.last - args.first + 1) * 4
    lk = beat_loudness(y, sr, first["t"], cues["beatSeconds"], beats)
    for i, v in enumerate(lk):
        bar = args.first + i // 4
        sec = cues["bars"][bar - 1]["section"] if i % 4 == 0 else ""
        print(f"  {bar}.{i % 4 + 1}  {v:6.1f}  {'#' * max(0, int(v + 40))}  {sec or ''}")


# ---------------------------------------------------------------- spectrogram


def cmd_spectrogram(args) -> None:
    import librosa
    import librosa.display
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    an = load_json(ANALYSIS)
    song = load_json(SONG)
    if args.which == "edit":
        cues = load_json(CUES)
        y = load_mono(EDIT_WAV)
        lines = [(b["t"], f"{b['n']}" + (f" {b['section']}" if b["section"] and (b["n"] == 1 or cues["bars"][b["n"] - 2]["section"] != b["section"]) else "")) for b in cues["bars"]]
        marks = [s["t"] for s in cues["splices"]]
    else:
        y = load_mono(SOURCE_WAV)
        starts = {s["bars"][0]: s["id"] for s in song["sections"]}
        lines = [(b["t"], f"{b['n']}" + (f" {starts[b['n']]}" if b["n"] in starts else "")) for b in an["bars"]]
        marks = []
    dur = len(y) / SR
    M = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128, hop_length=256, fmax=11000), ref=np.max)
    dest = OUT / "spectrograms"
    dest.mkdir(parents=True, exist_ok=True)
    span = args.seconds
    for i, a in enumerate(np.arange(0, dur, span)):
        fig, ax = plt.subplots(figsize=(22, 6), dpi=80)
        librosa.display.specshow(M, sr=SR, hop_length=256, x_axis="time", y_axis="mel", fmax=11000, ax=ax, cmap="magma")
        ax.set_xlim(a, min(dur, a + span))
        for j, (t, label) in enumerate(lines):
            if a <= t <= a + span:
                ax.axvline(t, color="cyan", lw=1.2 if j % 4 == 0 else 0.4, alpha=0.8)
                ax.text(t + 0.05, 9500, label, color="white", fontsize=11, weight="bold")
        for t in marks:
            if a <= t <= a + span:
                ax.axvline(t, color="lime", lw=2.5)
        plt.tight_layout()
        path = dest / f"{args.which}-{i:02d}.png"
        plt.savefig(path)
        plt.close()
        print(path.relative_to(APP))


def main() -> None:
    from .project import load_project

    ap = argparse.ArgumentParser(prog="beatboard", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", "-C", default=".", help="the project directory (holds beatboard.json); default: here")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    sub.add_parser("analyze").set_defaults(fn=cmd_analyze)
    sub.add_parser("edit").set_defaults(fn=cmd_edit)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    mp = sub.add_parser("master")
    mp.add_argument("video", help="a silent render")
    mp.add_argument("out")
    mp.set_defaults(fn=cmd_master)
    cp = sub.add_parser("contour")
    cp.add_argument("first", type=int, help="first video bar")
    cp.add_argument("last", type=int, help="last video bar")
    cp.set_defaults(fn=cmd_contour)
    sp = sub.add_parser("spectrogram")
    sp.add_argument("which", choices=["source", "edit"])
    sp.add_argument("--seconds", type=float, default=24)
    sp.set_defaults(fn=cmd_spectrogram)
    from .storyboard import cmd_storyboard

    sb = sub.add_parser("storyboard", help="check a storyboard covers the edit, and print each shot against the sound")
    sb.add_argument("file", nargs="?", default="storyboard.json", help="the storyboard (default storyboard.json)")
    sb.add_argument("--out", help="also write the sheet as JSON here")
    sb.set_defaults(fn=cmd_storyboard)
    args = ap.parse_args()
    root, paths, fps = load_project(Path(args.project))
    configure(root, paths, fps)
    for tool in ["ffmpeg"] + (["yt-dlp"] if args.cmd == "fetch" else []):
        if not shutil.which(tool):
            sys.exit(f"{tool} is missing (brew install {tool})")
    args.fn(args)


if __name__ == "__main__":
    main()
