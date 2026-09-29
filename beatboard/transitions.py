"""Transitions built from the song's own material, for the places a straight splice can't go.

A run in edit.json may carry `fx` (processing over its own audio) and `layers` (sound added over it).
Every position is in beats from the start of the run, so a transition stays on the grid whatever
the tempo. The layers take their material from the song itself: a snare hit lifted from the mix,
the landing bar's first hit reversed. Only the noise of a riser is generated.

  fx      lowpass / highpass   {"path": [[beat, hz], ...]}   the cutoff moves exponentially point to
                               point and holds before the first and after the last, so one filter
                               can close and open again
          gain                 {"path": [[beat, db], ...]}
          mute                 {"beats": [a, b]}              a gap
          level                {"path": [[beat, lu], ...]}    loudness automation, applied after the
                               layers: each beat is brought to the path, in LU against the run's own
                               unprocessed loudness (K-weighted), with the gain smoothed between beat
                               centres. The design says how loud the build is, not each layer.
  layers  riser                {"beats": [a, b], "hz": [from, to], "db": [from, to]}
          roll                 {"beats": [a, b], "hit": {"bar", "step"}, "rate": [from, to], "db": [from, to]}
                               hits per beat from `rate[0]` to `rate[1]`, doubling in steps
          reverse              {"beats": [a, b], "of": {"bar"}, "db": g}  the landing bar's first
                               beats reversed, so they swell into the downbeat
          boom                 {"at": beat, "hz": [from, to], "ms": decay, "db": g}  a sub drop: a
                               sine gliding down, the weight of an impact
          crash                {"at": beat, "hz": cutoff, "ms": decay, "db": g}  a noise wash above
                               `hz`, the air of an impact
          hit                  {"at": beat, "hit": {"bar", "step", "decayMs"?}, "db": g}  one drum hit
                               lifted from the song, the song's own snap (or thump) in an impact
          sweep                {"beats": [a, b], "hz": [from, to], "db": [from, to]}  a pitched riser:
                               a soft saw gliding up (end it on the note the drop lands on)
The song has no impact hits of its own (every section downbeat stands under 1 dB above an ordinary
one), so a drop that pays off a build is made of these three, stacked on the downbeat.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _span(fx: dict, beat_n: int, n: int) -> tuple[int, int]:
    a, b = fx["beats"]
    return max(0, round(a * beat_n)), min(n, round(b * beat_n))


def _db(x):
    return 10 ** (np.asarray(x) / 20)


def _exp_path(a: float, b: float, n: int):
    return np.exp(np.linspace(np.log(a), np.log(b), max(1, n)))


def path_curve(path: list[list[float]], beat_n: float, n: int, log: bool) -> np.ndarray:
    """A value per sample from [beat, value] points: exponential (log) or linear between them, held
    before the first and after the last."""
    xs = np.array([p[0] * beat_n for p in path])
    ys = np.array([p[1] for p in path], dtype=float)
    t = np.arange(n)
    return np.exp(np.interp(t, xs, np.log(ys))) if log else np.interp(t, xs, ys)


def sweep_filter(x: np.ndarray, sr: int, kind: str, cut: np.ndarray) -> np.ndarray:
    """A 2nd-order Butterworth whose cutoff follows `cut` (Hz per sample), stepped every 64 samples
    with its state carried. Skips blocks where the filter would do nothing audible."""
    out = x.copy()
    block = 64
    zi = None
    for i in range(0, x.shape[1], block):
        f = float(cut[i])
        passthrough = (kind == "low" and f >= min(15000, sr / 2 * 0.9)) or (kind == "high" and f <= 12)
        if passthrough:
            zi = None
            continue
        sos = signal.butter(2, min(f, sr / 2 * 0.98), btype=kind, fs=sr, output="sos")
        if zi is None:
            zi = np.stack([signal.sosfilt_zi(sos) * x[c, i] for c in range(x.shape[0])], axis=1)
        seg, zi_new = [], []
        for c in range(x.shape[0]):
            y, z = signal.sosfilt(sos, x[c, i: i + block], zi=zi[:, c])
            seg.append(y)
            zi_new.append(z)
        out[:, i: i + block] = np.stack(seg)
        zi = np.stack(zi_new, axis=1)
    return out


def lift_hit(source: np.ndarray, sr: int, t: float, length_ms: float = 140, decay_ms: float = 55) -> np.ndarray:
    """A drum hit lifted from the mix: the percussive half around `t`, from its attack, gated with a
    short decay so the bleed under it doesn't smear a fast roll. Mono."""
    import librosa

    mono = source.mean(0)
    seg = mono[int((t - 0.03) * sr): int((t + 0.2) * sr)]
    D = librosa.stft(seg, n_fft=1024, hop_length=64)
    _, P = librosa.decompose.hpss(D, margin=2.0)
    p = librosa.istft(P, hop_length=64, length=len(seg))
    env = np.abs(p)
    attack = max(0, int(np.argmax(env > env.max() * 0.5)) - int(0.002 * sr))
    hit = p[attack: attack + int(length_ms / 1000 * sr)]
    t_ms = np.arange(len(hit)) / sr * 1000
    gate = np.exp(-t_ms / decay_ms) * np.minimum(1, t_ms / 1.0)
    hit = hit * gate
    return hit / (np.abs(hit).max() + 1e-9)


def limit(x: np.ndarray, sr: int, ceiling_db: float = -0.3, lookahead_ms: float = 5, release_ms: float = 80) -> tuple[np.ndarray, float]:
    """A look-ahead peak limiter, as a transition bus would have: the gain starts coming down before a
    peak arrives and recovers over `release_ms`. Returns the audio and the most it turned down, in dB."""
    ceiling = 10 ** (ceiling_db / 20)
    peak = np.abs(x).max(0)
    need = np.minimum(1, ceiling / np.maximum(peak, 1e-9))
    la = max(1, int(lookahead_ms / 1000 * sr))
    # The gain a sample needs is the least of what the next `la` samples need (a running minimum).
    padded = np.concatenate([need, np.ones(la)])
    ahead = np.lib.stride_tricks.sliding_window_view(padded, la).min(1)[: len(need)]
    rel = np.exp(-1 / (release_ms / 1000 * sr))
    g = np.empty_like(ahead)
    cur = 1.0
    for i, v in enumerate(ahead):
        cur = v if v < cur else v + (cur - v) * rel
        g[i] = cur
    return x * g[None, :], float(-20 * np.log10(g.min()))


def level_to(x: np.ndarray, ref: np.ndarray, sr: int, beat_n: float, path: list[list[float]], mutes: list[list[float]] = ()) -> np.ndarray:
    """Bring each beat of `x` to `path` (LU against `ref`'s mean loudness, K-weighted). Beats inside
    a mute are the design's silence: they are never measured or lifted, and the gain holds across
    them (a level threshold can't tell a gap from the tail of the fade into it)."""
    from .pipeline import k_weight

    def lu(y):
        return -0.691 + 10 * np.log10((k_weight(y.mean(0), sr) ** 2).mean() + 1e-12)

    base = lu(ref)
    beats = round(x.shape[1] / beat_n)  # a run is whole beats; floor would drop the last to rounding
    centres, gains = [], []
    for i in range(beats):
        if any(m[0] <= i + 0.5 < m[1] for m in mutes):
            continue
        a, b = round(i * beat_n), min(x.shape[1], round((i + 1) * beat_n))
        now = lu(x[:, a:b])
        want = base + float(np.interp(i + 0.5, [p[0] for p in path], [p[1] for p in path]))
        if now < want - 25:
            continue  # a designed silence (a gap, a mute's fade) stays silent: never lift it
        centres.append((a + b) / 2)
        gains.append(np.clip(want - now, -18, 18))
    g = np.interp(np.arange(x.shape[1]), centres, gains)
    return x * _db(g)[None, :]


def apply(run: np.ndarray, sr: int, beat_s: float, piece: dict, song, bar_start) -> np.ndarray:
    """Process one run (channels × samples) by its piece's fx and layers."""
    beat_n = beat_s * sr
    n = run.shape[1]
    out = run.copy()
    for fx in piece.get("fx", []):
        t = fx["type"]
        if t in ("lowpass", "highpass"):
            out = sweep_filter(out, sr, t.replace("pass", ""), path_curve(fx["path"], beat_n, n, log=True))
        elif t == "gain":
            out *= _db(path_curve(fx["path"], beat_n, n, log=False))[None, :]
        elif t == "level":
            continue  # after the layers, below
        elif t == "mute":
            a, b = _span(fx, beat_n, n)
            ramp = min(int(0.004 * sr), b - a)
            out[:, a: a + ramp] *= np.linspace(1, 0, ramp)
            out[:, a + ramp: b] = 0
        else:
            raise ValueError(f"unknown fx {t}")

    rng = np.random.default_rng(142)  # the same noise every run: the edit is reproducible
    for layer in piece.get("layers", []):
        a, b = _span(layer, beat_n, n) if "beats" in layer else (0, n)
        t = layer["type"]
        add = np.zeros(n)
        if t == "riser":
            noise = rng.standard_normal(b - a)
            # A band of noise climbing: a moving band-pass, built as a high-pass then a low-pass sweep.
            m = b - a
            lo = sweep_filter(noise[None, :], sr, "high", _exp_path(layer["hz"][0], layer["hz"][1], m))[0]
            band = sweep_filter(lo[None, :], sr, "low", _exp_path(layer["hz"][0] * 4, min(sr / 2 * 0.85, layer["hz"][1] * 3), m))[0]
            band /= np.abs(band).max() + 1e-9
            add[a:b] = band * _db(np.linspace(layer["db"][0], layer["db"][1], b - a)) ** 1
        elif t == "roll":
            hit = lift_hit(song, sr, bar_start(layer["hit"]["bar"]) + layer["hit"]["step"] * beat_s / 4, decay_ms=layer["hit"].get("decayMs", 55))
            r0, r1 = layer["rate"]
            beats = layer["beats"][1] - layer["beats"][0]
            levels = [r0 * 2 ** k for k in range(int(np.log2(r1 / r0)) + 1)]
            per_level = beats / len(levels)
            pos, gains = [], []
            for k, rate in enumerate(levels):
                start = layer["beats"][0] + k * per_level
                for j in range(int(round(per_level * rate))):
                    pos.append(start + j / rate)
            for i, p in enumerate(pos):
                frac = i / max(1, len(pos) - 1)
                g = _db(layer["db"][0] + (layer["db"][1] - layer["db"][0]) * frac)
                s = round(p * beat_n)
                e = min(n, s + len(hit))
                add[s:e] += hit[: e - s] * g
        elif t == "boom":
            s = round(layer["at"] * beat_n)
            m = min(n - s, int(layer["ms"] / 1000 * sr))
            tt = np.arange(m) / sr
            f = layer["hz"][1] + (layer["hz"][0] - layer["hz"][1]) * np.exp(-tt / 0.09)
            phase = 2 * np.pi * np.cumsum(f) / sr
            env = np.minimum(1, tt / 0.003) * np.exp(-tt / (layer["ms"] / 1000 / 5))
            add[s: s + m] = np.sin(phase) * env * _db(layer["db"])
        elif t == "crash":
            s = round(layer["at"] * beat_n)
            m = min(n - s, int(layer["ms"] / 1000 * sr))
            tt = np.arange(m) / sr
            wash = signal.sosfilt(signal.butter(2, layer["hz"], btype="high", fs=sr, output="sos"), rng.standard_normal(m))
            wash /= np.abs(wash).max() + 1e-9
            env = np.minimum(1, tt / 0.002) * np.exp(-tt / (layer["ms"] / 1000 / 5))
            add[s: s + m] = wash * env * _db(layer["db"])
        elif t == "hit":
            h = lift_hit(song, sr, bar_start(layer["hit"]["bar"]) + layer["hit"]["step"] * beat_s / 4, length_ms=layer["hit"].get("decayMs", 55) * 4, decay_ms=layer["hit"].get("decayMs", 55))
            s = round(layer["at"] * beat_n)
            e = min(n, s + len(h))
            add[s:e] = h[: e - s] * _db(layer["db"])
        elif t == "sweep":
            m = b - a
            f = _exp_path(layer["hz"][0], layer["hz"][1], m)
            ph = 2 * np.pi * np.cumsum(f) / sr
            saw = signal.sawtooth(ph) + 0.5 * signal.sawtooth(ph * 1.005)  # two saws, a hair apart
            saw = signal.sosfilt(signal.butter(2, 3500, btype="low", fs=sr, output="sos"), saw)
            saw /= np.abs(saw).max() + 1e-9
            add[a:b] = saw * _db(np.linspace(layer["db"][0], layer["db"][1], m))
        elif t == "reverse":
            land = bar_start(layer["of"]["bar"])
            length = b - a
            take = song[:, round(land * sr): round(land * sr) + length].mean(0)
            rev = take[::-1] * np.linspace(0, 1, length) ** 2  # swell in, loudest at the downbeat
            add[a:b] = rev * _db(layer["db"])
        else:
            raise ValueError(f"unknown layer {t}")
        out += add[None, :]
    for fx in piece.get("fx", []):
        if fx["type"] == "level":
            mutes = [f["beats"] for f in piece.get("fx", []) if f["type"] == "mute"]
            out = level_to(out, run, sr, beat_n, fx["path"], mutes)
    out, reduced = limit(out, sr)
    apply.last_reduction_db = reduced
    return out
