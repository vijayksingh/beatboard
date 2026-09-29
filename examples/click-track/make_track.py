"""Synthesise the example's track: 16 bars at 128 BPM after a half-second pickup, built up the way
arrangements are, on the one: bars 1-2 the kick alone, 3-4 add hats, 5-6 add the snare, 7-8 the hats
go to sixteenths, 9-16 add a bassline. Every fact beatboard should find in it is known: the tempo,
where bar 1 is, the drum grids, and the texture change at bar 9. Writes <work>/source.wav."""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BPM, PICKUP, BARS, SR = 128.0, 0.5, 16, 44100
BEAT = 60 / BPM


def main(out: Path) -> None:
    n = int((PICKUP + BARS * 4 * BEAT + 1.0) * SR)
    y = np.zeros(n)
    rng = np.random.default_rng(7)

    def put(t: float, sound: np.ndarray) -> None:
        i = int(round(t * SR))
        y[i:i + len(sound)] += sound[: max(0, n - i)]

    t = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-t * 30)) * t) * np.exp(-t * 14)
    snare = rng.standard_normal(len(t)) * np.exp(-t * 22) * 0.5
    th = np.arange(int(0.05 * SR)) / SR
    hat = np.diff(rng.standard_normal(len(th) + 1)) * np.exp(-th * 90) * 0.25
    for bar in range(BARS):
        for beat in range(4):
            at = PICKUP + (bar * 4 + beat) * BEAT
            put(at, kick)
            if bar >= 4 and beat in (1, 3):
                put(at, snare)
            if bar >= 2:
                for k in range(4 if bar >= 6 else 2):
                    put(at + k * BEAT / (4 if bar >= 6 else 2), hat)
        if bar >= 8:  # the bassline: one held note a bar
            tb = np.arange(int(4 * BEAT * SR)) / SR
            put(PICKUP + bar * 4 * BEAT, 0.3 * np.sin(2 * np.pi * [55, 65.4, 49, 58.3][bar % 4] * tb) * np.minimum(1, (4 * BEAT - tb) * 20))
    y /= np.abs(y).max() * 1.12
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "source.wav", np.stack([y, y], 1), SR)
    print(f"-> {out / 'source.wav'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "out")
