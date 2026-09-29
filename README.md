# beatboard

Cut a picture to music by reading the music first.

A person can't hand a track to an animation by ear, and a beat tracker that drifts a few milliseconds is a cut that feels late. beatboard reads a track into facts a picture can run on: one exact tempo grid, where bar 1 is, what every bar sounds like, the drums on a sixteenth-note grid, the named sounds you want to answer one by one, and where the arrangement changes. Then it cuts a sample-exact edit of the track from bar runs you write down, exports cues the picture places every event against, and proves the edit and the final sync to the millisecond.

It was built cutting a 58-second launch film for [MetalUI](https://github.com/vijayksingh/metalui) and extracted so the next film starts from it. The film's analysis, edit and cues reproduce byte-for-byte from this repo.

## What it does

| Command | What it does |
|---|---|
| `fetch` | Downloads `song.json`'s `source` (yt-dlp) and decodes it to WAV. Or drop a file at `<work>/source.<ext>`. |
| `analyze` | One fixed tempo grid (the BPM and phase whose beats sit on the most onset energy, with a drift report), bar 1 (the downbeat where the arrangement changes most sharply), per-bar loudness and bands (sub, low, mid, high), 16-step kick/snare/hat grids, the key, and where the texture changes. → `analysis.json` |
| `edit` | Cuts `edit.json`'s runs of song bars into one edit, sample-exact, with loops, filter/level paths, mutes, risers, rolls, crashes and booms, then masters it to a LUFS target under a limiter. → the audio, `cues` (bars, beats, splices, the timing rule), `meters` (per-frame loudness, bands and drum onsets) and `events` (each named sound, onset by onset, with its pitch) |
| `check` | Re-reads the finished edit and proves it: every run is the song's own samples where the cues say, every splice is on a bar line, one tempo straight through, no clicks, builds that build for 4+ bars, payoffs that land (weight back, a slam over the build, over the drop they answer), cues fresh, and what a person heard at each splice. |
| `storyboard` | Checks a storyboard (shots that each own a run of bars) covers the edit once, and prints every shot against the sound: seconds, frames, song sections, splices, named sounds, loudness, drum grids. |
| `contour` | How loud the edit plays beat by beat between two bars (K-weighted, like LUFS), for shaping transitions. |
| `master` | Lays the edit under a silent render and proves the sync by cross-correlation (renderers' own AAC muxes often run a couple of thousand samples late). |
| `spectrogram` | Images of the track or the edit with bar lines and section names, for a reviewer (person or model) to read what the numbers can't say. |

## Quick start

```bash
uv sync
uv run python examples/click-track/make_track.py      # synthesise the example's track
uv run beatboard -C examples/click-track analyze
uv run beatboard -C examples/click-track edit
uv run beatboard -C examples/click-track check
uv run beatboard -C examples/click-track storyboard
```

Needs Python 3.10+, [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install ffmpeg`; yt-dlp too for `fetch`).

## A project

A project is a directory with a `beatboard.json`. Every path in it is relative to the project, and anything left out takes a default, so the smallest manifest is `{}` beside a `song.json` and an `edit.json`.

```json
{
  "fps": 60,
  "paths": {
    "song": "music/song.json",
    "edit": "music/edit.json",
    "work": "music/out",
    "audio": "public/soundtrack.m4a",
    "cues": "src/cues.generated.json",
    "meters": "src/meters.generated.json",
    "eventsOut": "src/events.generated.json"
  }
}
```

Defaults: authored `song.json`, `edit.json`, `listening.json`, `events.json`; generated `analysis.json`, `out/` (work: WAVs, spectrograms), `out/edit.m4a`, `generated/cues.json`, `generated/meters.json`, `generated/events.json`. `fps` is the picture's frame rate (meters are sampled per frame).

### Authored files

- **`song.json`**: the `source` and the song's sections by bar (`{ "id", "bars": [first, last], "what" }`), labels for spectrograms and cues.
- **`edit.json`**: the cut, as `pieces` of song bars played in order (`{ "bars": [a, b] }`, `"end"` to play out), each optionally looped (`times`) with transition `fx` (filter and level paths, mutes) and `layers` (riser, roll, reverse, boom, crash, hit, sweep), and a `payoff` for a landing that must hit. Plus `maxSeconds`, `gainDb`, `master` (`lufs`, `ceilingDb`), fades. See `examples/click-track/edit.json` for the smallest one.
- **`events.json`**: sounds the picture answers one by one: a band of the mix over a span of bars, harmonic or percussive.
- **`listening.json`**: what a person heard at each splice. `check` fails on a bad verdict and asks for a listen where nobody has.
- **`storyboard.json`**: `{ "shots": [{ "id", "bars": [first, last], "title", "picture", "sync" }] }`.

Never hand-edit a generated file: change what it's generated from and re-run.

## In the picture: `beatboard-clock`

`clock/` is a small TypeScript module (no framework) that turns the cues into frames. It isn't on npm yet: add it as a git or path dependency (`"beatboard-clock": "file:../beatboard/clock"`) or copy `clock/src/index.ts`. Place every event in musical time and let it round each one to a frame on its own; adding frame counts drifts a frame every few bars.

```ts
import { createClock } from 'beatboard-clock';
import cues from './generated/cues.json';
import meters from './generated/meters.json';

const clock = createClock({ cues, meters, fps: 60 });
clock.frameAt(9, 1);            // the frame the verse lands on
clock.frameAt(12, 3, 2);        // bar 12, beat 3, the "and"
clock.position(frame);          // { bar, beat, step, phase, beats }
clock.meter('kick', frame);     // the edit's kick at a frame, 0..1
```

## Tests

Feature slices only: each test runs the real `beatboard` command end to end on a fresh copy of the example project, whose every fact is known, and reads what it wrote.

```bash
uv run --group dev pytest
```

## Known limits, and where it goes next

- **One tempo.** The grid is fixed for the whole track: right for step-sequenced and click-tracked music, wrong for a live band that drifts. A tempo map is the next grid.
- **Four beats to the bar.** Bars are assumed 4/4.
- **Texture changes** report the top 18 candidates whatever the track's length; on a short track that is every bar. They want a threshold.
- **The checks' thresholds** (build length, payoff weight and slam) were calibrated on one film against one person's verdicts; they are a starting point.
- **`pipeline.py` is one file.** Splitting it into analysis, edit, master and check modules is the next refactor, with the MetalUI film's outputs as the byte-for-byte proof.
- **Render harness.** Rendering a picture by bar range with the right slice of audio muxed in, and stills and contact sheets by bar, belong here too.

## License

MIT
