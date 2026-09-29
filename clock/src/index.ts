/**
 * Musical time for a picture, from the files `beatboard edit` writes.
 *
 * Every event is placed in musical time (bar, beat, sixteenth) and turned into a frame on its own:
 * a beat is rarely a whole number of frames, so adding frame counts drifts a frame every few bars.
 * No framework: works in Remotion, a canvas loop, or a test.
 *
 *   import cues from './generated/cues.json';
 *   import meters from './generated/meters.json';
 *   const clock = createClock({ cues, meters, fps: 60 });
 *   clock.frameAt(9, 1);          // the frame the verse lands on
 *   clock.position(frame).beat;   // which beat a frame sits in
 *   clock.meter('kick', frame);   // the edit's kick, 0..1, at a frame
 */

export interface Cues {
  seconds: number;
  bpm: number;
  beatSeconds: number;
  barSeconds: number;
  /** Where edit bar 1 starts, in seconds (a pickup can come before it). */
  firstDownbeat: number;
  bars: { n: number; t: number; section?: string | null; loud?: number; drums?: Record<string, string> }[];
  splices?: { t: number; from: number[]; to: number[] }[];
  audio?: string;
}

export interface Meters {
  fps: number;
  frames: number;
  [series: string]: number | number[];
}

export type MeterName = 'loud' | 'sub' | 'low' | 'mid' | 'high' | 'kick' | 'snare' | 'hat';

export interface Position {
  bar: number;
  beat: number;
  step: number;
  /** 0..1 through the current beat. */
  phase: number;
  /** Beats since bar 1, fractional. */
  beats: number;
}

export function createClock({ cues, meters, fps }: { cues: Cues; meters?: Meters; fps: number }) {
  /** Seconds at a point in the edit. Bars and beats count from 1, sixteenth steps from 0. */
  const at = (bar: number, beat = 1, step = 0) =>
    cues.firstDownbeat + (bar - 1) * cues.barSeconds + (beat - 1) * cues.beatSeconds + (step * cues.beatSeconds) / 4;

  /** The frame a musical event lands on. */
  const frameAt = (bar: number, beat = 1, step = 0) => Math.round(at(bar, beat, step) * fps);

  /**
   * Where a frame sits in the music. Read from the middle of the frame: frameAt() rounds, so an
   * event can land up to half a frame before its exact time, and its own frame must still read as it.
   */
  const position = (frame: number): Position => {
    const beats = ((frame + 0.5) / fps - cues.firstDownbeat) / cues.beatSeconds;
    const whole = Math.floor(beats);
    return { bar: Math.floor(whole / 4) + 1, beat: (((whole % 4) + 4) % 4) + 1, step: Math.floor((beats - whole) * 4), phase: beats - whole, beats };
  };

  /** A reading of the finished edit at a frame, 0..1: loudness, a band, or a drum hit. */
  const meter = (name: MeterName, frame: number) => {
    if (!meters) throw new Error('createClock: pass meters to read them');
    const series = meters[name] as number[];
    const i = Math.round((frame * meters.fps) / fps);
    return series[Math.max(0, Math.min(series.length - 1, i))] ?? 0;
  };

  return {
    fps,
    duration: Math.ceil(cues.seconds * fps),
    bars: cues.bars,
    splices: cues.splices ?? [],
    at,
    frameAt,
    position,
    meter,
  };
}

export type Clock = ReturnType<typeof createClock>;
