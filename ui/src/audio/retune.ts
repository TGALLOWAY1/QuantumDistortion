export const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;

export const RETUNE_SCALES = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  pentatonic: [0, 2, 4, 7, 9],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  mixolydian: [0, 2, 4, 5, 7, 9, 10],
  harmonic_minor: [0, 2, 3, 5, 7, 8, 11],
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
} as const;

export type RetuneScale = keyof typeof RETUNE_SCALES;
export type RetuneOutOfMaskMode = 'nearest' | 'preserve' | 'mute';
export type RetuneLowEndMode = 'hybrid';
export type RetuneMode = 'polyphonic';
export type RetuneNoteState = 'mapped' | 'preserved' | 'muted';

export interface RetuneTrackedNote {
  sourceMidi: number;
  targetMidi: number;
  confidence: number;
  energy: number;
  bandwidth: number;
  isLowEnd: boolean;
  state: RetuneNoteState;
}

export function buildScaleMask(key: number, scale: RetuneScale): boolean[] {
  const mask = new Array(12).fill(false);
  const intervals = RETUNE_SCALES[scale] ?? RETUNE_SCALES.major;
  for (const interval of intervals) {
    mask[(key + interval + 12) % 12] = true;
  }
  return mask;
}

export function formatMidiNote(midi: number): string {
  if (!Number.isFinite(midi)) return '—';
  const rounded = Math.round(midi);
  const note = NOTE_NAMES[((rounded % 12) + 12) % 12];
  const octave = Math.floor(rounded / 12) - 1;
  return `${note}${octave}`;
}

export function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function freqToMidi(freq: number): number {
  if (!Number.isFinite(freq) || freq <= 0) return Number.NEGATIVE_INFINITY;
  return 69 + 12 * Math.log2(freq / 440);
}
