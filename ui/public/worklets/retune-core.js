export const SCALES = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  pentatonic: [0, 2, 4, 7, 9],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  mixolydian: [0, 2, 4, 5, 7, 9, 10],
  harmonic_minor: [0, 2, 3, 5, 7, 8, 11],
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};

const MAX_TRACKED_NOTES = 8;
const MAX_LOW_NOTES = 3;
const MAX_CHANNELS = 2;
const TWO_PI = Math.PI * 2;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function midiToFreq(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

function freqToMidi(freq) {
  if (freq <= 0) return -Infinity;
  return 69 + 12 * Math.log2(freq / 440);
}

function centsBetween(freqA, freqB) {
  if (freqA <= 0 || freqB <= 0) return 0;
  return 1200 * Math.log2(freqA / freqB);
}

function pitchClassForMidi(midi) {
  return ((Math.round(midi) % 12) + 12) % 12;
}

export function buildScaleMask(key, scaleName) {
  const scale = SCALES[scaleName] || SCALES.major;
  const mask = new Array(12).fill(false);
  for (let i = 0; i < scale.length; i++) {
    mask[(key + scale[i] + 12) % 12] = true;
  }
  return mask;
}

export const DEFAULT_RETUNE_PARAMS = {
  retuneEnabled: false,
  retuneMode: 'polyphonic',
  retuneKey: 0,
  retuneScale: 'major',
  retuneStrength: 0.7,
  retuneDeadbandCents: 18,
  retuneConfidenceGate: 0.22,
  retuneTargetMask: buildScaleMask(0, 'major'),
  retuneOutOfMaskMode: 'nearest',
  retunePreserveTransients: true,
  retuneTextureAmount: 0.2,
  retuneLowEndMode: 'hybrid',
  retuneLowEndBlend: 0.55,
  retuneCollapseDuplicates: false,
  retuneAirMix: 1.0,
};

const LEGACY_PARAM_ALIASES = {
  quantizeEnabled: 'retuneEnabled',
  quantizeKey: 'retuneKey',
  quantizeScale: 'retuneScale',
  quantizeStrength: 'retuneStrength',
  quantizeAirMix: 'retuneAirMix',
};

function cloneMask(mask) {
  if (!Array.isArray(mask)) {
    return buildScaleMask(DEFAULT_RETUNE_PARAMS.retuneKey, DEFAULT_RETUNE_PARAMS.retuneScale);
  }
  const normalized = new Array(12).fill(false);
  for (let i = 0; i < 12; i++) {
    normalized[i] = Boolean(mask[i]);
  }
  return normalized;
}

export function normalizeRetuneParamPatch(patch, currentParams = DEFAULT_RETUNE_PARAMS) {
  const normalized = {};
  const source = patch || {};

  for (const [key, value] of Object.entries(source)) {
    if (Object.prototype.hasOwnProperty.call(DEFAULT_RETUNE_PARAMS, key)) {
      normalized[key] = value;
      continue;
    }

    const alias = LEGACY_PARAM_ALIASES[key];
    if (!alias) {
      continue;
    }

    normalized[alias] = value;
  }

  if (Object.prototype.hasOwnProperty.call(normalized, 'retuneScale')) {
    normalized.retuneScale = normalized.retuneScale in SCALES ? normalized.retuneScale : currentParams.retuneScale;
  }

  if (Object.prototype.hasOwnProperty.call(normalized, 'retuneTargetMask')) {
    normalized.retuneTargetMask = cloneMask(normalized.retuneTargetMask);
  } else if (
    Object.prototype.hasOwnProperty.call(normalized, 'retuneKey')
    || Object.prototype.hasOwnProperty.call(normalized, 'retuneScale')
  ) {
    const nextKey = Object.prototype.hasOwnProperty.call(normalized, 'retuneKey')
      ? normalized.retuneKey
      : currentParams.retuneKey;
    const nextScale = Object.prototype.hasOwnProperty.call(normalized, 'retuneScale')
      ? normalized.retuneScale
      : currentParams.retuneScale;
    normalized.retuneTargetMask = buildScaleMask(nextKey, nextScale);
  }

  if (Object.prototype.hasOwnProperty.call(normalized, 'retuneTargetMask')) {
    const hasAny = normalized.retuneTargetMask.some(Boolean);
    if (!hasAny) {
      normalized.retuneTargetMask = buildScaleMask(
        normalized.retuneKey ?? currentParams.retuneKey,
        normalized.retuneScale ?? currentParams.retuneScale,
      );
    }
  }

  if (Object.prototype.hasOwnProperty.call(normalized, 'retuneDeadbandCents')) {
    normalized.retuneDeadbandCents = clamp(Number(normalized.retuneDeadbandCents) || 0, 0, 60);
  }
  if (Object.prototype.hasOwnProperty.call(normalized, 'retuneConfidenceGate')) {
    normalized.retuneConfidenceGate = clamp(Number(normalized.retuneConfidenceGate) || 0, 0, 1);
  }

  return normalized;
}

class Radix2FFT {
  constructor(size) {
    this.size = size;
    this.levels = Math.log2(size) | 0;
    this.bitReverse = new Uint32Array(size);
    this.cosTable = new Float32Array(size / 2);
    this.sinTable = new Float32Array(size / 2);

    for (let i = 0; i < size; i++) {
      let x = i;
      let y = 0;
      for (let j = 0; j < this.levels; j++) {
        y = (y << 1) | (x & 1);
        x >>= 1;
      }
      this.bitReverse[i] = y;
    }

    for (let i = 0; i < size / 2; i++) {
      const angle = TWO_PI * i / size;
      this.cosTable[i] = Math.cos(angle);
      this.sinTable[i] = Math.sin(angle);
    }
  }

  transform(real, imag, inverse = false) {
    const n = this.size;

    for (let i = 0; i < n; i++) {
      const j = this.bitReverse[i];
      if (j > i) {
        const realTmp = real[i];
        real[i] = real[j];
        real[j] = realTmp;

        const imagTmp = imag[i];
        imag[i] = imag[j];
        imag[j] = imagTmp;
      }
    }

    for (let size = 2; size <= n; size <<= 1) {
      const halfSize = size >> 1;
      const tableStep = n / size;
      for (let start = 0; start < n; start += size) {
        let tableIndex = 0;
        for (let i = start; i < start + halfSize; i++) {
          const j = i + halfSize;
          const wr = this.cosTable[tableIndex];
          const wi = inverse ? this.sinTable[tableIndex] : -this.sinTable[tableIndex];
          const tr = wr * real[j] - wi * imag[j];
          const ti = wr * imag[j] + wi * real[j];

          real[j] = real[i] - tr;
          imag[j] = imag[i] - ti;
          real[i] += tr;
          imag[i] += ti;
          tableIndex += tableStep;
        }
      }
    }

    if (inverse) {
      const inv = 1 / n;
      for (let i = 0; i < n; i++) {
        real[i] *= inv;
        imag[i] *= inv;
      }
    }
  }
}

function createWindow(size) {
  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    window[i] = 0.5 - 0.5 * Math.cos(TWO_PI * i / size);
  }
  return window;
}

function createSynthesisWindow(window, hopSize) {
  const size = window.length;
  const overlapNorm = new Float32Array(size);
  for (let shift = 0; shift < size; shift += hopSize) {
    for (let i = 0; i < size; i++) {
      const idx = (i + shift) % size;
      overlapNorm[i] += window[idx] * window[idx];
    }
  }

  const synthesis = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    synthesis[i] = window[i] / Math.max(overlapNorm[i], 1e-6);
  }
  return synthesis;
}

function createRingSet(length, itemLength) {
  return Array.from({ length }, () => new Float32Array(itemLength));
}

function nearestMidiForPitchClass(sourceMidi, pitchClass) {
  const rounded = Math.round(sourceMidi);
  let bestMidi = rounded;
  let bestDistance = Infinity;

  for (let candidate = rounded - 24; candidate <= rounded + 24; candidate++) {
    if (((candidate % 12) + 12) % 12 !== pitchClass) {
      continue;
    }
    const distance = Math.abs(candidate - sourceMidi);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestMidi = candidate;
    }
  }

  return bestMidi;
}

function findNearestAllowedMidi(sourceMidi, targetMask, occupiedPitchClasses = null) {
  const rounded = Math.round(sourceMidi);
  let bestMidi = rounded;
  let bestDistance = Infinity;

  for (let candidate = rounded - 24; candidate <= rounded + 24; candidate++) {
    const pitchClass = ((candidate % 12) + 12) % 12;
    if (!targetMask[pitchClass]) {
      continue;
    }
    if (occupiedPitchClasses && occupiedPitchClasses.has(pitchClass)) {
      continue;
    }
    const distance = Math.abs(candidate - sourceMidi);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestMidi = candidate;
    }
  }

  return Number.isFinite(bestDistance) ? bestMidi : null;
}

function makeTrackedNote(id, candidate) {
  return {
    id,
    sourceMidi: candidate.sourceMidi,
    targetMidi: candidate.sourceMidi,
    confidence: clamp(candidate.confidence, 0, 1),
    energy: candidate.energy,
    bandwidth: candidate.bandwidth,
    isLowEnd: candidate.isLowEnd,
    state: 'mapped',
    age: 0,
    missingFrames: 0,
    targetLockFrames: 0,
    phases: [0, 0, 0],
  };
}

export class PolyphonicRetuneCore {
  constructor({ sampleRate, maxChannels = MAX_CHANNELS }) {
    this.sampleRate = sampleRate;
    this.maxChannels = maxChannels;
    this.params = { ...DEFAULT_RETUNE_PARAMS, retuneTargetMask: cloneMask(DEFAULT_RETUNE_PARAMS.retuneTargetMask) };

    this.tonalFrameSize = 4096;
    this.tonalHopSize = 1024;
    this.lowFrameSize = 8192;
    this.lowHopSize = 2048;
    this.latencySamples = this.tonalFrameSize - this.tonalHopSize;
    this.analysisLatencyMs = (this.latencySamples / this.sampleRate) * 1000;

    this.tonalRingSize = 16384;
    this.tonalRingMask = this.tonalRingSize - 1;
    this.outputRingSize = 32768;
    this.outputRingMask = this.outputRingSize - 1;
    this.delayRingSize = 8192;
    this.delayRingMask = this.delayRingSize - 1;

    this.tonalInputRings = createRingSet(maxChannels, this.tonalRingSize);
    this.tonalMonoRing = new Float32Array(this.tonalRingSize);
    this.lowMonoRing = new Float32Array(this.lowFrameSize * 2);
    this.lowMonoRingMask = this.lowMonoRing.length - 1;
    this.outputRings = createRingSet(maxChannels, this.outputRingSize);
    this.airDelayRings = createRingSet(maxChannels, this.delayRingSize);

    this.tonalWindow = createWindow(this.tonalFrameSize);
    this.tonalSynthesisWindow = createSynthesisWindow(this.tonalWindow, this.tonalHopSize);
    this.lowWindow = createWindow(this.lowFrameSize);

    this.tonalFFT = new Radix2FFT(this.tonalFrameSize);
    this.lowFFT = new Radix2FFT(this.lowFrameSize);
    this.tonalMonoReal = new Float32Array(this.tonalFrameSize);
    this.tonalMonoImag = new Float32Array(this.tonalFrameSize);
    this.tonalWorkReal = new Float32Array(this.tonalFrameSize);
    this.tonalWorkImag = new Float32Array(this.tonalFrameSize);
    this.tonalTargetReal = new Float32Array(this.tonalFrameSize);
    this.tonalTargetImag = new Float32Array(this.tonalFrameSize);
    this.tonalMagnitudes = new Float32Array(this.tonalFrameSize / 2 + 1);
    this.prevTonalMagnitudes = new Float32Array(this.tonalFrameSize / 2 + 1);
    this.lowReal = new Float32Array(this.lowFrameSize);
    this.lowImag = new Float32Array(this.lowFrameSize);
    this.lowMagnitudes = new Float32Array(this.lowFrameSize / 2 + 1);
    this.blockInputScratch = createRingSet(maxChannels, this.tonalFrameSize);
    this.blockOutputScratch = createRingSet(maxChannels, 128);

    this.noteEnergyScratch = new Float32Array(97);
    this.lowNoteEnergyScratch = new Float32Array(53);

    this.lowCutState = new Array(maxChannels).fill(0);
    this.airLpState = new Array(maxChannels).fill(0);
    this.lowBlendState = new Array(maxChannels).fill(0);
    this.lowAnalysisState = new Array(maxChannels).fill(0);

    this.trackedNotes = [];
    this.nextTrackedNoteId = 1;
    this.transientBypassActive = false;
    this.transientMix = 1.0;
    this.lowEndLocked = false;
    this.overloaded = false;
    this.noteCount = 0;
    this.detectedPitch = 0;
    this.targetPitch = 0;
    this.targetRatio = 1;
    this.smoothedRatio = 1;
    this.detectorEnv = 0;
    this.confidence = 0;
    this.subGuideFrequency = 0;
    this.subGuideLevel = 0;

    this.tonalWritePos = 0;
    this.tonalInputSampleCount = 0;
    this.tonalSamplesSinceFrame = 0;
    this.lowWritePos = 0;
    this.lowSampleCount = 0;
    this.lowSamplesSinceFrame = 0;
    this.outputSampleCount = 0;
    this.delayWritePos = 0;
  }

  ensureBlockSize(length) {
    if (this.blockOutputScratch[0].length === length) {
      return;
    }
    this.blockOutputScratch = createRingSet(this.maxChannels, length);
  }

  setParams(patch) {
    const normalized = normalizeRetuneParamPatch(patch, this.params);
    const nextParams = {
      ...this.params,
      ...normalized,
      retuneTargetMask: normalized.retuneTargetMask ? cloneMask(normalized.retuneTargetMask) : cloneMask(this.params.retuneTargetMask),
    };
    const enabledChanged = nextParams.retuneEnabled !== this.params.retuneEnabled;
    const mappingChanged = (
      nextParams.retuneKey !== this.params.retuneKey
      || nextParams.retuneScale !== this.params.retuneScale
      || nextParams.retuneOutOfMaskMode !== this.params.retuneOutOfMaskMode
      || nextParams.retuneCollapseDuplicates !== this.params.retuneCollapseDuplicates
      || nextParams.retuneTargetMask.some((enabled, index) => enabled !== this.params.retuneTargetMask[index])
    );
    this.params = nextParams;
    if (mappingChanged) {
      for (const note of this.trackedNotes) {
        note.targetLockFrames = 0;
      }
    }
    if (enabledChanged && !nextParams.retuneEnabled) {
      this.reset();
    }
  }

  setRuntimeHealth({ overloaded }) {
    this.overloaded = overloaded;
  }

  reset() {
    this.tonalWritePos = 0;
    this.tonalInputSampleCount = 0;
    this.tonalSamplesSinceFrame = 0;
    this.lowWritePos = 0;
    this.lowSampleCount = 0;
    this.lowSamplesSinceFrame = 0;
    this.outputSampleCount = 0;
    this.delayWritePos = 0;
    this.trackedNotes = [];
    this.nextTrackedNoteId = 1;
    this.transientBypassActive = false;
    this.transientMix = 1;
    this.lowEndLocked = false;
    this.noteCount = 0;
    this.detectedPitch = 0;
    this.targetPitch = 0;
    this.targetRatio = 1;
    this.smoothedRatio = 1;
    this.detectorEnv = 0;
    this.confidence = 0;
    this.subGuideFrequency = 0;
    this.subGuideLevel = 0;
    this.prevTonalMagnitudes.fill(0);
    this.tonalMonoRing.fill(0);
    this.lowMonoRing.fill(0);

    for (let ch = 0; ch < this.maxChannels; ch++) {
      this.tonalInputRings[ch].fill(0);
      this.outputRings[ch].fill(0);
      this.airDelayRings[ch].fill(0);
      this.lowCutState[ch] = 0;
      this.airLpState[ch] = 0;
      this.lowBlendState[ch] = 0;
      this.lowAnalysisState[ch] = 0;
    }
  }

  onePoleLowpass(sample, cutoff, stateArray, ch) {
    const safeCutoff = clamp(cutoff, 10, this.sampleRate * 0.45);
    const coeff = Math.exp(-TWO_PI * safeCutoff / this.sampleRate);
    const next = (1 - coeff) * sample + coeff * stateArray[ch];
    stateArray[ch] = next;
    return next;
  }

  splitSample(sample, ch) {
    const lowFloor = this.onePoleLowpass(sample, 35, this.lowCutState, ch);
    const lowRemoved = sample - lowFloor;
    const tonal = this.onePoleLowpass(lowRemoved, 5000, this.airLpState, ch);
    const air = lowRemoved - tonal;
    const lowSeed = this.onePoleLowpass(tonal, 220, this.lowAnalysisState, ch);
    return { tonal, air, lowSeed };
  }

  copyFrameFromRing(ring, writePos, ringMask, frameSize, frameBuffer) {
    const start = (writePos - frameSize + ring.length) & ringMask;
    for (let i = 0; i < frameSize; i++) {
      frameBuffer[i] = ring[(start + i) & ringMask];
    }
  }

  magnitudeAtFreq(magnitudes, frameSize, freq) {
    if (freq <= 0) return 0;
    const bin = freq * frameSize / this.sampleRate;
    const maxBin = magnitudes.length - 1;
    if (bin <= 0 || bin >= maxBin) return 0;
    const index = Math.floor(bin);
    const frac = bin - index;
    return magnitudes[index] * (1 - frac) + magnitudes[index + 1] * frac;
  }

  computeSpectralFlatness(magnitudes, minBin, maxBin) {
    let logSum = 0;
    let linearSum = 0;
    let count = 0;

    for (let bin = minBin; bin <= maxBin; bin++) {
      const magnitude = Math.max(magnitudes[bin], 1e-12);
      logSum += Math.log(magnitude);
      linearSum += magnitude;
      count++;
    }

    if (count === 0 || linearSum <= 0) {
      return 1;
    }

    const geometricMean = Math.exp(logSum / count);
    const arithmeticMean = linearSum / count;
    return clamp(geometricMean / Math.max(arithmeticMean, 1e-12), 0, 1);
  }

  detectCandidatesFromSpectrum(magnitudes, frameSize, options) {
    const {
      minMidi,
      maxMidi,
      maxHarmonics,
      maxCandidates,
      lowCutoffHz,
      isLowEnd,
    } = options;
    const scratch = isLowEnd ? this.lowNoteEnergyScratch : this.noteEnergyScratch;
    scratch.fill(0);

    for (let midi = minMidi; midi <= maxMidi; midi++) {
      const fundamental = midiToFreq(midi);
      let energy = 0;
      let support = 0;
      for (let harmonic = 1; harmonic <= maxHarmonics; harmonic++) {
        const harmonicFreq = fundamental * harmonic;
        if (harmonicFreq >= lowCutoffHz || harmonicFreq >= this.sampleRate * 0.48) {
          break;
        }
        const weight = isLowEnd ? 1 / Math.pow(harmonic, 0.6) : 1 / Math.pow(harmonic, 0.8);
        const mag = this.magnitudeAtFreq(magnitudes, frameSize, harmonicFreq);
        const lower = this.magnitudeAtFreq(magnitudes, frameSize, harmonicFreq * Math.pow(2, -25 / 1200));
        const upper = this.magnitudeAtFreq(magnitudes, frameSize, harmonicFreq * Math.pow(2, 25 / 1200));
        const prominence = Math.max(0, mag - 0.45 * (lower + upper));
        energy += prominence * weight;
        support += weight;
      }
      scratch[midi] = support > 0 ? energy / support : 0;
    }

    const smoothed = [];
    let energyMax = 0;
    let energySum = 0;
    for (let midi = minMidi; midi <= maxMidi; midi++) {
      const left = midi > minMidi ? scratch[midi - 1] : 0;
      const center = scratch[midi];
      const right = midi < maxMidi ? scratch[midi + 1] : 0;
      const smoothedEnergy = 0.2 * left + 0.6 * center + 0.2 * right;
      smoothed[midi] = smoothedEnergy;
      energyMax = Math.max(energyMax, smoothedEnergy);
      energySum += smoothedEnergy;
    }

    if (energyMax <= 1e-6) {
      return [];
    }

    const threshold = Math.max(energyMax * (isLowEnd ? 0.28 : 0.22), energySum / Math.max((maxMidi - minMidi + 1) * 4, 1));
    const candidates = [];

    for (let midi = minMidi; midi <= maxMidi; midi++) {
      const current = smoothed[midi] || 0;
      const left = midi > minMidi ? smoothed[midi - 1] || 0 : 0;
      const right = midi < maxMidi ? smoothed[midi + 1] || 0 : 0;
      if (current < threshold || current < left || current < right) {
        continue;
      }

      candidates.push({
        sourceMidi: midi,
        confidence: clamp(current / Math.max(energySum, 1e-6) * (isLowEnd ? 4 : 6), 0, 1),
        energy: current,
        bandwidth: isLowEnd ? 45 : 60,
        isLowEnd,
      });
    }

    candidates.sort((a, b) => b.energy - a.energy);
    return candidates.slice(0, maxCandidates);
  }

  updateTrackedNotes(candidates) {
    const matchedIds = new Set();
    const activeNotes = this.trackedNotes.filter(note => note.missingFrames <= 4);

    for (const candidate of candidates) {
      let bestNote = null;
      let bestDistance = Infinity;

      for (const note of activeNotes) {
        if (matchedIds.has(note.id) || note.isLowEnd !== candidate.isLowEnd) {
          continue;
        }
        const distance = Math.abs(note.sourceMidi - candidate.sourceMidi);
        if (distance < 1.75 && distance < bestDistance) {
          bestNote = note;
          bestDistance = distance;
        }
      }

      if (!bestNote) {
        this.trackedNotes.push(makeTrackedNote(this.nextTrackedNoteId++, candidate));
        continue;
      }

      matchedIds.add(bestNote.id);
      bestNote.sourceMidi += (candidate.sourceMidi - bestNote.sourceMidi) * 0.35;
      bestNote.energy += (candidate.energy - bestNote.energy) * 0.28;
      bestNote.confidence += (candidate.confidence - bestNote.confidence) * 0.35;
      bestNote.bandwidth += (candidate.bandwidth - bestNote.bandwidth) * 0.3;
      bestNote.age++;
      bestNote.missingFrames = 0;
    }

    const keptNotes = [];
    for (const note of this.trackedNotes) {
      const wasMatched = matchedIds.has(note.id);
      if (!wasMatched) {
        note.missingFrames++;
        note.energy *= 0.86;
        note.confidence *= 0.82;
      }
      if (note.missingFrames <= 4 && note.confidence > 0.03 && note.energy > 1e-5) {
        keptNotes.push(note);
      }
    }

    keptNotes.sort((a, b) => (
      b.confidence - a.confidence
      || b.age - a.age
      || b.energy - a.energy
      || a.id - b.id
    ));
    const tonalNotes = keptNotes.filter(note => !note.isLowEnd).slice(0, MAX_TRACKED_NOTES);
    const lowNotes = keptNotes.filter(note => note.isLowEnd).slice(0, MAX_LOW_NOTES);
    this.trackedNotes = [...tonalNotes, ...lowNotes];
    this.assignTargets();
  }

  resolveTargetDecision(sourceMidi, occupiedPitchClasses) {
    const sourceRounded = Math.round(sourceMidi);
    const sourcePitchClass = pitchClassForMidi(sourceRounded);
    const targetMask = this.params.retuneTargetMask;
    let targetMidi = sourceRounded;
    let state = 'mapped';

    if (targetMask[sourcePitchClass]) {
      targetMidi = nearestMidiForPitchClass(sourceMidi, sourcePitchClass);
    } else {
      switch (this.params.retuneOutOfMaskMode) {
        case 'preserve':
          state = 'preserved';
          break;
        case 'mute':
          state = 'muted';
          break;
        default: {
          const nearest = findNearestAllowedMidi(sourceMidi, targetMask, occupiedPitchClasses);
          if (nearest === null) {
            state = 'preserved';
          } else {
            targetMidi = nearest;
          }
          break;
        }
      }
    }

    return {
      sourceRounded,
      state,
      targetMidi,
    };
  }

  applyTargetHysteresis(note, nextDecision, occupiedPitchClasses) {
    const targetMask = this.params.retuneTargetMask;
    const previousState = note.state;
    const previousTarget = note.targetMidi;
    let { state, targetMidi } = nextDecision;

    if (
      previousState === 'mapped'
      && state === 'mapped'
      && Number.isFinite(previousTarget)
    ) {
      const previousPitchClass = pitchClassForMidi(previousTarget);
      const previousAllowed = targetMask[previousPitchClass];
      const previousFree = !occupiedPitchClasses || !occupiedPitchClasses.has(previousPitchClass);

      if (previousAllowed && previousFree) {
        const previousError = Math.abs(note.sourceMidi - previousTarget);
        const nextError = Math.abs(note.sourceMidi - targetMidi);
        const improvement = previousError - nextError;
        const hysteresisMargin = note.confidence < 0.28
          ? 0.95
          : note.confidence < 0.45
            ? 0.6
            : 0.32;
        const semitoneStep = Math.abs(targetMidi - previousTarget);

        if (
          note.targetLockFrames > 0
          || (semitoneStep <= 2 && improvement < hysteresisMargin)
        ) {
          targetMidi = previousTarget;
        }
      }
    }

    if (
      previousState === 'mapped'
      && state !== 'mapped'
      && note.confidence < 0.24
      && Number.isFinite(previousTarget)
    ) {
      const previousPitchClass = pitchClassForMidi(previousTarget);
      const previousAllowed = targetMask[previousPitchClass];
      const previousFree = !occupiedPitchClasses || !occupiedPitchClasses.has(previousPitchClass);
      if (previousAllowed && previousFree) {
        state = 'mapped';
        targetMidi = previousTarget;
      }
    }

    if (state !== previousState || targetMidi !== previousTarget) {
      note.targetLockFrames = note.confidence < 0.3 ? 8 : 5;
    } else {
      note.targetLockFrames = Math.max(0, note.targetLockFrames - 1);
    }

    return {
      state,
      targetMidi,
    };
  }

  assignTargets() {
    const occupiedPitchClasses = this.params.retuneCollapseDuplicates ? null : new Set();
    for (const note of this.trackedNotes) {
      const nextDecision = this.resolveTargetDecision(note.sourceMidi, occupiedPitchClasses);
      let { targetMidi, state } = this.applyTargetHysteresis(note, nextDecision, occupiedPitchClasses);
      const baseGate = this.params.retuneConfidenceGate ?? 0.22;
      const confidenceFloor = note.isLowEnd ? Math.max(0.1, baseGate * 0.75) : baseGate;
      const eligibleForMapping = note.confidence >= confidenceFloor && (note.age >= 2 || note.targetLockFrames > 0);

      if (state === 'mapped' && !eligibleForMapping) {
        if (note.targetLockFrames > 0 && Number.isFinite(note.targetMidi)) {
          targetMidi = note.targetMidi;
        } else {
          state = 'preserved';
          targetMidi = nextDecision.sourceRounded;
        }
      }

      if (!this.params.retuneCollapseDuplicates && state === 'mapped') {
        const targetPitchClass = pitchClassForMidi(targetMidi);
        if (occupiedPitchClasses.has(targetPitchClass)) {
          const alternate = findNearestAllowedMidi(note.sourceMidi, this.params.retuneTargetMask, occupiedPitchClasses);
          if (alternate !== null && alternate !== note.targetMidi) {
            targetMidi = alternate;
            note.targetLockFrames = Math.min(note.targetLockFrames, 2);
          } else {
            state = 'preserved';
            targetMidi = nextDecision.sourceRounded;
          }
        }
        if (state === 'mapped') {
          occupiedPitchClasses.add(pitchClassForMidi(targetMidi));
        }
      }

      note.targetMidi = targetMidi;
      note.state = state;
    }

    this.trackedNotes.sort((a, b) => (
      b.confidence - a.confidence
      || b.age - a.age
      || b.energy - a.energy
      || a.id - b.id
    ));
    this.noteCount = this.trackedNotes.length;
    const strongest = this.trackedNotes[0];
    this.detectedPitch = strongest ? midiToFreq(strongest.sourceMidi) : 0;
    this.targetPitch = strongest ? midiToFreq(strongest.targetMidi) : 0;
    this.targetRatio = strongest ? clamp(this.targetPitch / Math.max(this.detectedPitch, 1e-6), 0.25, 4) : 1;
    this.smoothedRatio += (this.targetRatio - this.smoothedRatio) * 0.18;
    this.confidence = strongest ? strongest.confidence : 0;
    this.lowEndLocked = this.trackedNotes.some(note => note.isLowEnd && note.confidence > 0.18 && note.state !== 'muted');

    const strongestLow = this.trackedNotes.find(
      (note) => note.isLowEnd && note.confidence > 0.12 && note.state !== 'muted',
    );
    if (strongestLow) {
      const correctionBlend = this.getCorrectionBlend(strongestLow);
      const effectiveMidi = strongestLow.state === 'mapped'
        ? strongestLow.sourceMidi + (strongestLow.targetMidi - strongestLow.sourceMidi) * correctionBlend
        : strongestLow.sourceMidi;
      this.subGuideFrequency = midiToFreq(effectiveMidi);
      this.subGuideLevel = clamp(strongestLow.confidence * 0.9, 0, 1);
    } else {
      this.subGuideFrequency = 0;
      this.subGuideLevel = 0;
    }
  }

  analyzeTonalFrame() {
    this.copyFrameFromRing(
      this.tonalMonoRing,
      this.tonalWritePos,
      this.tonalRingMask,
      this.tonalFrameSize,
      this.tonalMonoReal,
    );

    let frameEnergy = 0;
    for (let i = 0; i < this.tonalFrameSize; i++) {
      const sample = this.tonalMonoReal[i] * this.tonalWindow[i];
      this.tonalMonoReal[i] = sample;
      this.tonalMonoImag[i] = 0;
      frameEnergy += sample * sample;
    }

    this.tonalFFT.transform(this.tonalMonoReal, this.tonalMonoImag, false);
    for (let bin = 0; bin < this.tonalMagnitudes.length; bin++) {
      this.tonalMagnitudes[bin] = Math.hypot(this.tonalMonoReal[bin], this.tonalMonoImag[bin]);
    }

    let flux = 0;
    let previousTotal = 0;
    for (let bin = 1; bin < this.tonalMagnitudes.length; bin++) {
      const current = this.tonalMagnitudes[bin];
      const previous = this.prevTonalMagnitudes[bin];
      if (current > previous) {
        flux += current - previous;
      }
      previousTotal += previous;
      this.prevTonalMagnitudes[bin] = current;
    }

    const fluxRatio = flux / Math.max(previousTotal, 1e-6);
    this.transientBypassActive = this.params.retunePreserveTransients && fluxRatio > 0.55;
    this.transientMix = this.transientBypassActive ? 0.35 : 1.0;
    this.detectorEnv += (Math.sqrt(frameEnergy / this.tonalFrameSize) - this.detectorEnv) * 0.18;
    const tonalFlatness = this.computeSpectralFlatness(
      this.tonalMagnitudes,
      2,
      Math.min(this.tonalMagnitudes.length - 1, Math.floor(5000 * this.tonalFrameSize / this.sampleRate)),
    );
    const tonalTonality = clamp((0.82 - tonalFlatness) / 0.62, 0, 1);

    const tonalCandidates = this.detectCandidatesFromSpectrum(this.tonalMagnitudes, this.tonalFrameSize, {
      minMidi: 40,
      maxMidi: 96,
      maxHarmonics: 8,
      maxCandidates: MAX_TRACKED_NOTES,
      lowCutoffHz: 5000,
      isLowEnd: false,
    });
    for (const candidate of tonalCandidates) {
      candidate.confidence *= tonalTonality;
    }
    this.updateTrackedNotes(tonalCandidates);
  }

  analyzeLowFrame() {
    this.copyFrameFromRing(
      this.lowMonoRing,
      this.lowWritePos,
      this.lowMonoRingMask,
      this.lowFrameSize,
      this.lowReal,
    );

    for (let i = 0; i < this.lowFrameSize; i++) {
      this.lowReal[i] *= this.lowWindow[i];
      this.lowImag[i] = 0;
    }

    this.lowFFT.transform(this.lowReal, this.lowImag, false);
    for (let bin = 0; bin < this.lowMagnitudes.length; bin++) {
      this.lowMagnitudes[bin] = Math.hypot(this.lowReal[bin], this.lowImag[bin]);
    }
    const lowFlatness = this.computeSpectralFlatness(
      this.lowMagnitudes,
      2,
      Math.min(this.lowMagnitudes.length - 1, Math.floor(320 * this.lowFrameSize / this.sampleRate)),
    );
    const lowTonality = clamp((0.9 - lowFlatness) / 0.7, 0, 1);

    const lowCandidates = this.detectCandidatesFromSpectrum(this.lowMagnitudes, this.lowFrameSize, {
      minMidi: 28,
      maxMidi: 52,
      maxHarmonics: 6,
      maxCandidates: MAX_LOW_NOTES,
      lowCutoffHz: 320,
      isLowEnd: true,
    });
    for (const candidate of lowCandidates) {
      candidate.confidence *= lowTonality;
    }
    this.updateTrackedNotes(lowCandidates);
  }

  getActiveTonalNotes() {
    return this.trackedNotes.filter(
      note => !note.isLowEnd && note.missingFrames <= 2 && note.confidence > 0.08 && note.energy > 1e-5,
    );
  }

  getCorrectionBlend(note) {
    if (note.state !== 'mapped') {
      return 0;
    }

    const centsError = Math.abs(note.sourceMidi - note.targetMidi) * 100;
    const baseDeadband = this.params.retuneDeadbandCents ?? 18;
    const deadbandCents = note.isLowEnd ? baseDeadband * 0.85 : baseDeadband;
    const fullCorrectionCents = deadbandCents + (note.isLowEnd ? 38 : 50);

    if (centsError <= deadbandCents) {
      return 0;
    }
    if (centsError >= fullCorrectionCents) {
      return 1;
    }

    const normalized = (centsError - deadbandCents) / (fullCorrectionCents - deadbandCents);
    return normalized * normalized * (3 - 2 * normalized);
  }

  applySpectralRetune(real, imag) {
    const activeNotes = this.getActiveTonalNotes();
    const halfBins = this.tonalMagnitudes.length;
    const strengthBase = clamp(this.params.retuneStrength * this.transientMix, 0, 1);
    const textureScalar = 1 - this.params.retuneTextureAmount * 0.45;
    const binHz = this.sampleRate / this.tonalFrameSize;

    this.tonalTargetReal.fill(0);
    this.tonalTargetImag.fill(0);

    for (let bin = 1; bin < halfBins - 1; bin++) {
      const sourceReal = real[bin];
      const sourceImag = imag[bin];
      const sourceMagnitude = Math.hypot(sourceReal, sourceImag);
      if (sourceMagnitude < 1e-6) {
        continue;
      }

      const frequency = bin * binHz;
      let bestNote = null;
      let bestScore = 0;

      for (const note of activeNotes) {
        const sourceFreq = midiToFreq(note.sourceMidi);
        for (let harmonic = 1; harmonic <= 8; harmonic++) {
          const harmonicFreq = sourceFreq * harmonic;
          if (harmonicFreq >= this.sampleRate * 0.48 || harmonicFreq > 5000) {
            break;
          }

          const width = note.bandwidth + harmonic * 8;
          const centsOff = Math.abs(centsBetween(frequency, harmonicFreq));
          if (centsOff > width) {
            continue;
          }

          const score = note.energy * note.confidence * (1 - centsOff / width) / Math.pow(harmonic, 0.72);
          if (score > bestScore) {
            bestScore = score;
            bestNote = note;
          }
        }
      }

      if (!bestNote || bestScore < 0.01) {
        this.tonalTargetReal[bin] += sourceReal;
        this.tonalTargetImag[bin] += sourceImag;
        continue;
      }

      if (bestNote.state === 'muted') {
        continue;
      }

      if (bestNote.state === 'preserved') {
        this.tonalTargetReal[bin] += sourceReal;
        this.tonalTargetImag[bin] += sourceImag;
        continue;
      }

      const correctionBlend = this.getCorrectionBlend(bestNote);
      if (correctionBlend <= 1e-3) {
        this.tonalTargetReal[bin] += sourceReal;
        this.tonalTargetImag[bin] += sourceImag;
        continue;
      }

      const effectiveTargetMidi = bestNote.sourceMidi + (bestNote.targetMidi - bestNote.sourceMidi) * correctionBlend;
      const sourceFreq = midiToFreq(bestNote.sourceMidi);
      const targetFreq = midiToFreq(effectiveTargetMidi);
      const ratio = targetFreq / Math.max(sourceFreq, 1e-6);
      const moveAmount = clamp(
        strengthBase * textureScalar * correctionBlend * Math.min(1, bestScore * 1.5),
        0,
        1,
      );
      const stayAmount = 1 - moveAmount;

      this.tonalTargetReal[bin] += sourceReal * stayAmount;
      this.tonalTargetImag[bin] += sourceImag * stayAmount;

      const targetBin = clamp(bin * ratio, 1, halfBins - 2);
      const baseIndex = Math.floor(targetBin);
      const frac = targetBin - baseIndex;
      const movedReal = sourceReal * moveAmount;
      const movedImag = sourceImag * moveAmount;

      this.tonalTargetReal[baseIndex] += movedReal * (1 - frac);
      this.tonalTargetImag[baseIndex] += movedImag * (1 - frac);
      this.tonalTargetReal[baseIndex + 1] += movedReal * frac;
      this.tonalTargetImag[baseIndex + 1] += movedImag * frac;
    }

    real[0] = 0;
    imag[0] = 0;
    for (let bin = 1; bin < halfBins - 1; bin++) {
      real[bin] = this.tonalTargetReal[bin];
      imag[bin] = this.tonalTargetImag[bin];
      const mirror = this.tonalFrameSize - bin;
      real[mirror] = real[bin];
      imag[mirror] = -imag[bin];
    }
    real[halfBins - 1] = this.tonalTargetReal[halfBins - 1];
    imag[halfBins - 1] = 0;
  }

  renderTonalFrame(numChannels) {
    const frameStart = this.tonalInputSampleCount - this.tonalFrameSize;
    const synthStart = frameStart + this.latencySamples;

    for (let ch = 0; ch < numChannels; ch++) {
      const frame = this.blockInputScratch[ch];
      this.copyFrameFromRing(
        this.tonalInputRings[ch],
        this.tonalWritePos,
        this.tonalRingMask,
        this.tonalFrameSize,
        frame,
      );

      for (let i = 0; i < this.tonalFrameSize; i++) {
        this.tonalWorkReal[i] = frame[i] * this.tonalWindow[i];
        this.tonalWorkImag[i] = 0;
      }

      this.tonalFFT.transform(this.tonalWorkReal, this.tonalWorkImag, false);
      this.applySpectralRetune(this.tonalWorkReal, this.tonalWorkImag);
      this.tonalFFT.transform(this.tonalWorkReal, this.tonalWorkImag, true);

      const outputRing = this.outputRings[ch];
      for (let i = 0; i < this.tonalFrameSize; i++) {
        const sample = this.tonalWorkReal[i] * this.tonalSynthesisWindow[i];
        const outputIndex = (synthStart + i) & this.outputRingMask;
        outputRing[outputIndex] += sample;
      }
    }
  }

  renderLowResynthSample() {
    const lowNotes = this.trackedNotes.filter(
      note => note.isLowEnd && note.missingFrames <= 2 && note.confidence > 0.1 && note.state !== 'muted',
    );
    if (lowNotes.length === 0) {
      return 0;
    }

    const totalEnergy = lowNotes.reduce((sum, note) => sum + note.energy, 0);
    let sample = 0;

    for (const note of lowNotes) {
      const correctionBlend = this.getCorrectionBlend(note);
      const targetMidi = note.state === 'mapped'
        ? note.sourceMidi + (note.targetMidi - note.sourceMidi) * correctionBlend
        : note.sourceMidi;
      const targetFreq = midiToFreq(targetMidi);
      const weight = clamp(note.energy / Math.max(totalEnergy, 1e-6), 0, 1) * note.confidence;

      const harmonicWeights = [1.0, 0.45, 0.18];
      for (let harmonic = 0; harmonic < harmonicWeights.length; harmonic++) {
        const frequency = targetFreq * (harmonic + 1);
        note.phases[harmonic] += TWO_PI * frequency / this.sampleRate;
        if (note.phases[harmonic] > TWO_PI) {
          note.phases[harmonic] -= TWO_PI;
        }
        sample += Math.sin(note.phases[harmonic]) * harmonicWeights[harmonic] * weight;
      }
    }

    return sample * 0.35;
  }

  getSubGuide() {
    return {
      frequency: this.subGuideFrequency,
      level: this.subGuideLevel,
      locked: this.lowEndLocked,
    };
  }

  processBlock(inputBlock, numChannels, length, mix = { lowGain: 1, highGain: 1 }) {
    this.ensureBlockSize(length);

    if (!this.params.retuneEnabled) {
      for (let ch = 0; ch < numChannels; ch++) {
        this.blockOutputScratch[ch].set(inputBlock[ch].subarray(0, length));
      }
      return this.blockOutputScratch;
    }

    for (let sampleIndex = 0; sampleIndex < length; sampleIndex++) {
      let monoTonal = 0;
      let monoLow = 0;

      for (let ch = 0; ch < numChannels; ch++) {
        const split = this.splitSample(inputBlock[ch][sampleIndex], ch);
        this.tonalInputRings[ch][this.tonalWritePos] = split.tonal;
        this.airDelayRings[ch][this.delayWritePos] = split.air;
        monoTonal += split.tonal;
        monoLow += split.lowSeed;
      }

      this.tonalMonoRing[this.tonalWritePos] = monoTonal / Math.max(numChannels, 1);
      this.lowMonoRing[this.lowWritePos] = monoLow / Math.max(numChannels, 1);

      this.tonalWritePos = (this.tonalWritePos + 1) & this.tonalRingMask;
      this.lowWritePos = (this.lowWritePos + 1) & this.lowMonoRingMask;
      this.tonalInputSampleCount++;
      this.lowSampleCount++;
      this.tonalSamplesSinceFrame++;
      this.lowSamplesSinceFrame++;

      if (this.tonalInputSampleCount >= this.tonalFrameSize && this.tonalSamplesSinceFrame >= this.tonalHopSize) {
        this.tonalSamplesSinceFrame -= this.tonalHopSize;
        this.analyzeTonalFrame();
        this.renderTonalFrame(numChannels);
      }

      if (this.lowSampleCount >= this.lowFrameSize && this.lowSamplesSinceFrame >= this.lowHopSize) {
        this.lowSamplesSinceFrame -= this.lowHopSize;
        this.analyzeLowFrame();
      }

      const lowResynth = this.renderLowResynthSample();

      for (let ch = 0; ch < numChannels; ch++) {
        const outputIndex = this.outputSampleCount & this.outputRingMask;
        const airReadIndex = (this.delayWritePos - this.latencySamples + this.delayRingSize) & this.delayRingMask;
        const tonalSample = this.outputRings[ch][outputIndex];
        this.outputRings[ch][outputIndex] = 0;

        const delayedAir = this.airDelayRings[ch][airReadIndex];
        const lowRetuned = this.onePoleLowpass(tonalSample, 180, this.lowBlendState, ch);
        const midHighRetuned = tonalSample - lowRetuned;
        const lowBlend = this.params.retuneLowEndMode === 'hybrid' ? this.params.retuneLowEndBlend : 0;
        const hybridLow = lowRetuned * (1 - lowBlend) + lowResynth * lowBlend * mix.lowGain;
        this.blockOutputScratch[ch][sampleIndex] = midHighRetuned + hybridLow + delayedAir * this.params.retuneAirMix * mix.highGain;
      }

      this.outputSampleCount++;
      this.delayWritePos = (this.delayWritePos + 1) & this.delayRingMask;
    }

    return this.blockOutputScratch;
  }

  getStatus() {
    const notes = this.trackedNotes.map(note => ({
      sourceMidi: note.sourceMidi,
      targetMidi: note.targetMidi,
      confidence: note.confidence,
      energy: note.energy,
      bandwidth: note.bandwidth,
      isLowEnd: note.isLowEnd,
      state: note.state,
    }));

    return {
      backendName: 'Polyphonic Retune Core',
      retuneEnabled: this.params.retuneEnabled,
      voiced: this.confidence > 0.08,
      detectedPitch: this.detectedPitch,
      targetPitch: this.targetPitch,
      targetRatio: this.targetRatio,
      smoothedRatio: this.smoothedRatio,
      confidence: this.confidence,
      detectorEnv: this.detectorEnv,
      lowEndStrategy: this.params.retuneLowEndMode === 'hybrid'
        ? 'Tracked Low End + Hybrid Resynth'
        : 'Tracked Low End',
      notes,
      lowEndLocked: this.lowEndLocked,
      transientBypassActive: this.transientBypassActive,
      analysisLatencyMs: this.analysisLatencyMs,
      overloaded: this.overloaded,
      noteCount: notes.length,
    };
  }
}
