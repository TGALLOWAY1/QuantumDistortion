/**
 * Quantum Distortion - Realtime AudioWorklet Processor
 *
 * Runs in the audio thread. Implements:
 * - Wavefold distortion
 * - Soft-tube saturation (two independent stages)
 * - Body-band pitch quantization with detector filtering and generated sub
 * - Bitcrush / Lo-Fi
 * - Simple delay + feedback
 * - Chorus modulation
 * - 5-band parametric EQ with multiple filter types
 */

// Musical scale intervals (semitones from root)
const SCALES = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  pentatonic: [0, 2, 4, 7, 9],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  mixolydian: [0, 2, 4, 5, 7, 9, 10],
  harmonic_minor: [0, 2, 3, 5, 7, 8, 11],
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

const MAX_CHANNELS = 2;

class QuantumProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Parameters controlled via message port
    this.params = {
      // Global
      bypass: false,
      dryWet: 1.0,
      masterGain: 1.0,

      // Saturation (stage 1)
      saturateEnabled: true,
      saturateType: 'tape',     // tape, tube, wavefold
      saturateDrive: 0.5,
      saturateTilt: 0.5,

      // Saturation (stage 2)
      saturate2Enabled: true,
      saturate2Type: 'tape',
      saturate2Drive: 0.5,
      saturate2Tilt: 0.5,

      // Spectral Quantize
      quantizeEnabled: true,
      quantizeKey: 0,           // 0=C, 1=C#, etc.
      quantizeScale: 'major',
      quantizeStrength: 1.0,
      quantizeSubEnabled: true,
      quantizeSubSource: 'root',
      quantizeSubNote: 0,
      quantizeSubDegree: 0,
      quantizeSubOctave: 2,
      quantizeSubLevel: 0.35,
      quantizeAirMix: 1.0,

      // Low/High end gains (applied post-chain)
      lowGain: 1.0,
      highGain: 1.0,

      // Dev-only: drive range scaler (0-1, controls saturation intensity ceiling)
      _devDriveRange: 0.4,

      // Delay
      delayEnabled: false,
      delayTime: 0.25,          // seconds
      delayFeedback: 0.3,

      // Modulation (chorus)
      modEnabled: false,
      modDepth: 0.5,
      modRate: 1.0,             // Hz

      // Lo-Fi
      lofiEnabled: false,
      lofiWear: 0.5,
      lofiWobble: 0.3,

      // EQ bands (5-band parametric with filter type)
      eqEnabled: true,
      eqBands: [
        { freq: 60, gain: 0, q: 1.0, type: 'lowshelf' },
        { freq: 250, gain: 0, q: 1.0, type: 'peak' },
        { freq: 1000, gain: 0, q: 1.0, type: 'peak' },
        { freq: 4000, gain: 0, q: 1.0, type: 'peak' },
        { freq: 12000, gain: 0, q: 1.0, type: 'highshelf' },
      ],

      // Key-aware parametric EQ instances
      peqInstances: [],
    };

    // Per-channel state for stereo processing
    // Delay buffers (max 2 seconds at 48kHz)
    this.delayBuffers = Array.from({ length: MAX_CHANNELS }, () => new Float32Array(96000));
    this.delayWriteIndexes = new Array(MAX_CHANNELS).fill(0);

    // Chorus LFO phase (shared across channels)
    this.chorusPhase = 0;

    // Lo-Fi sample-and-hold state (per-channel)
    this.lofiHoldSamples = new Array(MAX_CHANNELS).fill(0);
    this.lofiHoldCounters = new Array(MAX_CHANNELS).fill(0);

    // Biquad filter states for EQ (per-channel, 5 bands each)
    this.eqStates = Array.from({ length: MAX_CHANNELS }, () =>
      Array.from({ length: 5 }, () => ({ x1: 0, x2: 0, y1: 0, y2: 0 }))
    );
    this.eqCoeffs = Array.from({ length: 5 }, () => ({
      b0: 1, b1: 0, b2: 0, a1: 0, a2: 0,
    }));
    this.eqDirty = true;

    // --- Pitch correction (quantize / autotune) state ---
    this.quantizeGrainSize = 1024;
    this.quantizeBufSize = 4096;
    this.quantizeBuffers = Array.from({ length: MAX_CHANNELS }, () => new Float32Array(4096));
    this.quantizeWritePos = new Array(MAX_CHANNELS).fill(0);
    this.quantizeDelayTaps = Array.from(
      { length: MAX_CHANNELS },
      () => [this.quantizeGrainSize * 0.25, this.quantizeGrainSize * 0.75],
    );
    this.quantizeTargetRatio = 1.0;
    this.quantizeSmoothedRatio = 1.0;
    this.quantizeHeldTargetFreq = 0.0;
    this.quantizeCandidateTargetFreq = 0.0;
    this.quantizeCandidateCount = 0;
    this.quantizeMissingCount = 3;
    this.quantizeDetectorEnv = 0.0;
    this.quantizeSubPhase = 0.0;
    this.quantizeSubGain = 0.0;
    this.quantizeSubCutHz = 110.0;
    this.quantizeAirCutHz = 5000.0;
    this.quantizeDetectorHighHz = 3000.0;
    this.quantizeMinConfidence = 0.72;
    this.quantizeMinEnv = 0.01;
    this.quantizeChangeCents = 40.0;
    this.quantizeConfirmFrames = 3;
    this.quantizeReleaseFrames = 2;
    this.quantizeLowState = new Array(MAX_CHANNELS).fill(0.0);
    this.quantizeBodyLpState = new Array(MAX_CHANNELS).fill(0.0);
    this.quantizeDetectorLpState = new Array(MAX_CHANNELS).fill(0.0);

    // Deferred air/sub samples for post-chain mixing
    this.deferredAir = new Array(MAX_CHANNELS).fill(0.0);
    this.deferredSubOsc = 0.0;

    // YIN pitch detection buffers
    this.yinBufSize = 2048;
    this.yinBuffer = new Float32Array(2048);
    this.yinWritePos = 0;
    this.yinTempBuffer = new Float32Array(961); // tau range 0..960
    this.pitchDetectCounter = 0;
    this.pitchDetectInterval = 512; // detect pitch every ~10ms at 48kHz
    this.detectedPitch = 0;

    // --- Parametric EQ (key-aware) state ---
    // peqInstances param is an array of { id, enabled, mode, key, scale, amount, q }
    // For each instance we precompute biquad filters targeting in-key or out-of-key notes
    this.peqMaxInstances = 8;
    this.peqMaxFilters = 60; // 12 pitch classes × 5 octaves
    this.peqOctaveStart = 2;
    this.peqOctaveEnd = 6;
    // Coefficients: [instance][filter] = { b0, b1, b2, a1, a2 }
    this.peqCoeffs = Array.from({ length: this.peqMaxInstances }, () =>
      Array.from({ length: this.peqMaxFilters }, () => ({ b0: 1, b1: 0, b2: 0, a1: 0, a2: 0 }))
    );
    // Filter count per instance (how many filters are active)
    this.peqFilterCounts = new Array(this.peqMaxInstances).fill(0);
    // Per-channel filter states: [channel][instance][filter] = { x1, x2, y1, y2 }
    this.peqStates = Array.from({ length: MAX_CHANNELS }, () =>
      Array.from({ length: this.peqMaxInstances }, () =>
        Array.from({ length: this.peqMaxFilters }, () => ({ x1: 0, x2: 0, y1: 0, y2: 0 }))
      )
    );
    this.peqDirty = true;

    // FFT for spectrum analysis (sent to UI)
    this.analysisSamples = new Float32Array(2048);
    this.analysisWritePos = 0;
    this.frameCount = 0;

    this.port.onmessage = (e) => {
      if (e.data.type === 'params') {
        Object.assign(this.params, e.data.params);
        this.eqDirty = true;
        this.peqDirty = true;
      }
    };
  }

  static get parameterDescriptors() {
    return [];
  }

  // Compute biquad EQ coefficients for all filter types
  computeEQCoeffs() {
    const sr = sampleRate;
    for (let i = 0; i < 5; i++) {
      const band = this.params.eqBands[i];
      const f0 = band.freq;
      const gainDB = band.gain;
      const Q = Math.max(band.q, 0.1);
      const type = band.type || 'peak';

      const A = Math.pow(10, gainDB / 40);
      const w0 = 2 * Math.PI * f0 / sr;
      const sinW = Math.sin(w0);
      const cosW = Math.cos(w0);
      const alpha = sinW / (2 * Q);

      let b0, b1, b2, a0, a1, a2;

      switch (type) {
        case 'peak':
          b0 = 1 + alpha * A;
          b1 = -2 * cosW;
          b2 = 1 - alpha * A;
          a0 = 1 + alpha / A;
          a1 = -2 * cosW;
          a2 = 1 - alpha / A;
          break;

        case 'lowshelf': {
          const twoSqrtAAlpha = 2 * Math.sqrt(A) * alpha;
          b0 = A * ((A + 1) - (A - 1) * cosW + twoSqrtAAlpha);
          b1 = 2 * A * ((A - 1) - (A + 1) * cosW);
          b2 = A * ((A + 1) - (A - 1) * cosW - twoSqrtAAlpha);
          a0 = (A + 1) + (A - 1) * cosW + twoSqrtAAlpha;
          a1 = -2 * ((A - 1) + (A + 1) * cosW);
          a2 = (A + 1) + (A - 1) * cosW - twoSqrtAAlpha;
          break;
        }

        case 'highshelf': {
          const twoSqrtAAlpha = 2 * Math.sqrt(A) * alpha;
          b0 = A * ((A + 1) + (A - 1) * cosW + twoSqrtAAlpha);
          b1 = -2 * A * ((A - 1) + (A + 1) * cosW);
          b2 = A * ((A + 1) + (A - 1) * cosW - twoSqrtAAlpha);
          a0 = (A + 1) - (A - 1) * cosW + twoSqrtAAlpha;
          a1 = 2 * ((A - 1) - (A + 1) * cosW);
          a2 = (A + 1) - (A - 1) * cosW - twoSqrtAAlpha;
          break;
        }

        case 'lowpass':
          b0 = (1 - cosW) / 2;
          b1 = 1 - cosW;
          b2 = (1 - cosW) / 2;
          a0 = 1 + alpha;
          a1 = -2 * cosW;
          a2 = 1 - alpha;
          break;

        case 'highpass':
          b0 = (1 + cosW) / 2;
          b1 = -(1 + cosW);
          b2 = (1 + cosW) / 2;
          a0 = 1 + alpha;
          a1 = -2 * cosW;
          a2 = 1 - alpha;
          break;

        default:
          // Fallback to peaking
          b0 = 1 + alpha * A;
          b1 = -2 * cosW;
          b2 = 1 - alpha * A;
          a0 = 1 + alpha / A;
          a1 = -2 * cosW;
          a2 = 1 - alpha / A;
          break;
      }

      this.eqCoeffs[i] = {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
      };
    }
    this.eqDirty = false;
  }

  // Compute biquad coefficients for all active PEQ (key-aware) instances
  computePeqCoeffs() {
    const sr = sampleRate;
    const instances = this.params.peqInstances || [];

    for (let inst = 0; inst < this.peqMaxInstances; inst++) {
      if (inst >= instances.length || !instances[inst].enabled) {
        this.peqFilterCounts[inst] = 0;
        continue;
      }

      const cfg = instances[inst];
      const scale = SCALES[cfg.scale] || SCALES.major;
      const key = cfg.key || 0;
      const amount = cfg.amount || 0;
      const Q = Math.max(cfg.q || 2.0, 0.1);
      const isBoost = cfg.mode === 'boost';

      // Gain in dB: amount 0-1 maps to 0-18 dB
      const gainDB = amount * 18;
      if (gainDB < 0.1) {
        this.peqFilterCounts[inst] = 0;
        continue;
      }

      // Determine which pitch classes are in the scale
      const inScale = new Set(scale.map(s => (key + s) % 12));

      let filterIdx = 0;
      for (let octave = this.peqOctaveStart; octave <= this.peqOctaveEnd; octave++) {
        for (let pc = 0; pc < 12; pc++) {
          const isInKey = inScale.has(pc);
          // Boost mode: filter in-key notes (positive gain)
          // Cut mode: filter out-of-key notes (negative gain)
          const shouldFilter = isBoost ? isInKey : !isInKey;
          if (!shouldFilter) continue;
          if (filterIdx >= this.peqMaxFilters) break;

          const midi = (octave + 1) * 12 + pc;
          const freq = 440 * Math.pow(2, (midi - 69) / 12);

          // Skip frequencies outside useful range
          if (freq < 30 || freq > sr * 0.45) continue;

          // Compute peak biquad coefficients
          const appliedGain = isBoost ? gainDB : -gainDB;
          const A = Math.pow(10, appliedGain / 40);
          const w0 = 2 * Math.PI * freq / sr;
          const sinW = Math.sin(w0);
          const cosW = Math.cos(w0);
          const alpha = sinW / (2 * Q);

          const b0 = 1 + alpha * A;
          const b1 = -2 * cosW;
          const b2 = 1 - alpha * A;
          const a0 = 1 + alpha / A;
          const a1 = -2 * cosW;
          const a2 = 1 - alpha / A;

          this.peqCoeffs[inst][filterIdx] = {
            b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
            a1: a1 / a0, a2: a2 / a0,
          };
          filterIdx++;
        }
      }
      this.peqFilterCounts[inst] = filterIdx;
    }
    this.peqDirty = false;
  }

  // Apply a single biquad filter sample
  biquad(sample, coeffs, state) {
    const out = coeffs.b0 * sample + coeffs.b1 * state.x1 + coeffs.b2 * state.x2
      - coeffs.a1 * state.y1 - coeffs.a2 * state.y2;
    state.x2 = state.x1;
    state.x1 = sample;
    state.y2 = state.y1;
    state.y1 = out;
    return out;
  }

  // Tape saturation: soft-clip with harmonics
  tapeSaturate(x, drive) {
    const range = this.params._devDriveRange;
    const d = 1 + drive * 20 * range;
    const driven = x * d;
    const raw = Math.tanh(driven);
    const normalized = raw / Math.tanh(d);
    return normalized * 0.4 + raw * 0.6;
  }

  // Tube saturation: asymmetric soft-clip
  tubeSaturate(x, drive) {
    const range = this.params._devDriveRange;
    const d = 1 + drive * 25 * range;
    const driven = x * d;
    if (driven >= 0) {
      const raw = Math.tanh(driven * 1.2);
      return raw * 0.7 + (raw / Math.tanh(d * 1.2)) * 0.3;
    }
    const raw = Math.tanh(driven * 0.8);
    return raw * 0.7 + (raw / Math.tanh(d * 0.8)) * 0.3;
  }

  // Wavefold distortion
  wavefold(x, drive) {
    const range = this.params._devDriveRange;
    const d = 1 + drive * 20 * range;
    let s = x * d;
    // Triangle wavefold
    s = Math.asin(Math.sin(s * Math.PI / 2)) * 2 / Math.PI;
    return s;
  }

  // Apply saturation with tilt for a given stage
  applySaturation(sample, drive, type, tilt) {
    switch (type) {
      case 'tape':
        sample = this.tapeSaturate(sample, drive);
        break;
      case 'tube':
        sample = this.tubeSaturate(sample, drive);
        break;
      case 'wavefold':
        sample = this.wavefold(sample, drive);
        break;
    }

    // Tilt: simple high-shelf approximation
    const t = (tilt - 0.5) * 2;
    if (Math.abs(t) > 0.01) {
      sample = sample * (1 + t * 0.3);
    }

    return sample;
  }

  onePoleLowpass(sample, cutoff, stateArray, ch) {
    const safeCutoff = Math.max(10, Math.min(cutoff, sampleRate * 0.45));
    const coeff = Math.exp(-2 * Math.PI * safeCutoff / sampleRate);
    const next = (1 - coeff) * sample + coeff * stateArray[ch];
    stateArray[ch] = next;
    return next;
  }

  splitQuantizeBands(sample, ch) {
    const sub = this.onePoleLowpass(sample, this.quantizeSubCutHz, this.quantizeLowState, ch);
    const bodyLp = this.onePoleLowpass(sample, this.quantizeAirCutHz, this.quantizeBodyLpState, ch);
    const body = bodyLp - sub;
    const air = sample - bodyLp;
    const detector = this.onePoleLowpass(body, this.quantizeDetectorHighHz, this.quantizeDetectorLpState, ch);
    return { sub, body, air, detector };
  }

  centsBetween(freqA, freqB) {
    if (freqA <= 0 || freqB <= 0) return 0;
    return 1200 * Math.log2(freqA / freqB);
  }

  updateQuantizePitchState(pitch, confidence) {
    if (pitch > 0 && confidence >= this.quantizeMinConfidence && this.quantizeDetectorEnv >= this.quantizeMinEnv) {
      this.detectedPitch = pitch;
      const target = this.nearestScaleFreq(
        pitch, this.params.quantizeKey, this.params.quantizeScale
      );

      if (this.quantizeHeldTargetFreq <= 0) {
        this.quantizeHeldTargetFreq = target;
        this.quantizeCandidateTargetFreq = 0;
        this.quantizeCandidateCount = 0;
      } else {
        const deltaCents = Math.abs(this.centsBetween(target, this.quantizeHeldTargetFreq));
        if (deltaCents >= this.quantizeChangeCents) {
          const candidateDelta = this.quantizeCandidateTargetFreq > 0
            ? Math.abs(this.centsBetween(target, this.quantizeCandidateTargetFreq))
            : Infinity;
          if (this.quantizeCandidateTargetFreq > 0 && candidateDelta < 20) {
            this.quantizeCandidateCount++;
          } else {
            this.quantizeCandidateTargetFreq = target;
            this.quantizeCandidateCount = 1;
          }
          if (this.quantizeCandidateCount >= this.quantizeConfirmFrames) {
            this.quantizeHeldTargetFreq = this.quantizeCandidateTargetFreq;
            this.quantizeCandidateTargetFreq = 0;
            this.quantizeCandidateCount = 0;
          }
        } else {
          this.quantizeCandidateTargetFreq = 0;
          this.quantizeCandidateCount = 0;
        }
      }

      const rawRatio = this.quantizeHeldTargetFreq / pitch;
      this.quantizeTargetRatio = Math.max(0.5, Math.min(2.0, rawRatio));
      this.quantizeMissingCount = 0;
      return;
    }

    this.quantizeMissingCount++;
    if (this.quantizeMissingCount > this.quantizeReleaseFrames) {
      this.quantizeHeldTargetFreq = 0;
      this.quantizeCandidateTargetFreq = 0;
      this.quantizeCandidateCount = 0;
      this.quantizeTargetRatio = 1.0;
    }
  }

  computeSubFrequency() {
    const scale = SCALES[this.params.quantizeScale] || SCALES.major;
    let pitchClass = this.params.quantizeKey;

    if (this.params.quantizeSubSource === 'manual') {
      // Manual note is an index into the scale, so sub is always in-key
      const idx = Math.max(0, Math.min(scale.length - 1, Math.round(this.params.quantizeSubNote)));
      pitchClass = (this.params.quantizeKey + scale[idx]) % 12;
    } else if (this.params.quantizeSubSource === 'scale_degree') {
      const degree = Math.max(0, Math.min(scale.length - 1, Math.round(this.params.quantizeSubDegree)));
      pitchClass = (this.params.quantizeKey + scale[degree]) % 12;
    }

    const octave = Math.max(0, Math.min(4, Math.round(this.params.quantizeSubOctave)));
    const midi = 12 * (octave + 1) + pitchClass;
    return 440 * Math.pow(2, (midi - 69) / 12);
  }

  // --- YIN pitch detection algorithm ---
  // Returns detected pitch in Hz, or 0 if no pitch found
  detectPitch() {
    const buf = this.yinBuffer;
    const W = this.yinBufSize;
    const temp = this.yinTempBuffer;
    const minTau = 12;   // ~4000 Hz at 48kHz
    const maxTau = 960;  // ~50 Hz at 48kHz
    const integrationLen = W - maxTau;

    // Step 1: Difference function
    for (let tau = 0; tau <= maxTau; tau++) {
      let sum = 0;
      for (let j = 0; j < integrationLen; j++) {
        const idx1 = (this.yinWritePos + j) & (W - 1);
        const idx2 = (this.yinWritePos + j + tau) & (W - 1);
        const diff = buf[idx1] - buf[idx2];
        sum += diff * diff;
      }
      temp[tau] = sum;
    }

    // Step 2: Cumulative mean normalized difference function
    temp[0] = 1;
    let runningSum = 0;
    for (let tau = 1; tau <= maxTau; tau++) {
      runningSum += temp[tau];
      temp[tau] = runningSum > 0 ? (temp[tau] * tau) / runningSum : 1;
    }

    // Step 3: Absolute threshold — find first dip below threshold
    const threshold = 0.15;
    let tauEstimate = -1;
    for (let tau = minTau; tau <= maxTau; tau++) {
      if (temp[tau] < threshold) {
        // Walk to the local minimum
        while (tau + 1 <= maxTau && temp[tau + 1] < temp[tau]) {
          tau++;
        }
        tauEstimate = tau;
        break;
      }
    }

    if (tauEstimate === -1) return { pitch: 0, confidence: 0 };

    // Step 4: Parabolic interpolation for sub-sample accuracy
    let betterTau = tauEstimate;
    if (tauEstimate > minTau && tauEstimate < maxTau) {
      const s0 = temp[tauEstimate - 1];
      const s1 = temp[tauEstimate];
      const s2 = temp[tauEstimate + 1];
      const denom = 2 * (s0 - 2 * s1 + s2);
      if (Math.abs(denom) > 1e-10) {
        betterTau = tauEstimate + (s0 - s2) / denom;
      }
    }

    // Step 5: Convert period to Hz
    const pitch = sampleRate / betterTau;
    const confidence = Math.max(0, Math.min(1, 1 - temp[tauEstimate]));

    // Sanity check: musical range only
    if (pitch < 50 || pitch > 4000) return { pitch: 0, confidence: 0 };

    return { pitch, confidence };
  }

  // Find the nearest frequency in the given musical scale
  nearestScaleFreq(freq, key, scaleName) {
    if (freq <= 0) return freq;

    const scale = SCALES[scaleName] || SCALES.major;

    // Convert frequency to continuous MIDI note number
    const midiNote = 12 * Math.log2(freq / 440) + 69;

    // Determine which octave (12-note group) relative to the key
    const noteInOctave = ((midiNote - key) % 12 + 12) % 12;
    const octaveBase = midiNote - noteInOctave;

    let bestNote = Math.round(midiNote);
    let bestDist = 100;

    // Check scale degrees in current octave and neighbors
    for (let octOff = -1; octOff <= 1; octOff++) {
      for (const interval of scale) {
        const candidate = octaveBase + interval + octOff * 12;
        const dist = Math.abs(midiNote - candidate);
        if (dist < bestDist) {
          bestDist = dist;
          bestNote = candidate;
        }
      }
    }

    return 440 * Math.pow(2, (bestNote - 69) / 12);
  }

  // Dual-tap variable-delay pitch shifter tuned for small correction moves
  pitchShiftSample(sample, ch) {
    const maxDelay = this.quantizeGrainSize;
    const bufSize = this.quantizeBufSize;
    const buf = this.quantizeBuffers[ch];
    const ratio = this.quantizeSmoothedRatio;
    const slope = 1.0 - ratio;

    // Write input sample to circular buffer
    buf[this.quantizeWritePos[ch]] = sample;

    let out = 0;
    let weightSum = 0;

    for (let tap = 0; tap < 2; tap++) {
      this.quantizeDelayTaps[ch][tap] += slope;
      while (this.quantizeDelayTaps[ch][tap] < 0) {
        this.quantizeDelayTaps[ch][tap] += maxDelay;
      }
      while (this.quantizeDelayTaps[ch][tap] >= maxDelay) {
        this.quantizeDelayTaps[ch][tap] -= maxDelay;
      }

      const phase = this.quantizeDelayTaps[ch][tap] / maxDelay;
      const windowVal = 0.5 * (1 - Math.cos(2 * Math.PI * phase));
      const readPos = (this.quantizeWritePos[ch] - this.quantizeDelayTaps[ch][tap] + bufSize) % bufSize;
      const idx0 = Math.floor(readPos) & (bufSize - 1);
      const idx1 = (idx0 + 1) & (bufSize - 1);
      const frac = readPos - idx0;
      const readSample = buf[idx0] * (1 - frac) + buf[idx1] * frac;
      out += readSample * windowVal;
      weightSum += windowVal;
    }

    this.quantizeWritePos[ch] = (this.quantizeWritePos[ch] + 1) & (bufSize - 1);
    return weightSum > 1e-6 ? out / weightSum : 0;
  }

  process(inputs, outputs, _parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const numChannels = Math.min(input.length, output.length, MAX_CHANNELS);
    const len = input[0].length;

    // --- Bypass: pass all channels through unchanged ---
    if (this.params.bypass) {
      for (let ch = 0; ch < numChannels; ch++) {
        for (let i = 0; i < input[ch].length; i++) {
          output[ch][i] = input[ch][i];
        }
      }
      // Zero-fill any extra output channels
      for (let ch = numChannels; ch < output.length; ch++) {
        for (let i = 0; i < output[ch].length; i++) {
          output[ch][i] = 0;
        }
      }
      this.sendAnalysis(input[0]);
      return true;
    }

    if (this.eqDirty) {
      this.computeEQCoeffs();
    }
    if (this.peqDirty) {
      this.computePeqCoeffs();
    }

    for (let i = 0; i < len; i++) {
      let monoSubSample = 0;

      for (let ch = 0; ch < numChannels; ch++) {
        const channelIn = input[ch];
        const channelOut = output[ch];
        const delayBuf = this.delayBuffers[ch];
        const eqChStates = this.eqStates[ch];

        let sample = channelIn[i];
        const dry = sample;

        // --- EQ ---
        if (this.params.eqEnabled) {
          for (let b = 0; b < 5; b++) {
            const bandType = this.params.eqBands[b].type || 'peak';
            if (bandType === 'lowpass' || bandType === 'highpass' || this.params.eqBands[b].gain !== 0) {
              sample = this.biquad(sample, this.eqCoeffs[b], eqChStates[b]);
            }
          }
        }

        // --- Parametric EQ (key-aware) ---
        const peqInstances = this.params.peqInstances || [];
        for (let inst = 0; inst < peqInstances.length && inst < this.peqMaxInstances; inst++) {
          const filterCount = this.peqFilterCounts[inst];
          if (filterCount > 0) {
            const instStates = this.peqStates[ch][inst];
            const instCoeffs = this.peqCoeffs[inst];
            for (let f = 0; f < filterCount; f++) {
              sample = this.biquad(sample, instCoeffs[f], instStates[f]);
            }
          }
        }

        // --- Saturation (stage 1) ---
        if (this.params.saturateEnabled) {
          sample = this.applySaturation(
            sample,
            this.params.saturateDrive,
            this.params.saturateType,
            this.params.saturateTilt
          );
        }

        if (this.params.quantizeEnabled) {
          const bands = this.splitQuantizeBands(sample, ch);

          if (ch === 0) {
            const detectorAbs = Math.abs(bands.detector);
            const envLerp = detectorAbs > this.quantizeDetectorEnv ? 0.08 : 0.005;
            this.quantizeDetectorEnv += (detectorAbs - this.quantizeDetectorEnv) * envLerp;

            this.yinBuffer[this.yinWritePos] = bands.detector;
            this.yinWritePos = (this.yinWritePos + 1) & (this.yinBufSize - 1);
            this.pitchDetectCounter++;
            if (this.pitchDetectCounter >= this.pitchDetectInterval) {
              this.pitchDetectCounter = 0;
              const detection = this.detectPitch();
              this.updateQuantizePitchState(detection.pitch, detection.confidence);
            }

            const strength = this.params.quantizeStrength;
            const effectiveTarget = 1.0 + strength * (this.quantizeTargetRatio - 1.0);
            this.quantizeSmoothedRatio += (effectiveTarget - this.quantizeSmoothedRatio) * 0.01;
            this.quantizeSmoothedRatio = Math.max(0.5, Math.min(2.0, this.quantizeSmoothedRatio));

            // Generate sub oscillator sample (deferred — mixed in after sat2)
            if (this.params.quantizeSubEnabled) {
              const subFreq = this.computeSubFrequency();
              const targetGain = Math.min(1.0, this.quantizeDetectorEnv * 4.0) * this.params.quantizeSubLevel;
              this.quantizeSubGain += (targetGain - this.quantizeSubGain) * 0.02;
              this.quantizeSubPhase += (2 * Math.PI * subFreq) / sampleRate;
              if (this.quantizeSubPhase > 2 * Math.PI) this.quantizeSubPhase -= 2 * Math.PI;
              this.deferredSubOsc = Math.sin(this.quantizeSubPhase) * this.quantizeSubGain;
            } else {
              this.quantizeSubGain += (0 - this.quantizeSubGain) * 0.05;
              this.deferredSubOsc = 0;
            }
          }

          const shiftedBody = this.pitchShiftSample(bands.body, ch);
          const correctedBody = Math.abs(this.quantizeSmoothedRatio - 1.0) < 0.015
            ? bands.body
            : shiftedBody;

          // Pass through original sub + corrected body only; air and sub osc deferred
          sample = bands.sub + correctedBody;
          this.deferredAir[ch] = bands.air;
        } else {
          // Quantize disabled — no air to defer
          this.deferredAir[ch] = 0;
          this.deferredSubOsc = 0;
          if (ch === 0) {
            this.quantizeDetectorEnv *= 0.995;
            this.quantizeSubGain *= 0.95;
            this.quantizeTargetRatio += (1.0 - this.quantizeTargetRatio) * 0.05;
            this.quantizeSmoothedRatio += (1.0 - this.quantizeSmoothedRatio) * 0.05;
            this.quantizeHeldTargetFreq = 0.0;
            this.quantizeCandidateTargetFreq = 0.0;
            this.quantizeCandidateCount = 0;
            this.quantizeMissingCount = this.quantizeReleaseFrames + 1;
          }
        }

        // --- Delay ---
        if (this.params.delayEnabled) {
          const delaySamples = Math.floor(this.params.delayTime * sampleRate);
          const readIndex = (this.delayWriteIndexes[ch] - delaySamples + delayBuf.length) % delayBuf.length;
          const delayed = delayBuf[readIndex];
          delayBuf[this.delayWriteIndexes[ch]] = sample + delayed * this.params.delayFeedback;
          this.delayWriteIndexes[ch] = (this.delayWriteIndexes[ch] + 1) % delayBuf.length;
          sample = sample + delayed * 0.5;
        }

        // --- Chorus Modulation ---
        if (this.params.modEnabled) {
          const depth = this.params.modDepth * 0.005 * sampleRate;
          const lfoVal = Math.sin(2 * Math.PI * this.chorusPhase);
          const modDelay = Math.floor(depth * (1 + lfoVal) + 0.002 * sampleRate);
          const modReadIndex = (this.delayWriteIndexes[ch] - modDelay + delayBuf.length) % delayBuf.length;
          const modSample = delayBuf[modReadIndex] || 0;
          sample = sample * 0.7 + modSample * 0.3;
        }

        // --- Lo-Fi ---
        if (this.params.lofiEnabled) {
          const wear = this.params.lofiWear;
          const holdLength = Math.max(1, Math.floor(1 + wear * 15));
          this.lofiHoldCounters[ch]++;
          if (this.lofiHoldCounters[ch] >= holdLength) {
            this.lofiHoldSamples[ch] = sample;
            this.lofiHoldCounters[ch] = 0;
          }
          sample = this.lofiHoldSamples[ch];

          const wobble = this.params.lofiWobble;
          if (wobble > 0.01) {
            const wobbleLFO = Math.sin(this.chorusPhase * 0.3 * 2 * Math.PI) * wobble * 0.15;
            sample *= (1 + wobbleLFO);
          }
        }

        // --- Saturation (stage 2) ---
        if (this.params.saturate2Enabled) {
          sample = this.applySaturation(
            sample,
            this.params.saturate2Drive,
            this.params.saturate2Type,
            this.params.saturate2Tilt
          );
        }

        // --- Sub oscillator (after distortion, with low-end gain) ---
        if (this.params.quantizeEnabled && this.params.quantizeSubEnabled) {
          sample += this.deferredSubOsc * this.params.lowGain;
        }

        // --- Air (end of chain, with high-end gain) ---
        sample += this.deferredAir[ch] * this.params.quantizeAirMix * this.params.highGain;

        sample = dry * (1 - this.params.dryWet) + sample * this.params.dryWet;
        sample *= this.params.masterGain;
        sample = Math.tanh(sample);
        channelOut[i] = sample;
      }

      if (this.params.modEnabled || this.params.lofiEnabled) {
        this.chorusPhase += this.params.modRate / sampleRate;
        if (this.chorusPhase > 1) this.chorusPhase -= 1;
      }
    }

    // Zero-fill any extra output channels
    for (let ch = numChannels; ch < output.length; ch++) {
      for (let i = 0; i < output[ch].length; i++) {
        output[ch][i] = 0;
      }
    }

    this.sendAnalysis(output[0]);
    return true;
  }

  sendAnalysis(data) {
    // Accumulate samples for spectrum analysis, send every ~50ms
    for (let i = 0; i < data.length; i++) {
      this.analysisSamples[this.analysisWritePos] = data[i];
      this.analysisWritePos = (this.analysisWritePos + 1) % this.analysisSamples.length;
    }

    this.frameCount++;
    // Send analysis data roughly every 50ms (at 128 samples/frame @ 48kHz ≈ every 18 frames)
    if (this.frameCount % 18 === 0) {
      // Reorder buffer so it's contiguous
      const ordered = new Float32Array(this.analysisSamples.length);
      const start = this.analysisWritePos;
      for (let i = 0; i < ordered.length; i++) {
        ordered[i] = this.analysisSamples[(start + i) % this.analysisSamples.length];
      }
      this.port.postMessage({ type: 'analysis', samples: ordered });
    }
  }
}

registerProcessor('quantum-processor', QuantumProcessor);
