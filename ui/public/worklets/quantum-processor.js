/**
 * Quantum Distortion - Realtime AudioWorklet Processor
 *
 * Runs in the audio thread. Implements:
 * - Wavefold distortion
 * - Soft-tube saturation (two independent stages)
 * - Spectral quantization (FFT-based pitch snapping)
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
      quantizeEnabled: false,
      quantizeKey: 0,           // 0=C, 1=C#, etc.
      quantizeScale: 'major',
      quantizeStrength: 0.7,

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
    // Two overlapping grains per channel: [channel][grain] = fractional read position
    this.quantizeGrainReadPos = Array.from({ length: MAX_CHANNELS }, () => [0.0, 0.0]);
    // Grain phase counters: [channel][grain], offset by half grain size
    this.quantizeGrainCounter = Array.from({ length: MAX_CHANNELS }, () => [0, 512]);
    this.quantizeTargetRatio = 1.0;
    this.quantizeSmoothedRatio = 1.0;

    // Precompute Hann window for grain-based pitch shifting
    this.hannWindow = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) {
      this.hannWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / 1024));
    }

    // YIN pitch detection buffers
    this.yinBufSize = 2048;
    this.yinBuffer = new Float32Array(2048);
    this.yinWritePos = 0;
    this.yinTempBuffer = new Float32Array(961); // tau range 0..960
    this.pitchDetectCounter = 0;
    this.pitchDetectInterval = 512; // detect pitch every ~10ms at 48kHz
    this.detectedPitch = 0;

    // FFT for spectrum analysis (sent to UI)
    this.analysisSamples = new Float32Array(2048);
    this.analysisWritePos = 0;
    this.frameCount = 0;

    this.port.onmessage = (e) => {
      if (e.data.type === 'params') {
        Object.assign(this.params, e.data.params);
        this.eqDirty = true;
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
    const d = 1 + drive * 20;
    const driven = x * d;
    const raw = Math.tanh(driven);
    const normalized = raw / Math.tanh(d);
    return normalized * 0.4 + raw * 0.6;
  }

  // Tube saturation: asymmetric soft-clip
  tubeSaturate(x, drive) {
    const d = 1 + drive * 25;
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
    const d = 1 + drive * 20;
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

    if (tauEstimate === -1) return 0; // No pitch detected

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

    // Sanity check: musical range only
    if (pitch < 50 || pitch > 4000) return 0;

    return pitch;
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

  // Grain-based pitch shifting: dual-grain overlap-add with Hann window
  pitchShiftSample(sample, ch) {
    const grainSize = this.quantizeGrainSize;
    const bufSize = this.quantizeBufSize;
    const buf = this.quantizeBuffers[ch];
    const ratio = this.quantizeSmoothedRatio;

    // Write input sample to circular buffer
    buf[this.quantizeWritePos[ch]] = sample;
    this.quantizeWritePos[ch] = (this.quantizeWritePos[ch] + 1) & (bufSize - 1);

    let out = 0;

    for (let g = 0; g < 2; g++) {
      // Advance read position by the pitch shift ratio
      this.quantizeGrainReadPos[ch][g] += ratio;
      if (this.quantizeGrainReadPos[ch][g] >= bufSize) {
        this.quantizeGrainReadPos[ch][g] -= bufSize;
      }

      // Advance grain phase counter
      const phase = this.quantizeGrainCounter[ch][g]++;

      // Reset grain when its window completes
      if (this.quantizeGrainCounter[ch][g] >= grainSize) {
        this.quantizeGrainCounter[ch][g] = 0;
        // Resync read pointer: place it one grain behind the write pointer
        this.quantizeGrainReadPos[ch][g] =
          (this.quantizeWritePos[ch] - grainSize + bufSize) & (bufSize - 1);
      }

      // Read sample with linear interpolation
      const readPos = this.quantizeGrainReadPos[ch][g];
      const idx0 = Math.floor(readPos) & (bufSize - 1);
      const idx1 = (idx0 + 1) & (bufSize - 1);
      const frac = readPos - Math.floor(readPos);
      const readSample = buf[idx0] * (1 - frac) + buf[idx1] * frac;

      // Apply Hann window (two grains sum to unity at 50% overlap)
      const windowVal = this.hannWindow[phase < grainSize ? phase : 0];
      out += readSample * windowVal;
    }

    return out;
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

    // --- Pitch detection pre-pass (feeds channel 0 into YIN, updates shift ratio) ---
    if (this.params.quantizeEnabled) {
      const inputCh0 = input[0];
      for (let i = 0; i < len; i++) {
        // Feed raw input into YIN buffer for pitch analysis
        this.yinBuffer[this.yinWritePos] = inputCh0[i];
        this.yinWritePos = (this.yinWritePos + 1) & (this.yinBufSize - 1);

        this.pitchDetectCounter++;
        if (this.pitchDetectCounter >= this.pitchDetectInterval) {
          this.pitchDetectCounter = 0;
          const pitch = this.detectPitch();
          if (pitch > 0) {
            this.detectedPitch = pitch;
            const target = this.nearestScaleFreq(
              pitch, this.params.quantizeKey, this.params.quantizeScale
            );
            const rawRatio = target / pitch;
            // Clamp to ±1 octave for safety
            this.quantizeTargetRatio = Math.max(0.5, Math.min(2.0, rawRatio));
          } else {
            // No pitch detected — smoothly return to pass-through
            this.quantizeTargetRatio = 1.0;
          }
        }

        // Smooth the shift ratio (strength-adjusted)
        const strength = this.params.quantizeStrength;
        const effectiveTarget = 1.0 + strength * (this.quantizeTargetRatio - 1.0);
        this.quantizeSmoothedRatio += (effectiveTarget - this.quantizeSmoothedRatio) * 0.005;
      }
    }

    // --- Process each channel independently for true stereo ---
    for (let ch = 0; ch < numChannels; ch++) {
      const channelIn = input[ch];
      const channelOut = output[ch];
      const delayBuf = this.delayBuffers[ch];
      const eqChStates = this.eqStates[ch];

      for (let i = 0; i < len; i++) {
        let sample = channelIn[i];
        const dry = sample;

        // --- EQ ---
        if (this.params.eqEnabled) {
          for (let b = 0; b < 5; b++) {
            const bandType = this.params.eqBands[b].type || 'peak';
            // Always apply lowpass/highpass; for peak/shelf only apply if gain != 0
            if (bandType === 'lowpass' || bandType === 'highpass' || this.params.eqBands[b].gain !== 0) {
              sample = this.biquad(sample, this.eqCoeffs[b], eqChStates[b]);
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

        // --- Spectral Quantize (autotune pitch correction) ---
        if (this.params.quantizeEnabled) {
          sample = this.pitchShiftSample(sample, ch);
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
          const depth = this.params.modDepth * 0.005 * sampleRate; // max ~5ms
          const lfoVal = Math.sin(2 * Math.PI * this.chorusPhase);

          const modDelay = Math.floor(depth * (1 + lfoVal) + 0.002 * sampleRate);
          const modReadIndex = (this.delayWriteIndexes[ch] - modDelay + delayBuf.length) % delayBuf.length;
          const modSample = delayBuf[modReadIndex] || 0;
          sample = sample * 0.7 + modSample * 0.3;
        }

        // --- Lo-Fi ---
        if (this.params.lofiEnabled) {
          // Bitcrush-style sample rate reduction
          const wear = this.params.lofiWear;
          const holdLength = Math.max(1, Math.floor(1 + wear * 15));
          this.lofiHoldCounters[ch]++;
          if (this.lofiHoldCounters[ch] >= holdLength) {
            this.lofiHoldSamples[ch] = sample;
            this.lofiHoldCounters[ch] = 0;
          }
          sample = this.lofiHoldSamples[ch];

          // Wobble: pitch-drift LFO (subtle gain modulation)
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

        // --- Dry/Wet Mix ---
        sample = dry * (1 - this.params.dryWet) + sample * this.params.dryWet;

        // --- Master Gain ---
        sample *= this.params.masterGain;

        // Soft clip output
        sample = Math.tanh(sample);

        channelOut[i] = sample;
      }
    }

    // Advance chorus LFO once per frame (shared across channels)
    if (this.params.modEnabled || this.params.lofiEnabled) {
      for (let i = 0; i < len; i++) {
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
