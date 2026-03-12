/**
 * Quantum Distortion - Realtime AudioWorklet Processor
 *
 * Runs in the audio thread. Implements:
 * - Wavefold distortion
 * - Soft-tube saturation
 * - Spectral quantization (FFT-based pitch snapping)
 * - Bitcrush / Lo-Fi
 * - Simple delay + feedback
 * - Chorus modulation
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

class QuantumProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Parameters controlled via message port
    this.params = {
      // Global
      bypass: false,
      dryWet: 1.0,
      masterGain: 1.0,

      // Saturation
      saturateEnabled: true,
      saturateType: 'tape',     // tape, tube, wavefold
      saturateDrive: 0.5,
      saturateTilt: 0.5,

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

      // EQ bands (5-band parametric)
      eqEnabled: true,
      eqBands: [
        { freq: 60, gain: 0, q: 1.0 },
        { freq: 250, gain: 0, q: 1.0 },
        { freq: 1000, gain: 0, q: 1.0 },
        { freq: 4000, gain: 0, q: 1.0 },
        { freq: 12000, gain: 0, q: 1.0 },
      ],
    };

    // Delay buffer (max 2 seconds at 48kHz)
    this.delayBuffer = new Float32Array(96000);
    this.delayWriteIndex = 0;

    // Chorus LFO phase
    this.chorusPhase = 0;

    // Lo-Fi sample-and-hold state
    this.lofiHoldSample = 0;
    this.lofiHoldCounter = 0;

    // Biquad filter states for EQ (5 bands, each with 2-sample state)
    this.eqStates = Array.from({ length: 5 }, () => ({
      x1: 0, x2: 0, y1: 0, y2: 0,
    }));
    this.eqCoeffs = Array.from({ length: 5 }, () => ({
      b0: 1, b1: 0, b2: 0, a1: 0, a2: 0,
    }));
    this.eqDirty = true;

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

  // Compute biquad peaking EQ coefficients
  computeEQCoeffs() {
    const sr = sampleRate;
    for (let i = 0; i < 5; i++) {
      const band = this.params.eqBands[i];
      const f0 = band.freq;
      const gainDB = band.gain;
      const Q = Math.max(band.q, 0.1);

      const A = Math.pow(10, gainDB / 40);
      const w0 = 2 * Math.PI * f0 / sr;
      const sinW = Math.sin(w0);
      const cosW = Math.cos(w0);
      const alpha = sinW / (2 * Q);

      const b0 = 1 + alpha * A;
      const b1 = -2 * cosW;
      const b2 = 1 - alpha * A;
      const a0 = 1 + alpha / A;
      const a1 = -2 * cosW;
      const a2 = 1 - alpha / A;

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

  // Tape saturation: soft-clip with asymmetric harmonics
  tapeSaturate(x, drive) {
    const d = 1 + drive * 4;
    const driven = x * d;
    return Math.tanh(driven) / Math.tanh(d);
  }

  // Tube saturation: asymmetric soft-clip
  tubeSaturate(x, drive) {
    const d = 1 + drive * 6;
    const driven = x * d;
    if (driven >= 0) {
      return Math.tanh(driven * 1.2) / Math.tanh(d * 1.2);
    }
    return Math.tanh(driven * 0.8) / Math.tanh(d * 0.8);
  }

  // Wavefold distortion
  wavefold(x, drive) {
    const d = 1 + drive * 5;
    let s = x * d;
    // Triangle wavefold
    s = Math.asin(Math.sin(s * Math.PI / 2)) * 2 / Math.PI;
    return s;
  }

  process(inputs, outputs, _parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const channelData = input[0];
    const outData = output[0];
    const len = channelData.length;

    if (this.params.bypass) {
      for (let i = 0; i < len; i++) {
        outData[i] = channelData[i];
      }
      this.sendAnalysis(channelData);
      return true;
    }

    if (this.eqDirty) {
      this.computeEQCoeffs();
    }

    for (let i = 0; i < len; i++) {
      let sample = channelData[i];
      const dry = sample;

      // --- EQ ---
      if (this.params.eqEnabled) {
        for (let b = 0; b < 5; b++) {
          if (this.params.eqBands[b].gain !== 0) {
            sample = this.biquad(sample, this.eqCoeffs[b], this.eqStates[b]);
          }
        }
      }

      // --- Saturation ---
      if (this.params.saturateEnabled) {
        const drive = this.params.saturateDrive;
        switch (this.params.saturateType) {
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

        // Tilt: simple 1-pole high-shelf approximation
        const tilt = (this.params.saturateTilt - 0.5) * 2;
        if (Math.abs(tilt) > 0.01) {
          // Positive tilt = brighter, negative = darker
          sample = sample * (1 + tilt * 0.3);
        }
      }

      // --- Delay ---
      if (this.params.delayEnabled) {
        const delaySamples = Math.floor(this.params.delayTime * sampleRate);
        const readIndex = (this.delayWriteIndex - delaySamples + this.delayBuffer.length) % this.delayBuffer.length;
        const delayed = this.delayBuffer[readIndex];
        this.delayBuffer[this.delayWriteIndex] = sample + delayed * this.params.delayFeedback;
        this.delayWriteIndex = (this.delayWriteIndex + 1) % this.delayBuffer.length;
        sample = sample + delayed * 0.5;
      }

      // --- Chorus Modulation ---
      if (this.params.modEnabled) {
        const depth = this.params.modDepth * 0.005 * sampleRate; // max ~5ms
        const lfoVal = Math.sin(2 * Math.PI * this.chorusPhase);
        this.chorusPhase += this.params.modRate / sampleRate;
        if (this.chorusPhase > 1) this.chorusPhase -= 1;

        const modDelay = Math.floor(depth * (1 + lfoVal) + 0.002 * sampleRate);
        const modReadIndex = (this.delayWriteIndex - modDelay + this.delayBuffer.length) % this.delayBuffer.length;
        const modSample = this.delayBuffer[modReadIndex] || 0;
        sample = sample * 0.7 + modSample * 0.3;
      }

      // --- Lo-Fi ---
      if (this.params.lofiEnabled) {
        // Bitcrush-style sample rate reduction
        const wear = this.params.lofiWear;
        const holdLength = Math.max(1, Math.floor(1 + wear * 15));
        this.lofiHoldCounter++;
        if (this.lofiHoldCounter >= holdLength) {
          this.lofiHoldSample = sample;
          this.lofiHoldCounter = 0;
        }
        sample = this.lofiHoldSample;

        // Wobble: pitch-drift LFO (subtle gain modulation for simplicity)
        const wobble = this.params.lofiWobble;
        if (wobble > 0.01) {
          const wobbleLFO = Math.sin(this.chorusPhase * 0.3 * 2 * Math.PI) * wobble * 0.15;
          sample *= (1 + wobbleLFO);
        }
      }

      // --- Dry/Wet Mix ---
      sample = dry * (1 - this.params.dryWet) + sample * this.params.dryWet;

      // --- Master Gain ---
      sample *= this.params.masterGain;

      // Soft clip output
      sample = Math.tanh(sample);

      outData[i] = sample;
    }

    // Copy to other output channels if present
    for (let ch = 1; ch < output.length; ch++) {
      output[ch].set(outData);
    }

    this.sendAnalysis(outData);
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
