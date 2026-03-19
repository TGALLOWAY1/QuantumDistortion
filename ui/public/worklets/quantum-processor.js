import {
  DEFAULT_RETUNE_PARAMS,
  PolyphonicRetuneCore,
  normalizeRetuneParamPatch,
} from './retune-core.js';

const SCALES = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  pentatonic: [0, 2, 4, 7, 9],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  mixolydian: [0, 2, 4, 5, 7, 9, 10],
  harmonic_minor: [0, 2, 3, 5, 7, 8, 11],
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};

const MAX_CHANNELS = 2;
const TWO_PI = Math.PI * 2;

class QuantumProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.params = {
      bypass: false,
      dryWet: 1.0,
      masterGain: 1.0,

      saturateEnabled: true,
      saturateType: 'tape',
      saturateDrive: 0.5,
      saturateTilt: 0.5,

      saturate2Enabled: true,
      saturate2Type: 'tape',
      saturate2Drive: 0.5,
      saturate2Tilt: 0.5,

      ...DEFAULT_RETUNE_PARAMS,

      lowGain: 1.0,
      highGain: 1.0,
      _devDriveRange: 0.4,

      delayEnabled: false,
      delayTime: 0.25,
      delayFeedback: 0.3,

      modEnabled: false,
      modDepth: 0.5,
      modRate: 1.0,

      lofiEnabled: false,
      lofiWear: 0.5,
      lofiWobble: 0.3,

      eqEnabled: true,
      eqBands: [
        { freq: 60, gain: 0, q: 1.0, type: 'lowshelf' },
        { freq: 250, gain: 0, q: 1.0, type: 'peak' },
        { freq: 1000, gain: 0, q: 1.0, type: 'peak' },
        { freq: 4000, gain: 0, q: 1.0, type: 'peak' },
        { freq: 12000, gain: 0, q: 1.0, type: 'highshelf' },
      ],
      peqInstances: [],
    };

    this.delayBuffers = Array.from({ length: MAX_CHANNELS }, () => new Float32Array(96000));
    this.delayWriteIndexes = new Array(MAX_CHANNELS).fill(0);
    this.chorusPhase = 0;
    this.lofiHoldSamples = new Array(MAX_CHANNELS).fill(0);
    this.lofiHoldCounters = new Array(MAX_CHANNELS).fill(0);

    this.eqStates = Array.from({ length: MAX_CHANNELS }, () =>
      Array.from({ length: 5 }, () => ({ x1: 0, x2: 0, y1: 0, y2: 0 }))
    );
    this.eqCoeffs = Array.from({ length: 5 }, () => ({
      b0: 1, b1: 0, b2: 0, a1: 0, a2: 0,
    }));
    this.eqDirty = true;

    this.peqMaxInstances = 8;
    this.peqMaxFilters = 60;
    this.peqOctaveStart = 2;
    this.peqOctaveEnd = 6;
    this.peqCoeffs = Array.from({ length: this.peqMaxInstances }, () =>
      Array.from({ length: this.peqMaxFilters }, () => ({ b0: 1, b1: 0, b2: 0, a1: 0, a2: 0 }))
    );
    this.peqFilterCounts = new Array(this.peqMaxInstances).fill(0);
    this.peqStates = Array.from({ length: MAX_CHANNELS }, () =>
      Array.from({ length: this.peqMaxInstances }, () =>
        Array.from({ length: this.peqMaxFilters }, () => ({ x1: 0, x2: 0, y1: 0, y2: 0 }))
      )
    );
    this.peqDirty = true;

    this.blockInputScratch = Array.from({ length: MAX_CHANNELS }, () => new Float32Array(128));
    this.blockRetuneScratch = Array.from({ length: MAX_CHANNELS }, () => new Float32Array(128));

    this.analysisSamples = new Float32Array(2048);
    this.analysisWritePos = 0;
    this.frameCount = 0;

    this.retuneCore = new PolyphonicRetuneCore({
      sampleRate,
      maxChannels: MAX_CHANNELS,
    });

    this.port.onmessage = (event) => {
      if (event.data.type !== 'params') {
        return;
      }

      const retunePatch = normalizeRetuneParamPatch(event.data.params, this.params);
      Object.assign(this.params, event.data.params, retunePatch);
      this.retuneCore.setParams(retunePatch);
      this.eqDirty = true;
      this.peqDirty = true;
    };
  }

  static get parameterDescriptors() {
    return [];
  }

  ensureBlockScratch(length) {
    if (this.blockInputScratch[0].length === length) {
      return;
    }
    this.blockInputScratch = Array.from({ length: MAX_CHANNELS }, () => new Float32Array(length));
    this.blockRetuneScratch = Array.from({ length: MAX_CHANNELS }, () => new Float32Array(length));
  }

  computeEQCoeffs() {
    const sr = sampleRate;
    for (let i = 0; i < 5; i++) {
      const band = this.params.eqBands[i];
      const f0 = band.freq;
      const gainDB = band.gain;
      const Q = Math.max(band.q, 0.1);
      const type = band.type || 'peak';

      const A = Math.pow(10, gainDB / 40);
      const w0 = TWO_PI * f0 / sr;
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
          b0 = 1;
          b1 = 0;
          b2 = 0;
          a0 = 1;
          a1 = 0;
          a2 = 0;
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

  computePeqCoeffs() {
    const sr = sampleRate;
    const instances = this.params.peqInstances || [];

    for (let inst = 0; inst < this.peqMaxInstances; inst++) {
      if (inst >= instances.length || !instances[inst].enabled) {
        this.peqFilterCounts[inst] = 0;
        continue;
      }

      const config = instances[inst];
      const scale = SCALES[config.scale] || SCALES.major;
      const key = config.key || 0;
      const amount = config.amount || 0;
      const Q = Math.max(config.q || 2.0, 0.1);
      const isBoost = config.mode === 'boost';
      const gainDB = amount * 18;
      if (gainDB < 0.1) {
        this.peqFilterCounts[inst] = 0;
        continue;
      }

      const inScale = new Set(scale.map(interval => (key + interval) % 12));
      let filterIndex = 0;
      for (let octave = this.peqOctaveStart; octave <= this.peqOctaveEnd; octave++) {
        for (let pc = 0; pc < 12; pc++) {
          const shouldFilter = isBoost ? inScale.has(pc) : !inScale.has(pc);
          if (!shouldFilter || filterIndex >= this.peqMaxFilters) {
            continue;
          }

          const midi = (octave + 1) * 12 + pc;
          const freq = 440 * Math.pow(2, (midi - 69) / 12);
          if (freq < 30 || freq > sr * 0.45) {
            continue;
          }

          const appliedGain = isBoost ? gainDB : -gainDB;
          const A = Math.pow(10, appliedGain / 40);
          const w0 = TWO_PI * freq / sr;
          const sinW = Math.sin(w0);
          const cosW = Math.cos(w0);
          const alpha = sinW / (2 * Q);
          const b0 = 1 + alpha * A;
          const b1 = -2 * cosW;
          const b2 = 1 - alpha * A;
          const a0 = 1 + alpha / A;
          const a1 = -2 * cosW;
          const a2 = 1 - alpha / A;

          this.peqCoeffs[inst][filterIndex] = {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
          };
          filterIndex++;
        }
      }
      this.peqFilterCounts[inst] = filterIndex;
    }

    this.peqDirty = false;
  }

  biquad(sample, coeffs, state) {
    const output = coeffs.b0 * sample + coeffs.b1 * state.x1 + coeffs.b2 * state.x2
      - coeffs.a1 * state.y1 - coeffs.a2 * state.y2;
    state.x2 = state.x1;
    state.x1 = sample;
    state.y2 = state.y1;
    state.y1 = output;
    return output;
  }

  tapeSaturate(sample, drive) {
    const range = this.params._devDriveRange;
    const d = 1 + drive * 20 * range;
    const driven = sample * d;
    const raw = Math.tanh(driven);
    const normalized = raw / Math.tanh(d);
    return normalized * 0.4 + raw * 0.6;
  }

  tubeSaturate(sample, drive) {
    const range = this.params._devDriveRange;
    const d = 1 + drive * 25 * range;
    const driven = sample * d;
    if (driven >= 0) {
      const raw = Math.tanh(driven * 1.2);
      return raw * 0.7 + (raw / Math.tanh(d * 1.2)) * 0.3;
    }
    const raw = Math.tanh(driven * 0.8);
    return raw * 0.7 + (raw / Math.tanh(d * 0.8)) * 0.3;
  }

  wavefold(sample, drive) {
    const range = this.params._devDriveRange;
    const d = 1 + drive * 20 * range;
    return Math.asin(Math.sin(sample * d * Math.PI / 2)) * 2 / Math.PI;
  }

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

    const t = (tilt - 0.5) * 2;
    if (Math.abs(t) > 0.01) {
      sample *= (1 + t * 0.3);
    }

    return sample;
  }

  process(inputs, outputs, _parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const numChannels = Math.min(input.length, output.length, MAX_CHANNELS);
    const length = input[0].length;
    this.ensureBlockScratch(length);

    if (this.params.bypass) {
      for (let ch = 0; ch < numChannels; ch++) {
        output[ch].set(input[ch]);
      }
      for (let ch = numChannels; ch < output.length; ch++) {
        output[ch].fill(0);
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

    for (let ch = 0; ch < numChannels; ch++) {
      const channelInput = input[ch];
      const blockBuffer = this.blockInputScratch[ch];
      const eqStates = this.eqStates[ch];
      const peqInstances = this.params.peqInstances || [];

      for (let i = 0; i < length; i++) {
        let sample = channelInput[i];

        if (this.params.eqEnabled) {
          for (let band = 0; band < 5; band++) {
            const bandType = this.params.eqBands[band].type || 'peak';
            if (bandType === 'lowpass' || bandType === 'highpass' || this.params.eqBands[band].gain !== 0) {
              sample = this.biquad(sample, this.eqCoeffs[band], eqStates[band]);
            }
          }
        }

        for (let inst = 0; inst < peqInstances.length && inst < this.peqMaxInstances; inst++) {
          const filterCount = this.peqFilterCounts[inst];
          if (filterCount <= 0) {
            continue;
          }
          const instanceStates = this.peqStates[ch][inst];
          const instanceCoeffs = this.peqCoeffs[inst];
          for (let filter = 0; filter < filterCount; filter++) {
            sample = this.biquad(sample, instanceCoeffs[filter], instanceStates[filter]);
          }
        }

        if (this.params.saturateEnabled) {
          sample = this.applySaturation(
            sample,
            this.params.saturateDrive,
            this.params.saturateType,
            this.params.saturateTilt,
          );
        }

        blockBuffer[i] = sample;
      }
    }

    const retuneStart = globalThis.performance?.now?.() ?? 0;
    const retunedBlock = this.retuneCore.processBlock(
      this.blockInputScratch,
      numChannels,
      length,
      { lowGain: this.params.lowGain, highGain: this.params.highGain },
    );
    const retuneTimeMs = (globalThis.performance?.now?.() ?? retuneStart) - retuneStart;
    const blockBudgetMs = (length / sampleRate) * 1000;
    this.retuneCore.setRuntimeHealth({ overloaded: retuneTimeMs > blockBudgetMs * 0.85 });

    for (let ch = 0; ch < numChannels; ch++) {
      const channelOutput = output[ch];
      const channelInput = input[ch];
      const delayBuffer = this.delayBuffers[ch];

      for (let i = 0; i < length; i++) {
        const dry = channelInput[i];
        let sample = retunedBlock[ch][i];

        if (this.params.delayEnabled) {
          const delaySamples = Math.floor(this.params.delayTime * sampleRate);
          const readIndex = (this.delayWriteIndexes[ch] - delaySamples + delayBuffer.length) % delayBuffer.length;
          const delayed = delayBuffer[readIndex];
          delayBuffer[this.delayWriteIndexes[ch]] = sample + delayed * this.params.delayFeedback;
          this.delayWriteIndexes[ch] = (this.delayWriteIndexes[ch] + 1) % delayBuffer.length;
          sample += delayed * 0.5;
        }

        if (this.params.modEnabled) {
          const depth = this.params.modDepth * 0.005 * sampleRate;
          const lfoValue = Math.sin(TWO_PI * this.chorusPhase);
          const modDelay = Math.floor(depth * (1 + lfoValue) + 0.002 * sampleRate);
          const modReadIndex = (this.delayWriteIndexes[ch] - modDelay + delayBuffer.length) % delayBuffer.length;
          const modSample = delayBuffer[modReadIndex] || 0;
          sample = sample * 0.7 + modSample * 0.3;
        }

        if (this.params.lofiEnabled) {
          const holdLength = Math.max(1, Math.floor(1 + this.params.lofiWear * 15));
          this.lofiHoldCounters[ch]++;
          if (this.lofiHoldCounters[ch] >= holdLength) {
            this.lofiHoldSamples[ch] = sample;
            this.lofiHoldCounters[ch] = 0;
          }
          sample = this.lofiHoldSamples[ch];

          if (this.params.lofiWobble > 0.01) {
            const wobble = Math.sin(this.chorusPhase * 0.3 * TWO_PI) * this.params.lofiWobble * 0.15;
            sample *= (1 + wobble);
          }
        }

        if (this.params.saturate2Enabled) {
          sample = this.applySaturation(
            sample,
            this.params.saturate2Drive,
            this.params.saturate2Type,
            this.params.saturate2Tilt,
          );
        }

        sample = dry * (1 - this.params.dryWet) + sample * this.params.dryWet;
        sample *= this.params.masterGain;
        channelOutput[i] = Math.tanh(sample);
      }
    }

    if (this.params.modEnabled || this.params.lofiEnabled) {
      this.chorusPhase += (this.params.modRate * length) / sampleRate;
      if (this.chorusPhase > 1) {
        this.chorusPhase -= Math.floor(this.chorusPhase);
      }
    }

    for (let ch = numChannels; ch < output.length; ch++) {
      output[ch].fill(0);
    }

    this.sendAnalysis(output[0]);
    return true;
  }

  sendAnalysis(channelData) {
    for (let i = 0; i < channelData.length; i++) {
      this.analysisSamples[this.analysisWritePos] = channelData[i];
      this.analysisWritePos = (this.analysisWritePos + 1) % this.analysisSamples.length;
    }

    this.frameCount++;
    if (this.frameCount % 18 !== 0) {
      return;
    }

    const ordered = new Float32Array(this.analysisSamples.length);
    const start = this.analysisWritePos;
    for (let i = 0; i < ordered.length; i++) {
      ordered[i] = this.analysisSamples[(start + i) % this.analysisSamples.length];
    }

    this.port.postMessage({ type: 'analysis', samples: ordered });
    this.port.postMessage({
      type: 'retune-status',
      status: this.retuneCore.getStatus(),
    });
  }
}

registerProcessor('quantum-processor', QuantumProcessor);
