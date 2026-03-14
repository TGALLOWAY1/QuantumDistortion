/**
 * Audio Engine - manages the Web Audio graph and AudioWorklet
 */

export type FilterType = 'peak' | 'lowshelf' | 'highshelf' | 'lowpass' | 'highpass';

export interface EQBand {
  freq: number;
  gain: number;
  q: number;
  type: FilterType;
}

export type FxType = 'saturate' | 'quantize' | 'delay' | 'modulate' | 'lofi' | 'saturate2';

export interface FxSlot {
  id: string;
  type: FxType;
}

export const FX_CATALOG: Record<FxType, { label: string; color: string }> = {
  saturate:  { label: 'Saturate',   color: '#c45e3e' },
  quantize:  { label: 'Quantize',   color: '#4ec48a' },
  delay:     { label: 'Delay',      color: '#3eafc4' },
  modulate:  { label: 'Modulate',   color: '#8a5ec4' },
  lofi:      { label: 'Lo-Fi',      color: '#8a7e5e' },
  saturate2: { label: 'Saturate 2', color: '#d47a5e' },
};

export interface EngineParams {
  bypass: boolean;
  dryWet: number;
  masterGain: number;

  saturateEnabled: boolean;
  saturateType: 'tape' | 'tube' | 'wavefold';
  saturateDrive: number;
  saturateTilt: number;

  saturate2Enabled: boolean;
  saturate2Type: 'tape' | 'tube' | 'wavefold';
  saturate2Drive: number;
  saturate2Tilt: number;

  quantizeEnabled: boolean;
  quantizeKey: number;
  quantizeScale: string;
  quantizeStrength: number;
  quantizeSubEnabled: boolean;
  quantizeSubSource: 'root' | 'manual' | 'scale_degree';
  quantizeSubNote: number;
  quantizeSubDegree: number;
  quantizeSubOctave: number;
  quantizeSubLevel: number;
  quantizeAirMix: number;

  lowGain: number;
  highGain: number;

  _devDriveRange: number;

  delayEnabled: boolean;
  delayTime: number;
  delayFeedback: number;

  modEnabled: boolean;
  modDepth: number;
  modRate: number;

  lofiEnabled: boolean;
  lofiWear: number;
  lofiWobble: number;

  eqEnabled: boolean;
  eqBands: EQBand[];
}

export const DEFAULT_PARAMS: EngineParams = {
  bypass: false,
  dryWet: 1.0,
  masterGain: 1.0,

  saturateEnabled: true,
  saturateType: 'tape',
  saturateDrive: 0.3,
  saturateTilt: 0.5,

  saturate2Enabled: true,
  saturate2Type: 'tape',
  saturate2Drive: 0.5,
  saturate2Tilt: 0.5,

  quantizeEnabled: false,
  quantizeKey: 0,
  quantizeScale: 'major',
  quantizeStrength: 0.7,
  quantizeSubEnabled: true,
  quantizeSubSource: 'root',
  quantizeSubNote: 0,
  quantizeSubDegree: 0,
  quantizeSubOctave: 2,
  quantizeSubLevel: 0.35,
  quantizeAirMix: 1.0,

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
};

export type AnalysisCallback = (samples: Float32Array) => void;

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private sourceNode: AudioBufferSourceNode | null = null;
  private analyser: AnalyserNode | null = null;
  private audioBuffer: AudioBuffer | null = null;
  private isPlaying = false;
  private onAnalysis: AnalysisCallback | null = null;
  private startTime = 0;
  private startOffset = 0;

  async init() {
    this.ctx = new AudioContext({ sampleRate: 48000 });
    await this.ctx.audioWorklet.addModule('/worklets/quantum-processor.js');

    this.workletNode = new AudioWorkletNode(this.ctx, 'quantum-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2],
    });

    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 4096;
    this.analyser.smoothingTimeConstant = 0.8;

    this.workletNode.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);

    this.workletNode.port.onmessage = (e) => {
      if (e.data.type === 'analysis' && this.onAnalysis) {
        this.onAnalysis(e.data.samples);
      }
    };
  }

  setAnalysisCallback(cb: AnalysisCallback) {
    this.onAnalysis = cb;
  }

  getAnalyser(): AnalyserNode | null {
    return this.analyser;
  }

  getContext(): AudioContext | null {
    return this.ctx;
  }

  updateParams(params: Partial<EngineParams>) {
    this.workletNode?.port.postMessage({ type: 'params', params });
  }

  async loadAudioBuffer(arrayBuffer: ArrayBuffer): Promise<{ duration: number; sampleRate: number }> {
    if (!this.ctx) throw new Error('Engine not initialized');
    this.audioBuffer = await this.ctx.decodeAudioData(arrayBuffer);
    return {
      duration: this.audioBuffer.duration,
      sampleRate: this.audioBuffer.sampleRate,
    };
  }

  play(offset = 0) {
    if (!this.ctx || !this.audioBuffer || !this.workletNode) return;
    this.stop();

    this.sourceNode = this.ctx.createBufferSource();
    this.sourceNode.buffer = this.audioBuffer;
    this.sourceNode.loop = true;
    this.sourceNode.connect(this.workletNode);

    this.sourceNode.onended = () => {
      // Only fires if stop() was called (since loop=true won't end naturally)
      this.isPlaying = false;
    };

    this.startOffset = offset;
    this.startTime = this.ctx.currentTime;
    this.sourceNode.start(0, offset);
    this.isPlaying = true;
  }

  stop() {
    if (this.sourceNode) {
      try {
        this.sourceNode.stop();
        this.sourceNode.disconnect();
      } catch {
        // ignore
      }
      this.sourceNode = null;
    }
    this.isPlaying = false;
  }

  restart() {
    if (!this.audioBuffer) return;
    this.play(0);
  }

  get playing(): boolean {
    return this.isPlaying;
  }

  get currentTime(): number {
    if (!this.ctx || !this.isPlaying) return this.startOffset;
    const elapsed = this.startOffset + (this.ctx.currentTime - this.startTime);
    const dur = this.audioBuffer?.duration ?? 1;
    return dur > 0 ? elapsed % dur : 0;
  }

  get duration(): number {
    return this.audioBuffer?.duration ?? 0;
  }

  async suspend() {
    await this.ctx?.suspend();
  }

  async resume() {
    await this.ctx?.resume();
  }

  destroy() {
    this.stop();
    this.workletNode?.disconnect();
    this.analyser?.disconnect();
    this.ctx?.close();
  }
}
