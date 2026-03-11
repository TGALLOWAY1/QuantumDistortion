# Quantum Distortion — Architecture

## Overview

Quantum Distortion is a spectral pitch quantization + distortion audio DSP engine written in Python. It processes audio through a pipeline that combines frequency-domain spectral quantization with time-domain distortion effects.

## Signal Flow

```
Input Audio
  │
  ├─ [Preview Mode] Truncate to first N seconds
  │
  ├─ ensure_mono_float32()
  │
  ├─── Single-band path ──────────────────────────────────────────┐
  │    │                                                          │
  │    ├─ STFT (OLA-compliant, Hann window, 75% overlap)         │
  │    ├─ [Optional] Pre-quantization (spectral bin attraction)   │
  │    ├─ ISTFT → Time-domain distortion (wavefold / tube)       │
  │    ├─ [Optional] STFT → Post-quantization → ISTFT            │
  │    ├─ [Optional] Peak limiter                                 │
  │    └─ Dry/wet mix                                             │
  │                                                               │
  ├─── Multiband path ────────────────────────────────────────────┤
  │    │                                                          │
  │    ├─ Linkwitz-Riley crossover split (default 300 Hz)         │
  │    │                                                          │
  │    ├─ Low band (time-domain only):                            │
  │    │   ├─ Saturation (saturate_lowband)                       │
  │    │   ├─ Mono-maker                                          │
  │    │   └─ Output trim                                         │
  │    │                                                          │
  │    ├─ High band (full STFT pipeline):                         │
  │    │   ├─ STFT                                                │
  │    │   ├─ [Optional] Spectral FX (bitcrush/phase/scramble)    │
  │    │   ├─ [Optional] Creative FX (freeze/formant/harmonic)    │
  │    │   ├─ Pre/post quantization + distortion                  │
  │    │   ├─ Limiter + dry/wet mix                               │
  │    │   └─ ISTFT                                               │
  │    │                                                          │
  │    └─ Recombine: low_processed + high_processed               │
  │                                                               │
  ├─ [Optional] Delta listen (input - processed)                  │
  └─ Output Audio
```

## Module Map

| Module | Purpose |
|---|---|
| `quantum_distortion/dsp/pipeline.py` | Pipeline orchestration: `process_audio()`, multiband routing, STFT processing |
| `quantum_distortion/dsp/quantizer.py` | Spectral magnitude quantization (scale-aware bin attraction + smear) |
| `quantum_distortion/dsp/spectral_fx.py` | Spectral bitcrush, phase dispersal, bin scramble, formant shift |
| `quantum_distortion/dsp/distortion.py` | Time-domain wavefold + tube distortion |
| `quantum_distortion/dsp/saturation.py` | Low-band saturation (`saturate_lowband`) + mono-maker |
| `quantum_distortion/dsp/limiter.py` | Peak limiter with lookahead |
| `quantum_distortion/dsp/crossover.py` | Linkwitz-Riley 4th-order crossover split |
| `quantum_distortion/dsp/stft_utils.py` | Custom OLA-compliant STFT/ISTFT |
| `quantum_distortion/dsp/analyses.py` | Pitch analysis (requires librosa) |
| `quantum_distortion/dsp/harness.py` | File-to-file processing convenience wrapper |
| `quantum_distortion/io/audio_io.py` | Load/save audio via soundfile |
| `quantum_distortion/ui/app_streamlit.py` | Streamlit UI |
| `quantum_distortion/ui/visualizers.py` | Spectrum/oscilloscope/phase scope plots |
| `quantum_distortion/config.py` | Defaults, `PipelineConfig`, shared types, `ensure_mono_float32` |
| `quantum_distortion/presets.py` | JSON preset loader |

## Key Design Decisions

1. **OLA-compliant STFT**: hop_length is always n_fft/4, window is always Hann (sym=False). These are enforced internally and not configurable.

2. **Multiband architecture**: Low band stays in the time domain for tight transients. High band goes through the STFT pipeline for spectral processing. This prevents muddiness in bass frequencies.

3. **Mono processing**: The pipeline converts all audio to mono float32 at entry. Stereo support is not a goal for this prototype.

4. **PipelineConfig**: A dataclass that consolidates all processing parameters, replacing the 20+ keyword arguments pattern. Both interfaces (config object and kwargs) are supported.
