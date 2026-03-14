# Quantum Distortion

<img width="1024" height="559" alt="Quantum Distortion UI Mockup" src="https://github.com/user-attachments/assets/2000d791-a24c-44bf-aecd-5eb478be0ce0" />

**Spectral pitch quantization meets time-domain distortion.** Quantum Distortion is an experimental DSP engine that snaps audio frequencies to a musical scale while applying creative distortion effects, producing unique harmonic textures from any source material.

This Python prototype validates the DSP architecture before building a production JUCE VST plugin. It ships with a Streamlit UI, a CLI renderer, and an Electron/React desktop app for real-time processing.

## Features

### DSP Engine
- **Spectral Pitch Quantization** -- Snap frequency content to any key and scale (major, minor, pentatonic, dorian, mixolydian, harmonic minor) with adjustable snap strength and spectral smearing
- **Distortion Modes** -- Wavefold (mirrored clipping for rich harmonics) and Soft Tube (tanh saturation with warmth control)
- **Multiband Processing** -- Linkwitz-Riley crossover keeps the low band in time-domain for tight transients while the high band runs through the full STFT pipeline
- **Spectral FX** -- Bitcrush, phase dispersal, and bin scrambling for creative frequency-domain effects
- **Peak Limiter** -- Lookahead limiting with configurable ceiling to keep output safe
- **Dry/Wet Mix** -- Blend processed and original signals from subtle glue to full destruction

### Interfaces
- **Electron/React Desktop App** -- Real-time audio processing with knob-based effect modules, spectrum analyzer, and a dynamic FX chain builder
- **Streamlit Web UI** -- Interactive browser-based interface with visualization tap points (spectrum, oscilloscope, phase scope)
- **CLI Renderer** -- Offline file-to-file processing with full parameter control and named presets

### Presets

Four portfolio-ready presets are included:

| Preset | Key/Scale | Style |
|--------|-----------|-------|
| **Chordal Noise Wash** | C minor | Smeared harmonic wash from noisy sources |
| **Controlled Dubstep Growl** | F minor | Aggressive folded bass locked to root + fifth |
| **Perc To Tonal Clang** | D minor | Percussive hits pushed into pitched metallic impacts |
| **Subtle Tube Glue** | C major | Gentle saturation with light quantization |

## Signal Flow

```
Input Audio (mono float32)
    │
    ├─ Single-Band Path ─────────────────────────────────┐
    │   STFT ➜ Pre-Quant ➜ Distortion ➜ Post-Quant ➜ Limiter ➜ Dry/Wet
    │                                                     │
    ├─ Multiband Path ───────────────────────────────────┤
    │   Crossover @ 300 Hz                                │
    │   ├─ Low:  Saturation ➜ Mono-maker ➜ Trim          │
    │   └─ High: STFT ➜ Spectral FX ➜ Quant ➜ Dist ➜ …  │
    │   Recombine                                         │
    │                                                     │
    ▼                                                     │
Output Audio ◄────────────────────────────────────────────┘
```

## Quick Start

### Requirements

- Python 3.10+
- Node.js 18+ (for the Electron UI)

### Python Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CLI Offline Render

```bash
# Render with default parameters
python scripts/render_cli.py \
  --infile examples/example_bass.wav \
  --outfile examples/example_bass_qd.wav

# Render with a named preset
python scripts/render_preset.py \
  --infile examples/example_bass.wav \
  --outfile examples/example_bass_growl.wav \
  --preset "Controlled Dubstep Growl"

# List all presets
python scripts/render_preset.py --list-presets
```

### Streamlit UI

```bash
streamlit run quantum_distortion/ui/app_streamlit.py
```

Open the local URL in your browser, load an audio file, tweak controls, and hit Render.

### Electron Desktop App

```bash
cd ui
npm install
npm run electron:dev      # Development mode
npm run electron:build    # Build production installer
```

## Validation & Profiling

```bash
# Profile the DSP pipeline
python scripts/profile_pipeline.py \
  --infile examples/example_bass.wav

# Measure scale alignment (average cents offset)
python scripts/validate_dsp_metrics.py \
  --infile examples/example_bass.wav \
  --key C --scale minor
```

## Testing

```bash
python -m pytest tests/ -x -q
```

68 tests covering STFT round-trip fidelity, quantizer accuracy, distortion modes, multiband crossover, limiter compliance, preset loading, and more.

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| DSP Core | NumPy, SciPy, Numba (JIT), Matplotlib |
| Audio I/O | SoundFile, Librosa (optional analysis) |
| Web UI | Streamlit |
| Desktop App | Electron 41, React 19, TypeScript, Vite, Tailwind CSS |

## Documentation

- [Architecture](docs/architecture.md) -- Signal flow, module map, design decisions
- [API Reference](docs/api.md) -- `process_audio()`, `PipelineConfig`, I/O, CLI scripts
- [Development Guide](docs/development.md) -- Setup, testing, contributing

## License

See [LICENSE](LICENSE) for details.
