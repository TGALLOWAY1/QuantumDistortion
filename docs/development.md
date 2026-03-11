# Quantum Distortion — Development Guide

## Prerequisites

- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/)

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd QuantumDistortion

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package    | Purpose                          |
|------------|----------------------------------|
| numpy      | Array operations, DSP math       |
| scipy      | Crossover filters, convolution   |
| soundfile  | WAV/AIFF file I/O               |
| matplotlib | Visualization (spectrum, phase)  |
| streamlit  | Web UI                           |
| numba      | JIT compilation for hot paths    |
| librosa    | Pitch analysis (optional)        |
| pytest     | Test runner                      |

## Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -x -q

# Skip tests that require librosa (heavy dependency)
python -m pytest tests/ -x -q \
  --ignore=tests/test_analyses.py \
  --ignore=tests/test_quantization_integration.py

# Run a specific test file
python -m pytest tests/test_pipeline_core.py -v
```

All core tests should pass before committing. The test suite covers:
- STFT/ISTFT round-trip fidelity (null test)
- Spectral quantization accuracy
- Distortion modes (wavefold, tube)
- Multiband crossover reconstruction
- Limiter ceiling compliance
- Preset loading and rendering
- PipelineConfig dataclass behavior

## Project Structure

```
quantum_distortion/       # Main package
├── dsp/                  # DSP processing core
│   ├── pipeline.py       # Main orchestrator
│   ├── quantizer.py      # Spectral quantization
│   ├── spectral_fx.py    # Frequency-domain FX
│   ├── distortion.py     # Wavefold + tube distortion
│   ├── saturation.py     # Low-band saturation
│   ├── limiter.py        # Peak limiter
│   ├── crossover.py      # Multiband split
│   ├── stft_utils.py     # OLA-compliant STFT/ISTFT
│   ├── analyses.py       # Pitch analysis
│   └── harness.py        # File-to-file wrapper
├── io/
│   └── audio_io.py       # Load/save audio
├── ui/
│   ├── app_streamlit.py  # Streamlit UI
│   └── visualizers.py    # Visualization plots
├── config.py             # Defaults, PipelineConfig, type aliases
└── presets.py            # JSON preset loader
scripts/                  # CLI tools
tests/                    # Test suite
docs/                     # Documentation
presets/                  # Preset JSON files
```

See [architecture.md](architecture.md) for signal flow and design decisions.
See [api.md](api.md) for the public API reference.

## CLI Scripts

```bash
# Render with default parameters
python scripts/render_cli.py -i input.wav -o output.wav

# Render with a named preset
python scripts/render_preset.py -i input.wav -o output.wav -p "Controlled Dubstep Growl"

# List available presets
python scripts/render_preset.py --list-presets

# Profile pipeline performance
python scripts/profile_pipeline.py --infile input.wav

# Validate DSP metrics (scale alignment)
python scripts/validate_dsp_metrics.py --infile input.wav --key C --scale minor
```

## Contributing

1. Create a feature branch from `master`
2. Make changes and ensure all tests pass
3. Keep commits focused and descriptive
4. Open a pull request for review
