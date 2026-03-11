# Milestone M0: Project Skeleton

## Summary

This milestone establishes the foundational structure for the Quantum Distortion MVP project. All core directories, configuration files, and stub implementations have been created to support the development of a Python prototype for validating the DSP design before building a JUCE VST plugin.

## Prompts Executed

### PROMPT 0.1 — Create Base Repo Structure
- Created complete directory tree with all required modules
- Established `quantum_distortion` package with proper `__init__.py` files
- Created subdirectories: `dsp/`, `io/`, `ui/`
- Added `scripts/`, `tests/`, and `examples/` directories
- Verified package imports successfully: `import quantum_distortion`

### PROMPT 0.2 — Populate README.md
- Updated README with project description
- Documented MVP features:
  - Load audio files
  - Process through pre-quant → distortion → post-quant → limiter
  - Visualize tap points (spectrum, oscilloscope, phase scope)
- Noted that this is a Python prototype for DSP validation

### PROMPT 0.3 — Create requirements.txt
- Added all required dependencies:
  - `numpy`, `scipy` - Scientific computing
  - `librosa`, `soundfile` - Audio processing and I/O
  - `matplotlib` - Visualization
  - `streamlit` - Web UI framework
  - `pytest` - Testing framework

### PROMPT 0.4 — Create config.py
- Defined default configuration constants:
  - `DEFAULT_SAMPLE_RATE = 44100`
  - Pitch quantization settings: `DEFAULT_KEY`, `DEFAULT_SCALE`, `DEFAULT_SNAP_STRENGTH`, `DEFAULT_SMEAR`, `DEFAULT_BIN_SMOOTHING`
  - `DEFAULT_DISTORTION_MODE = "wavefold"`
  - Limiter settings: `DEFAULT_LIMITER_ON`, `DEFAULT_LIMITER_CEILING_DB`
  - `DEFAULT_DRY_WET = 1.0`

### PROMPT 0.5 — Create audio I/O utils
- Implemented `load_audio()` function:
  - Loads audio files using `soundfile`
  - Returns audio as float32 numpy array and sample rate
- Implemented `save_audio()` function:
  - Saves audio files with automatic directory creation
  - Uses `soundfile` for writing

### PROMPT 0.6 — Create pipeline stub
- Created `process_audio()` function with comprehensive parameter set:
  - Audio input and sample rate
  - Pitch quantization parameters (key, scale, snap_strength, smear, bin_smoothing)
  - Processing stage toggles (pre_quant, post_quant)
  - Distortion mode and parameters
  - Limiter settings
  - Dry/wet mix control
- Returns processed audio and tap points dictionary for visualization
- Currently implements pass-through (stub) with all tap points populated

### PROMPT 0.7 — Create CLI renderer
- Implemented `render_cli.py` command-line tool:
  - Accepts `--infile`/`-i` and `--outfile`/`-o` arguments
  - Loads audio, processes through pipeline, saves output
  - Prints processing information and available tap buffers

### PROMPT 0.8 — Add basic test
- Created `test_roundtrip()` test function:
  - Generates test sine wave (220 Hz, 0.5 seconds)
  - Tests audio save/load roundtrip
  - Verifies sample rate and shape preservation
  - Tests pipeline processing
  - Validates all tap buffers are present (input, pre_quant, post_dist, output)

## Files Created/Modified

### New Files
- `quantum_distortion/__init__.py` - Package initialization with version
- `quantum_distortion/config.py` - Configuration constants
- `quantum_distortion/dsp/__init__.py` - DSP module initialization
- `quantum_distortion/dsp/pipeline.py` - Audio processing pipeline stub
- `quantum_distortion/io/__init__.py` - I/O module initialization
- `quantum_distortion/io/audio_io.py` - Audio file I/O utilities
- `quantum_distortion/ui/__init__.py` - UI module initialization
- `scripts/render_cli.py` - Command-line renderer
- `tests/test_imports_and_io.py` - Basic roundtrip test
- `requirements.txt` - Project dependencies

### Modified Files
- `README.md` - Updated with project description and MVP features

## Project Structure

```
QuantumDistortion/
├── README.md
├── requirements.txt
├── quantum_distortion/
│   ├── __init__.py
│   ├── config.py
│   ├── dsp/
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── io/
│   │   ├── __init__.py
│   │   └── audio_io.py
│   └── ui/
│       └── __init__.py
├── scripts/
│   └── render_cli.py
├── tests/
│   └── test_imports_and_io.py
└── examples/
```

## Verification

- ✅ All directories and files created
- ✅ Package imports successfully: `import quantum_distortion`
- ✅ Git repository initialized
- ✅ No linting errors
- ✅ Python 3.11+ requirement documented

## Next Steps

The project skeleton is complete and ready for:
1. Implementation of spectral pitch quantization algorithms
2. Implementation of distortion algorithms (wavefold mode)
3. Implementation of limiter
4. Development of Streamlit UI for visualization
5. Integration of all components into the processing pipeline

