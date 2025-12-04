# Audio Pipeline Overview

This document describes the current audio processing entry points and related infrastructure in the Quantum Distortion project.

## Main Processing Entry Point

### `process_audio`

**Location**: `quantum_distortion/dsp/pipeline.py`

**Function Signature**:
```python
def process_audio(
    audio: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    key: str = DEFAULT_KEY,
    scale: str = DEFAULT_SCALE,
    snap_strength: float = DEFAULT_SNAP_STRENGTH,
    smear: float = DEFAULT_SMEAR,
    bin_smoothing: bool = DEFAULT_BIN_SMOOTHING,
    pre_quant: bool = True,
    post_quant: bool = True,
    distortion_mode: str = DEFAULT_DISTORTION_MODE,
    distortion_params: Union[Dict[str, Any], None] = None,
    limiter_on: bool = DEFAULT_LIMITER_ON,
    limiter_ceiling_db: float = DEFAULT_LIMITER_CEILING_DB,
    dry_wet: float = DEFAULT_DRY_WET,
    preview_enabled: Union[bool, None] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]
```

**Description**: Main offline processing entry point. Processes audio through the full DSP pipeline:
- Optional preview truncation (if `preview_enabled`)
- Optional spectral pre-quantization
- Time-domain distortion
- Optional spectral post-quantization
- Optional limiter
- Dry/wet mix with input

**Returns**: 
- `np.ndarray`: Processed audio signal
- `Dict[str, np.ndarray]`: Tap buffers containing intermediate stages:
  - `"input"`: Original input audio
  - `"pre_quant"`: Audio after pre-quantization (if enabled)
  - `"post_dist"`: Audio after distortion
  - `"output"`: Final processed output

**Default Parameters** (from `quantum_distortion/config.py`):
- `DEFAULT_SAMPLE_RATE = 48000`
- `DEFAULT_KEY = "D"`
- `DEFAULT_SCALE = "minor"`
- `DEFAULT_SNAP_STRENGTH = 1.0`
- `DEFAULT_SMEAR = 0.1`
- `DEFAULT_BIN_SMOOTHING = True`
- `DEFAULT_DISTORTION_MODE = "wavefold"`
- `DEFAULT_LIMITER_ON = True`
- `DEFAULT_LIMITER_CEILING_DB = -1.0`
- `DEFAULT_DRY_WET = 1.0`

## Audio I/O

### `load_audio`

**Location**: `quantum_distortion/io/audio_io.py`

**Function Signature**:
```python
def load_audio(path: Union[str, Path], target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]
```

**Description**: Loads audio from file using `soundfile`. Returns audio as float32 numpy array and sample rate.

### `save_audio`

**Location**: `quantum_distortion/io/audio_io.py`

**Function Signature**:
```python
def save_audio(path: Union[str, Path], audio: np.ndarray, sr: int) -> None
```

**Description**: Saves audio to file using `soundfile`. Creates parent directories if needed.

## Audio Rendering Scripts

### `scripts/render_cli.py`

**Description**: CLI tool for processing audio files with default parameters.

**Usage**:
```bash
python scripts/render_cli.py --infile <input.wav> --outfile <output.wav>
```

**Functionality**:
- Loads audio file
- Processes through `process_audio` with default parameters
- Saves processed output
- Prints tap buffer information

### `scripts/render_preset.py`

**Description**: CLI tool for processing audio files using named presets.

**Usage**:
```bash
python scripts/render_preset.py --infile <input.wav> --outfile <output.wav> --preset <preset_name>
python scripts/render_preset.py --list-presets  # List available presets
```

**Functionality**:
- Loads audio file
- Retrieves preset configuration from `quantum_distortion/presets.py`
- Processes through `process_audio` with preset parameters
- Saves processed output

### `scripts/preview_visualizers.py`

**Description**: Processes audio and generates visualization plots (spectrum, oscilloscope, phase scope) for each pipeline tap.

**Usage**:
```bash
python scripts/preview_visualizers.py --infile <input.wav> [--outdir <output_dir>]
```

**Functionality**:
- Loads audio file
- Processes through `process_audio`
- Generates visualization plots for each tap stage
- Saves PNG files to output directory (default: `examples/visualizations/`)

## Test Helpers

### Test Audio Generation

Several test files contain helper functions for generating test signals:

- **`tests/test_pipeline.py`**: `_make_sine(freq, sr=44100, seconds=0.2)` - Generates sine wave test signal
- **`tests/test_distortion.py`**: `_make_sine(freq, sr=44100, seconds=0.1)` - Generates sine wave for distortion tests
- **`tests/test_visualizers.py`**: `_make_sine(freq, sr=44100, seconds=0.1)` - Generates sine wave for visualization tests
- **`tests/test_analyses.py`**: `_sine_from_midi(midi, sr=44100, seconds=0.5)` - Generates sine wave from MIDI note number
- **`tests/test_quantization_integration.py`**: `_sine_from_midi(midi, sr=44100, seconds=0.5)` - Generates sine wave from MIDI note number

### Full Pipeline Test Example

**`tests/test_imports_and_io.py`**: Contains `test_roundtrip()` which demonstrates:
- Creating a test sine wave signal
- Saving to file using `save_audio`
- Loading from file using `load_audio`
- Processing through `process_audio`
- Saving processed output
- Validating tap buffers

This test serves as a good reference for building a test harness around `process_audio`.

## Processing Harness

### `process_file_to_file`

**Location**: `quantum_distortion/dsp/harness.py`

**Function Signature**:
```python
def process_file_to_file(
    infile: Path,
    outfile: Path,
    preset: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> None
```

**Description**: Convenience wrapper that loads audio, processes it through `process_audio`, and saves the result. This is the convenience layer used by automated regression tests and scripts that need simple file-to-file processing.

**Features**:
- Supports preset-based processing (loads configuration from `quantum_distortion.presets`)
- Supports default parameters (if no preset is specified)
- Supports parameter overrides via `extra_params`
- Handles audio loading, mono conversion, processing, and saving
- Discards tap buffers (focuses on final output only)

**Usage Example**:
```python
from pathlib import Path
from quantum_distortion.dsp.harness import process_file_to_file

# Process with default parameters
process_file_to_file(
    Path("input.wav"),
    Path("output.wav")
)

# Process with a preset
process_file_to_file(
    Path("input.wav"),
    Path("output.wav"),
    preset="my_preset"
)

# Process with parameter overrides
process_file_to_file(
    Path("input.wav"),
    Path("output.wav"),
    extra_params={"snap_strength": 0.5, "dry_wet": 0.8}
)
```

## STFT Utilities

**Location**: `quantum_distortion/dsp/stft_utils.py`

- `stft_mono()`: Computes complex STFT for mono audio
- `istft_mono()`: Reconstructs audio from complex STFT

These are used internally by `process_audio` and are available for direct use in tests if needed.

