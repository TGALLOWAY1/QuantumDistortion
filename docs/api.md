# Quantum Distortion — API Reference

## process_audio()

**Location**: `quantum_distortion/dsp/pipeline.py`

The main entry point for audio processing. Can be called with individual keyword arguments or a `PipelineConfig` object.

### Using PipelineConfig (recommended)

```python
from quantum_distortion.config import PipelineConfig
from quantum_distortion.dsp.pipeline import process_audio

config = PipelineConfig(
    key="C",
    scale="minor",
    snap_strength=0.8,
    distortion_mode="wavefold",
    distortion_params={"fold_amount": 3.0, "drive": 1.0},
    use_multiband=True,
)

processed, taps = process_audio(audio, sr=48000, pipeline_config=config)
```

### Using keyword arguments (backward-compatible)

```python
processed, taps = process_audio(
    audio, sr=48000,
    key="C", scale="minor",
    snap_strength=0.8,
    distortion_mode="wavefold",
    distortion_params={"fold_amount": 3.0, "drive": 1.0},
    use_multiband=True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | `np.ndarray` | required | Input audio (mono or stereo, converted to mono internally) |
| `sr` | `int` | 48000 | Sample rate in Hz |
| `key` | `str` | "D" | Musical key (e.g., "C", "D#", "Bb") |
| `scale` | `str` | "minor" | Scale name: major, minor, pentatonic, dorian, mixolydian, harmonic_minor |
| `snap_strength` | `float` | 1.0 | Bin attraction strength (0.0 = none, 1.0 = full) |
| `smear` | `float` | 0.1 | Energy smear around target bins (0.0 = direct, 1.0 = full smear) |
| `bin_smoothing` | `bool` | True | Apply moving-average smoothing after quantization |
| `pre_quant` | `bool` | True | Enable pre-distortion spectral quantization |
| `post_quant` | `bool` | True | Enable post-distortion spectral quantization |
| `distortion_mode` | `str` | "wavefold" | Distortion type: "wavefold" or "tube" |
| `distortion_params` | `dict` | {} | Mode-specific params (fold_amount, bias, drive, warmth) |
| `limiter_on` | `bool` | True | Enable peak limiter |
| `limiter_ceiling_db` | `float` | -1.0 | Limiter ceiling in dB |
| `dry_wet` | `float` | 1.0 | Mix ratio (0.0 = dry, 1.0 = fully processed) |
| `use_multiband` | `bool` | False | Enable multiband processing |
| `crossover_hz` | `float` | 300.0 | Crossover frequency for multiband split |
| `lowband_drive` | `float` | 1.0 | Low-band saturation drive |
| `spectral_fx_mode` | `str\|None` | None | Spectral FX: "bitcrush", "phase_dispersal", "bin_scramble" |
| `spectral_fx_strength` | `float` | 0.0 | Spectral FX intensity (0.0-1.0) |
| `spectral_freeze` | `bool` | False | Freeze magnitudes to first frame |
| `formant_shift` | `float` | 0.0 | Formant shift in semitones |
| `harmonic_lock_hz` | `float` | 0.0 | Harmonic locking fundamental frequency |
| `passthrough_test` | `bool` | False | Bypass all FX for null test verification |
| `pipeline_config` | `PipelineConfig\|None` | None | Config object (overrides all kwargs when provided) |

### Returns

- `np.ndarray`: Processed audio (mono float32)
- `Dict[str, np.ndarray]`: Tap buffers: `"input"`, `"pre_quant"`, `"post_dist"`, `"output"`

## PipelineConfig

**Location**: `quantum_distortion/config.py`

Dataclass that consolidates all processing parameters.

```python
from quantum_distortion.config import PipelineConfig

# From defaults
config = PipelineConfig()

# From a named preset
config = PipelineConfig.from_preset("aggressive_growl")
```

## Audio I/O

**Location**: `quantum_distortion/io/audio_io.py`

```python
from quantum_distortion.io.audio_io import load_audio, save_audio

audio, sr = load_audio("input.wav")  # Returns float32 array + sample rate
save_audio("output.wav", audio, sr)   # Creates parent dirs if needed
```

## File-to-File Processing

**Location**: `quantum_distortion/dsp/harness.py`

```python
from quantum_distortion.dsp.harness import process_file_to_file

process_file_to_file(Path("in.wav"), Path("out.wav"))                    # Defaults
process_file_to_file(Path("in.wav"), Path("out.wav"), preset="my_preset") # Preset
```

## CLI Scripts

| Script | Description |
|---|---|
| `scripts/render_cli.py` | Process audio with default parameters |
| `scripts/render_preset.py` | Process audio with a named preset |
| `scripts/profile_pipeline.py` | Profile pipeline runtime |
| `scripts/preview_visualizers.py` | Generate visualization PNGs |
| `scripts/generate_test_fixtures.py` | Generate synthetic test audio |
| `scripts/validate_dsp_metrics.py` | Validate DSP quality metrics |
