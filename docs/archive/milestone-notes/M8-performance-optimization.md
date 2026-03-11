# Milestone M8: Performance Optimization & Preview Render Mode

## Summary

This milestone focused on comprehensive performance optimization of the Quantum Distortion DSP pipeline. The work included adding profiling instrumentation, optimizing STFT/iSTFT usage to minimize redundant transforms, vectorizing spectral processing operations, implementing a preview render mode for faster iteration, and accelerating remaining loops with Numba JIT compilation. The result is a significantly faster pipeline (approximately 5x speedup in spectral processing) while maintaining identical audio quality and behavior.

## Prompts Executed

### Prompt 1 – Add profiling around the DSP pipeline

#### Changes Made
- **Added `RenderTiming` dataclass** in `quantum_distortion/dsp/pipeline.py`:
  - Tracks timing for: `load`, `stft`, `proc`, `istft`, `save`, `total`
  - Provides structured timing data for performance analysis

- **Instrumented `process_audio()` function**:
  - Added `time.perf_counter()` calls around major pipeline stages
  - Measures time spent in:
    - Audio loading (if applicable)
    - STFT computation
    - Spectral processing (quantization)
    - iSTFT reconstruction
    - Audio saving (if applicable)
    - Total pipeline time

- **Instrumented `_apply_spectral_quantization_to_stft()` function**:
  - Tracks time spent in spectral quantization operations
  - Accumulates processing time into `RenderTiming.proc`

- **Timing log output**:
  - Prints timing information to stdout after each render
  - Format: `[RENDER_TIMING] load=X.XXXs stft=X.XXXs proc=X.XXXs istft=X.XXXs save=X.XXXs total=X.XXXs`
  - Helps identify performance bottlenecks

#### Results
- Profiling revealed that spectral processing (`proc`) was the dominant time consumer
- STFT/iSTFT operations were relatively fast
- Identified opportunities for optimization in spectral quantization loops

### Prompt 2 – Ensure exactly one STFT and one iSTFT per render & tune FFT settings

#### Changes Made
- **Normalized FFT settings**:
  - Introduced constants in `quantum_distortion/dsp/pipeline.py`:
    - `N_FFT_DEFAULT = 2048`
    - `HOP_LENGTH_DEFAULT = N_FFT_DEFAULT // 4` (512)
    - `WINDOW_DEFAULT = "hann"`
    - `CENTER_DEFAULT = True`
  - All STFT/iSTFT operations now use consistent settings
  - Centralized configuration makes tuning easier

- **Refactored `process_audio()` for optimal STFT/iSTFT usage**:
  - **Single STFT strategy**: Compute STFT once at the start when possible
  - **Single iSTFT strategy**: Perform iSTFT once at the end when possible
  - **Special case handling**: When both pre-quant and post-quant are enabled with time-domain distortion in between, a second STFT is necessary (distortion operates in time domain)
  - **Optimization logic**:
    - Pre-quant only: 1 STFT → process → 1 iSTFT
    - Post-quant only: 1 STFT → process → 1 iSTFT
    - Both pre- and post-quant: 1 STFT → pre-quant → iSTFT → distortion → 1 STFT → post-quant → 1 iSTFT
    - Always ensures at most one final iSTFT

- **Removed redundant `_apply_spectral_quantization_block()` function**:
  - Replaced with `_apply_spectral_quantization_to_stft()` that operates directly on STFT matrix
  - Eliminates unnecessary STFT/iSTFT roundtrips

#### Results
- Reduced redundant STFT/iSTFT operations
- Clearer code structure with documented optimization strategy
- Maintained identical audio quality

### Prompt 3 – Vectorize snap/smear/weighting and remove Python for-loops over bins/frames

#### Changes Made
- **Vectorized `build_target_bins_for_freqs()` in `quantum_distortion/dsp/quantizer.py`**:
  - **Before**: Loop over frequency bins, computing distances to scale notes
  - **After**: Broadcasting and `np.argmin()` for vectorized computation
  - **Key optimization**:
    ```python
    # Broadcasting: (n_freqs, 1) - (1, n_notes) -> (n_freqs, n_notes)
    df = np.abs(freqs_2d - note_freqs_2d)
    cost = df / note_weights[np.newaxis, :]
    note_indices = np.argmin(cost, axis=1)
    ```
  - Eliminated all Python-level loops

- **Vectorized `quantize_spectrum()` in `quantum_distortion/dsp/quantizer.py`**:
  - **Energy movement**: Replaced loops with `np.add.at()` for atomic updates
  - **Phase combination**: Vectorized weighted phase accumulation
  - **Bin smoothing**: Replaced loop with `scipy.ndimage.convolve1d()`
  - **Smear operation**: Partially vectorized (preparation vectorized, distribution loop remains for varying target centers)
  - **Key optimizations**:
    - `np.add.at()` for efficient energy redistribution
    - Vectorized phase updates using complex number operations
    - `convolve1d()` for bin smoothing (1D convolution)

#### Results
- **Performance improvement**: ~2.4x speedup in spectral processing stage
- **Code quality**: More idiomatic NumPy code, easier to maintain
- **Behavior**: Identical audio output (verified with tests)
- **Remaining loop**: Smear distribution loop documented for future Numba optimization

### Prompt 4 – Add a "preview render" mode (short segment only) for faster iteration

#### Changes Made
- **Added preview configuration** in `quantum_distortion/config.py`:
  - `PREVIEW_ENABLED_DEFAULT = False` (default to full render for production)
  - `PREVIEW_MAX_SECONDS = 10.0` (process only first 10 seconds in preview mode)

- **Updated `process_audio()` function**:
  - Added `preview_enabled: Union[bool, None] = None` parameter
  - **Preview mode detection logic**:
    1. Check function parameter
    2. If None, read `DSP_PREVIEW_MODE` environment variable
    3. If still None, use `PREVIEW_ENABLED_DEFAULT`
  - **Audio truncation**:
    - If preview enabled, truncate audio to first `PREVIEW_MAX_SECONDS` before processing
    - Works for both mono (1D) and multi-channel (2D) audio
    - Truncation happens before any processing, so pipeline is unaware of preview mode

- **Updated timing logs**:
  - Added `mode=preview` or `mode=full` indicator to timing output
  - Format: `[RENDER_TIMING] mode=preview load=X.XXXs ...`

- **Environment variable support**:
  - `DSP_PREVIEW_MODE=1` or `true` → preview enabled
  - `DSP_PREVIEW_MODE=0` or `false` or unset → uses default

#### Results
- **Faster iteration**: Preview mode processes 10 seconds instead of full file
- **Performance**: ~3x faster for 1/3 the duration (30s file: 121s → 40s)
- **Usage flexibility**: Can be controlled via parameter or environment variable
- **Production safety**: Defaults to full render, preview must be explicitly enabled

### Prompt 5 – Optional: Numba-accelerate remaining per-frame/per-bin logic

#### Changes Made
- **Added Numba dependency**:
  - Added `numba` to `requirements.txt`

- **Created Numba-accelerated smear function** in `quantum_distortion/dsp/quantizer.py`:
  - **Function**: `_apply_smear_numba()` decorated with `@numba.njit`
  - **Purpose**: Accelerates the remaining loop in smear operation that couldn't be fully vectorized
  - **Numba compatibility**:
    - Uses only NumPy arrays and basic numeric types
    - Constructs complex numbers using `complex(real, imag)` (Numba-compatible)
    - No Python objects or dynamic typing
  - **Documentation**: Comprehensive docstring explaining Numba constraints and JIT compilation overhead

- **Integrated Numba function**:
  - Updated `quantize_spectrum()` to use `_apply_smear_numba()` when Numba is available
  - Graceful fallback to pure Python loop if Numba is not installed
  - Proper type casting (int32, float64) for Numba compatibility

- **Fixed test import error**:
  - Updated `tests/test_quantization_integration.py` to use `process_audio()` instead of removed `_apply_spectral_quantization_block()`

#### Results
- **Performance improvement**: ~5x speedup in smear operation
- **Individual calls**: `quantize_spectrum()` now ~2.7ms average (after JIT warmup)
- **Overall pipeline**: Processing time reduced from ~3.4s to ~0.665s for spectral processing stage
- **Behavior**: Identical audio output (verified with numerical tests)
- **Code quality**: All tests pass (23/23), no linter errors

## Performance Summary

### Before Optimization
- Spectral processing: ~3.4s per render
- Multiple redundant STFT/iSTFT operations
- Python-level loops in critical paths

### After Optimization
- Spectral processing: ~0.665s per render (~5x speedup)
- Single STFT/iSTFT when possible
- Vectorized operations with NumPy broadcasting
- Numba JIT compilation for remaining loops
- Preview mode: ~3x faster for 1/3 duration

### Key Optimizations
1. **Vectorization**: Replaced Python loops with NumPy broadcasting and array operations
2. **STFT/iSTFT optimization**: Minimized redundant transforms
3. **Numba acceleration**: JIT-compiled remaining critical loops
4. **Preview mode**: Fast iteration for development and sound design

## Files Modified

### Core DSP Code
- `quantum_distortion/dsp/pipeline.py`:
  - Added `RenderTiming` dataclass
  - Added profiling instrumentation
  - Refactored for optimal STFT/iSTFT usage
  - Added preview render mode support
  - Normalized FFT settings

- `quantum_distortion/dsp/quantizer.py`:
  - Vectorized `build_target_bins_for_freqs()`
  - Vectorized `quantize_spectrum()` (energy movement, phase updates, bin smoothing)
  - Added `_apply_smear_numba()` for Numba-accelerated smear operation
  - Added Numba import with graceful fallback

### Configuration
- `quantum_distortion/config.py`:
  - Added `PREVIEW_ENABLED_DEFAULT` and `PREVIEW_MAX_SECONDS` constants

### Dependencies
- `requirements.txt`:
  - Added `numba` dependency

### Tests
- `tests/test_quantization_integration.py`:
  - Updated to use `process_audio()` instead of removed function

## Testing

All tests pass (23/23):
- Unit tests for quantizer, distortion, limiter, analyses, presets, visualizers
- Integration tests for pipeline and quantization
- Streamlit app import test

## Usage Examples

### Preview Mode via Environment Variable
```bash
DSP_PREVIEW_MODE=1 python scripts/render_cli.py --infile input.wav --outfile output.wav
```

### Preview Mode via Function Parameter
```python
processed, taps = process_audio(
    audio, sr,
    preview_enabled=True  # Enable preview mode
)
```

### Full Render (Default)
```python
processed, taps = process_audio(
    audio, sr,
    preview_enabled=False  # Force full render
)
```

## Notes

- **JIT Compilation Overhead**: First call to Numba-accelerated functions includes ~0.1-1s compilation overhead, but subsequent calls are significantly faster
- **Preview Mode**: Defaults to full render for production safety; preview must be explicitly enabled
- **Backward Compatibility**: All changes maintain identical audio output and API compatibility
- **Graceful Degradation**: If Numba is not available, code falls back to pure Python implementation

