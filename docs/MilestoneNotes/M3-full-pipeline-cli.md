# Milestone M3: Full Pipeline and CLI Integration

## Summary

This milestone completes the full integration of all DSP components into a working audio processing pipeline. The pipeline now performs real-time spectral quantization, distortion, and limiting, with full STFT processing. The CLI tool has been updated to reflect real processing, and comprehensive integration tests validate the end-to-end functionality.

## Prompts Executed

### PROMPT 3.1 — Create STFT helpers
- Created `quantum_distortion/dsp/stft_utils.py` with STFT utility functions
- **`stft_mono()`** - Computes complex STFT for mono signals:
  - Returns complex STFT matrix `S` (shape: `n_fft//2 + 1, n_frames`)
  - Returns frequency array `freqs` (shape: `n_fft//2 + 1`)
  - Configurable: `n_fft`, `hop_length`, `window`, `center`
  - Default `hop_length = n_fft // 4`
  - Validates mono (1D) audio input
- **`istft_mono()`** - Inverse STFT for mono signals:
  - Reconstructs audio from complex STFT matrix
  - Returns float32 numpy array
  - Configurable: `n_fft`, `hop_length`, `window`, `length`, `center`
  - Default `hop_length = n_fft // 4`
- Uses librosa for STFT/iSTFT operations
- Python 3.7 compatible (uses `Union` instead of `|` union syntax)
- No circular imports (only imports numpy and librosa)

### PROMPT 3.2 — Wire quantizer + distortion + limiter into pipeline
- Completely rewrote `quantum_distortion/dsp/pipeline.py` with full implementation
- **`_ensure_mono()`** - Handles mono/stereo conversion:
  - Accepts 1D (mono) or 2D (stereo/multi-channel) audio
  - Downmixes multi-channel by averaging channels
  - Returns float32 mono audio
  - No longer raises on 2D audio (downmixes instead)
- **`_apply_spectral_quantization_block()`** - Helper for spectral quantization:
  - Computes STFT
  - Applies `quantize_spectrum()` to each frame independently
  - Reconstructs audio via iSTFT
  - Returns quantized audio
- **`process_audio()`** - Full processing pipeline:
  - **Stage A: Pre-Quantization** (optional) - Spectral quantization before distortion
  - **Stage B: Time-Domain Distortion** - Applies wavefold or soft-tube distortion
  - **Stage C: Post-Quantization** (optional) - Spectral quantization after distortion
  - **Stage D: Limiter** (optional) - Lookahead peak limiting
  - **Dry/Wet Mix** - Mixes processed signal with original input
  - Returns processed audio and tap buffers
- Pipeline flow: `input → (optional) pre-quant → distortion → (optional) post-quant → (optional) limiter → dry/wet mix → output`
- All tap buffers present: `input`, `pre_quant`, `post_dist`, `output`
- Output length matches input length
- Python 3.7 compatible

### PROMPT 3.3 — Keep CLI stable (optional docstring tweak)
- Updated `scripts/render_cli.py` with improved documentation
- Added docstring to `main()` function:
  - Describes that script loads audio, runs it through full DSP pipeline with default parameters
  - Notes that pre/post quantization, distortion, and limiter are applied
- Updated argument parser description:
  - Changed from "Quantum Distortion - Offline Renderer (M0)" to "Quantum Distortion - Offline Renderer"
- Updated print statement:
  - Changed "Saved pass-through output" to "Saved processed output" to reflect real processing
- No functional changes - still uses defaults only

### PROMPT 3.4 — Add pipeline integration tests
- Created `tests/test_pipeline.py` with comprehensive integration tests
- **Test coverage**:
  - `test_pipeline_runs_and_taps_shapes()` - Validates end-to-end execution and tap buffer consistency
    - Verifies all required tap buffers exist: `input`, `pre_quant`, `post_dist`, `output`
    - Asserts all tap buffers have same shape as input
    - Asserts output shape matches input shape
    - Uses minimal processing to keep runtime fast (<0.5s)
  - `test_pipeline_neutral_settings_approx_passthrough()` - Validates neutral settings produce output close to input
    - Disables pre/post quantization
    - Uses minimal distortion (fold_amount=1.0, bias=0.0)
    - Disables limiter
    - Uses dry_wet=1.0 (all wet)
    - Asserts output is close to input (within 1e-3 tolerance)
- Tests run fast: 0.49 seconds (< 0.5s requirement)
- Python 3.7 compatible

### PROMPT 3.5 — Full regression
- Verified all tests pass across entire test suite
- **Test results**: 13 tests passed in 8.35 seconds
  - `tests/test_imports_and_io.py` - 1 test passed
  - `tests/test_quantizer.py` - 4 tests passed
  - `tests/test_distortion.py` - 3 tests passed
  - `tests/test_limiter.py` - 3 tests passed
  - `tests/test_pipeline.py` - 2 tests passed
- CLI verification:
  - Created test audio file: `examples/example_bass.wav` (1 second, 60 Hz + 120 Hz bass tones)
  - CLI processing successful: `examples/example_bass_qd.wav` created
  - Processing confirmed: Output differs from input (RMS difference: 0.069031)
  - All tap buffers present and functional

## Files Created/Modified

### New Files
- `quantum_distortion/dsp/stft_utils.py` - STFT utility functions for mono audio
- `tests/test_pipeline.py` - Comprehensive integration tests for pipeline

### Modified Files
- `quantum_distortion/dsp/pipeline.py` - Complete rewrite with full processing pipeline
- `scripts/render_cli.py` - Updated docstring and print statements to reflect real processing

## Key Features Implemented

### STFT Processing
- **STFT/iSTFT utilities**: Wrapper functions around librosa for mono audio processing
  - Configurable FFT size, hop length, window type
  - Default: 2048-point FFT, 512-sample hop, Hann window
  - Proper frequency bin calculation
  - Length preservation with iSTFT

### Full Processing Pipeline
- **Multi-stage processing**: Complete integration of all DSP components
  - Optional pre-quantization (spectral quantization before distortion)
  - Time-domain distortion (wavefold or soft-tube)
  - Optional post-quantization (spectral quantization after distortion)
  - Optional lookahead limiter
  - Dry/wet mixing with original input
- **Stereo/mono handling**: Automatic downmixing of multi-channel audio
- **Tap buffers**: All processing stages available for visualization
- **Length preservation**: Output always matches input length

### CLI Integration
- **Real processing**: CLI now performs actual DSP processing (not pass-through)
- **Default parameters**: Uses sensible defaults for all processing stages
- **Clear documentation**: Updated descriptions reflect actual functionality

## Technical Details

### Python Compatibility
- Python 3.7 compatible (uses `Union` instead of `|` union syntax)
- All functions use `.copy()` or create new arrays to avoid mutating inputs
- Comprehensive type hints throughout

### Code Quality
- All functions validate mono (1D) audio input (after downmixing)
- All functions return float32 numpy arrays
- No side effects or in-place mutation
- Comprehensive docstrings for all public functions
- No circular imports

### Testing
- Fast integration tests using synthetic signals
- Tests validate end-to-end pipeline execution
- Tests validate tap buffer consistency
- Tests validate neutral settings behavior
- All tests pass consistently
- Total test suite: 13 tests in ~8 seconds

### Performance
- STFT processing uses efficient librosa implementation
- Frame-by-frame quantization for memory efficiency
- Configurable processing stages (can disable quantization/limiter)
- Pipeline test runs in <0.5s with minimal processing

## Verification

- ✅ All tests pass (13/13)
- ✅ No import errors
- ✅ Backward compatible with existing tests
- ✅ CLI successfully processes audio files
- ✅ Output audibly differs from input (processing confirmed)
- ✅ No linting errors
- ✅ Python 3.7 compatible

## Pipeline Behavior

### Default Settings (Full Processing)
- Pre-quantization: Enabled (snap_strength=0.8)
- Distortion: Wavefold mode (fold_amount=1.0)
- Post-quantization: Enabled (snap_strength=0.8)
- Limiter: Enabled (ceiling=-1.0 dB)
- Dry/wet: 1.0 (fully processed)

### Minimal Processing Mode
- Pre-quantization: Disabled
- Distortion: Minimal (fold_amount=1.0, bias=0.0)
- Post-quantization: Disabled
- Limiter: Disabled
- Result: Output very close to input (within 1e-3 tolerance)

## Next Steps

The full pipeline is complete and ready for:
1. UI development for real-time visualization
2. Parameter automation and modulation
3. Additional distortion modes
4. Advanced quantization algorithms
5. Real-time processing optimizations
6. VST plugin development (JUCE port)

