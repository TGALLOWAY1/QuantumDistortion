# Milestone M2: Distortion and Limiter

## Summary

This milestone implements the distortion and limiter modules for the Quantum Distortion project. The distortion module provides two algorithms (wavefold and soft-tube) for generating harmonic content, while the limiter module provides a lookahead peak limiter for preventing clipping and controlling dynamics. Both modules are fully tested and ready for integration into the audio processing pipeline.

## Prompts Executed

### PROMPT 2.1 — Create distortion.py with wavefold + soft-tube
- Created `quantum_distortion/dsp/distortion.py` with two distortion algorithms
- **Wavefold distortion** (`wavefold()`):
  - Applies bias (DC offset) and fold_amount (input gain)
  - Uses mirrored clipping around ±threshold for single-fold pass
  - Generates rich harmonics through wavefolding
  - Returns float32 output clipped to threshold range
- **Soft-tube saturation** (`soft_tube()`):
  - Uses tanh waveshaper for tube-style saturation
  - Configurable drive (input gain) and warmth (curve steepness)
  - Maps warmth (0..1) to shape parameter (1..5)
  - Normalized output to keep in [-1, 1] range
- **Convenience wrapper** (`apply_distortion()`):
  - Routes to appropriate function based on mode ("wavefold" or "tube")
  - Handles parameter routing for each mode
- Added Python 3.7 compatibility using `typing_extensions` for `Literal` type
- All functions validate mono (1D) audio and return float32 arrays
- No in-place mutation of input arrays

### PROMPT 2.2 — Unit tests for distortion
- Created `tests/test_distortion.py` with comprehensive unit tests
- **Test coverage**:
  - `test_wavefold_generates_harmonics()` - Validates that wavefold generates harmonics by comparing harmonic energy (total - fundamental) between clean and folded signals
  - `test_soft_tube_drive_increases_thd()` - Verifies that increasing drive increases total harmonic distortion (THD)
  - `test_apply_distortion_routes_modes()` - Ensures `apply_distortion` correctly routes parameters to wavefold and tube modes
- All tests use synthetic sine waves for fast, deterministic testing
- Tests validate harmonic generation, distortion intensity scaling, and parameter routing

### PROMPT 2.3 — Create limiter.py (lookahead peak limiter)
- Created `quantum_distortion/dsp/limiter.py` with lookahead peak limiter
- **Core functionality**:
  - `db_to_linear()` - Converts dB to linear gain
  - `peak_limiter()` - Main limiter function with configurable parameters
- **Limiter features**:
  - **Lookahead**: Anticipates peaks by examining upcoming samples in a configurable window
  - **Attack**: Instantly reduces gain when peaks exceed ceiling
  - **Release**: Smoothly returns gain toward 1.0 over time using exponential smoothing
  - Configurable ceiling in dBFS (default -1.0 dB)
  - Configurable lookahead window in milliseconds (default 5.0 ms)
  - Configurable release time in milliseconds (default 50.0 ms)
- Returns both limited audio and gain curve for visualization
- Handles edge cases (empty arrays, validates mono audio)
- No external dependencies beyond numpy

### PROMPT 2.4 — Unit tests for limiter
- Created `tests/test_limiter.py` with comprehensive unit tests
- **Test coverage**:
  - `test_limiter_reduces_peaks()` - Verifies that peaks exceeding ceiling are reduced to at or below ceiling (with 1% tolerance)
  - `test_limiter_minimal_effect_below_threshold()` - Ensures signals below threshold are mostly unaffected (gain curve stays near 1.0)
  - `test_limiter_gain_recovers_after_spike()` - Validates that gain curve recovers toward 1.0 after transient spikes
- All tests use synthetic signals (sine waves, spikes) for deterministic testing
- Tests validate peak reduction, minimal effect on low-level signals, and gain recovery behavior

### PROMPT 2.5 — Regression: run full test suite
- Verified all tests pass across the entire test suite
- **Test results**: 11 tests passed in 0.76 seconds
  - `tests/test_imports_and_io.py` - 1 test passed
  - `tests/test_quantizer.py` - 4 tests passed
  - `tests/test_distortion.py` - 3 tests passed
  - `tests/test_limiter.py` - 3 tests passed
- Confirmed no breaking changes to existing functionality
- Verified backward compatibility with CLI script and pipeline
- All imports successful, no circular dependencies

## Files Created/Modified

### New Files
- `quantum_distortion/dsp/distortion.py` - Distortion module with wavefold and soft-tube algorithms
- `quantum_distortion/dsp/limiter.py` - Lookahead peak limiter module
- `tests/test_distortion.py` - Comprehensive unit tests for distortion module
- `tests/test_limiter.py` - Comprehensive unit tests for limiter module

### Modified Files
- None (no changes to existing files required)

## Key Features Implemented

### Distortion Algorithms
- **Wavefold**: Mirrored clipping algorithm that generates rich harmonics
  - Configurable fold_amount (input gain)
  - Configurable bias (DC offset)
  - Configurable threshold (folding point)
  - Single-fold pass for MVP simplicity
- **Soft-tube**: Tanh-based saturation for tube-style warmth
  - Configurable drive (input gain, >1.0 increases distortion)
  - Configurable warmth (0..1, controls curve steepness)
  - Normalized output to maintain [-1, 1] range

### Limiter
- **Lookahead peak limiting**: Prevents clipping by anticipating peaks
  - Configurable ceiling in dBFS
  - Configurable lookahead window (milliseconds)
  - Configurable release time (milliseconds)
  - Exponential release smoothing
  - Returns gain curve for visualization

## Technical Details

### Python Compatibility
- Python 3.7 compatible (uses `typing_extensions` for `Literal` type)
- All functions use `.copy()` or create new arrays to avoid mutating inputs
- Comprehensive type hints throughout

### Code Quality
- All functions validate mono (1D) audio input
- All functions return float32 numpy arrays
- No side effects or in-place mutation
- Comprehensive docstrings for all public functions
- No circular imports (only standard library and numpy)

### Testing
- Fast unit tests using synthetic signals (no external file dependencies)
- Tests validate harmonic generation, distortion intensity, parameter routing
- Tests validate peak reduction, minimal effect on low signals, gain recovery
- All tests pass consistently (no flaky behavior)
- Total test suite runs in <1 second

## Verification

- ✅ All tests pass (11/11)
- ✅ No import errors
- ✅ Backward compatible with existing pipeline and CLI
- ✅ No linting errors
- ✅ Python 3.7 compatible
- ✅ No breaking changes to existing functionality

## Next Steps

The distortion and limiter modules are complete and ready for:
1. Integration into the audio processing pipeline (Milestone 3)
2. Application of distortion in pre/post-quantization stages
3. Limiter application at the end of the processing chain
4. UI visualization of distortion and limiting effects
5. Real-time processing optimizations

