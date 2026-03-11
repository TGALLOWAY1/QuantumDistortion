# Milestone M1: Quantizer Core

## Summary

This milestone implements the core spectral pitch quantization functionality for the Quantum Distortion project. The quantizer module provides utilities for converting frequencies to musical scales, building target bins for frequency quantization, and applying spectral quantization to FFT frames. The pipeline has been updated to expose quantizer parameters in preparation for full integration in future milestones.

## Prompts Executed

### PROMPT 1.1 — Create quantizer.py with scale + pitch utilities
- Created `quantum_distortion/dsp/quantizer.py` with comprehensive pitch quantization functionality
- Implemented scale definitions:
  - Major, minor, pentatonic, dorian, mixolydian, harmonic_minor scales
  - Scale intervals defined as pitch class offsets from root
- Implemented harmonic weight system:
  - Root (1.0), Fifth (0.8), Third (0.7), Seventh (0.6), Other (0.5)
  - Used to prioritize certain scale degrees during quantization
- Core functions:
  - `note_name_to_pitch_class()` - Converts note names (C, C#, Db, etc.) to pitch class 0-11
  - `midi_to_freq()` / `freq_to_midi()` - Frequency/MIDI conversion utilities
  - `build_scale_notes()` - Generates scale notes across octaves for a given frequency range
  - `build_target_bins_for_freqs()` - Maps frequency bins to nearest in-scale target bins
  - `quantize_spectrum()` - Applies spectral quantization to FFT frames with configurable snap strength, smear, and bin smoothing
- Added Python 3.7 compatibility by using `typing_extensions` for `Literal` type
- All functions use `.copy()` to avoid mutating input arrays

### PROMPT 1.2 — Add unit tests for quantizer behavior
- Created `tests/test_quantizer.py` with comprehensive unit tests
- Test coverage:
  - `test_build_target_bins_simple_monotonic()` - Validates target bin mapping for frequencies near scale notes
  - `test_quantize_moves_energy_to_target_bin()` - Verifies energy moves from off-scale bins to nearest in-scale bin with snap_strength=1.0
  - `test_quantize_smear_distributes_energy()` - Validates that smearing redistributes energy around target bins
  - `test_bin_smoothing_changes_shape_not_energy()` - Ensures bin smoothing changes shape while preserving total energy
- All tests use small synthetic spectra (not real FFTs) for fast, deterministic testing
- Tests validate core quantization behavior: energy movement, smear distribution, and energy conservation

### PROMPT 1.3 — Wire quantizer into pipeline stub
- Updated `process_audio()` in `quantum_distortion/dsp/pipeline.py`:
  - Added comprehensive docstring explaining Milestone 1 status
  - Added mono audio validation (raises ValueError for non-1D arrays)
  - Added `distortion_params` initialization if None
  - All quantizer parameters already exposed in function signature (key, scale, snap_strength, smear, bin_smoothing, pre_quant, post_quant)
  - Still operates as pass-through (quantization not yet applied, preparing for Milestone 2/3)
- Fixed Python 3.7 compatibility in `quantum_distortion/io/audio_io.py`:
  - Replaced `|` union syntax with `Union` from typing module
  - Updated `load_audio()` and `save_audio()` function signatures
- Maintained backward compatibility with CLI script and existing tests

### PROMPT 1.4 — Quick regression run
- Verified all tests pass:
  - `tests/test_imports_and_io.py::test_roundtrip` - PASSED
  - All 4 tests in `tests/test_quantizer.py` - PASSED
- Confirmed no import errors from `quantum_distortion.dsp.quantizer`
- Total: 5 tests passed in 0.47 seconds

## Files Created/Modified

### New Files
- `quantum_distortion/dsp/quantizer.py` - Core quantizer module with scale/pitch utilities and spectral quantization
- `tests/test_quantizer.py` - Comprehensive unit tests for quantizer functionality

### Modified Files
- `quantum_distortion/dsp/pipeline.py` - Updated `process_audio()` with quantizer parameter exposure, mono validation, and docstring
- `quantum_distortion/io/audio_io.py` - Fixed Python 3.7 compatibility (replaced `|` union syntax with `Union`)

## Key Features Implemented

### Scale and Pitch Utilities
- Support for 6 musical scales: major, minor, pentatonic, dorian, mixolydian, harmonic_minor
- Note name parsing with sharp/flat support (normalized to sharps)
- MIDI to frequency conversion and vice versa
- Scale note generation across multiple octaves for any frequency range

### Spectral Quantization
- Target bin computation based on weighted distance to in-scale notes
- Configurable snap strength (0.0 = no movement, 1.0 = full attraction)
- Energy smearing with Gaussian-like distribution around target bins
- Optional bin smoothing with moving-average kernel
- Energy conservation: total magnitude preserved during quantization

### Harmonic Weighting
- Root notes have highest weight (1.0)
- Fifths, thirds, sevenths have progressively lower weights
- Other scale degrees have lowest weight (0.5)
- Weights influence which scale notes attract off-scale frequencies

## Technical Details

### Python Compatibility
- Added Python 3.7 support by using `typing_extensions` for `Literal` type
- Replaced Python 3.10+ union syntax (`|`) with `Union` from typing module
- All code tested on Python 3.7.4

### Code Quality
- All functions use `.copy()` to avoid mutating input arrays
- Comprehensive type hints throughout
- Docstrings for all public functions
- No circular imports (quantizer only imports standard library and numpy)

### Testing
- Fast unit tests using synthetic spectra (no external file dependencies)
- Tests validate energy conservation, shape changes, and quantization behavior
- All tests pass consistently

## Verification

- ✅ All tests pass (5/5)
- ✅ No import errors
- ✅ Backward compatible with CLI script
- ✅ No linting errors
- ✅ Python 3.7 compatible

## Next Steps

The quantizer core is complete and ready for:
1. Integration with STFT processing in Milestone 2
2. Application of quantization in the audio pipeline
3. Real-time processing optimizations
4. UI visualization of quantization effects

