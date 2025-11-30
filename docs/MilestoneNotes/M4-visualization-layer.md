# Milestone M4: Visualization Layer

## Summary

This milestone implements a comprehensive visualization layer for the Quantum Distortion project. The visualizers module provides reusable plotting utilities for spectrum analysis, oscilloscope views, and phase scope visualization. A development script enables quick preview of visualizations for all pipeline tap points, facilitating development and debugging.

## Prompts Executed

### PROMPT 4.1 — Create visualizers.py
- Created `quantum_distortion/ui/visualizers.py` with reusable plotting utilities
- **`_select_segment()`** - Utility for selecting audio segments:
  - Extracts short segments from audio for visualization
  - Supports centered or start-based selection
  - Handles edge cases (short audio, empty audio)
- **`plot_spectrum()`** - Magnitude spectrum plot:
  - Computes FFT with Hanning window
  - Displays magnitude in dB vs frequency (Hz)
  - Optional scale lines showing in-key frequencies
  - Configurable max frequency limit
  - Returns matplotlib Figure (no display)
- **`plot_oscilloscope()`** - Time-domain oscilloscope view:
  - Plots short audio segment in time domain
  - Time axis in milliseconds
  - Configurable duration
  - Returns matplotlib Figure (no display)
- **`plot_phase_scope()`** - Phase scope / Lissajous figure:
  - For mono: creates pseudo-stereo with 1ms delay
  - For stereo: uses L vs R channels directly
  - Plots L vs R scatter plot
  - Returns matplotlib Figure (no display)
- All functions return matplotlib.figure.Figure objects
- No direct calls to plt.show() - caller controls rendering
- Python 3.7 compatible (uses typing_extensions for Literal)

### PROMPT 4.2 — Visualizer unit tests
- Created `tests/test_visualizers.py` with comprehensive unit tests
- **Test coverage**:
  - `test_plot_spectrum_returns_figure()` - Validates spectrum plot returns Figure
    - Tests with scale lines enabled
    - Asserts return type is matplotlib.figure.Figure
  - `test_plot_oscilloscope_returns_figure()` - Validates oscilloscope plot returns Figure
    - Tests with custom duration
    - Asserts return type is matplotlib.figure.Figure
  - `test_plot_phase_scope_mono_returns_figure()` - Validates phase scope with mono audio
    - Tests pseudo-stereo generation from mono
    - Asserts return type is matplotlib.figure.Figure
  - `test_plot_phase_scope_stereo_returns_figure()` - Validates phase scope with stereo audio
    - Tests with actual stereo input (L sine, R cosine)
    - Asserts return type is matplotlib.figure.Figure
- Tests run fast: 1.18-1.19 seconds total
- No windows displayed: Functions return Figure objects without calling plt.show()
- Structural validation: Tests verify return types are matplotlib.figure.Figure

### PROMPT 4.3 — Dev script to preview visualizations
- Created `scripts/preview_visualizers.py` - Development script for previewing visualizations
- **Script functionality**:
  - Loads audio file using load_audio()
  - Processes audio through full pipeline with process_audio()
  - Generates visualizations for each tap buffer
  - Saves PNG files to specified output directory
- **Visualization generation**:
  - Spectrum plots: Magnitude spectrum with scale lines (C minor)
  - Oscilloscope plots: Time-domain view (20ms segment)
  - Phase scope plots: Lissajous/phase scope visualization
  - For each tap: input, pre_quant, post_dist, output
- **File management**:
  - Creates output directory if it doesn't exist
  - Saves PNGs with descriptive filenames: {tap_name}_{viz_type}.png
  - Closes figures after saving to free memory
  - Uses 120 DPI for good quality
- Generates 12 PNG files (4 taps × 3 visualization types)

### PROMPT 4.4 — Full regression
- Verified all tests pass across entire test suite
- **Test results**: 17 tests passed in 5.04 seconds
  - `tests/test_imports_and_io.py` - 1 test passed
  - `tests/test_quantizer.py` - 4 tests passed
  - `tests/test_distortion.py` - 3 tests passed
  - `tests/test_limiter.py` - 3 tests passed
  - `tests/test_pipeline.py` - 2 tests passed
  - `tests/test_visualizers.py` - 4 tests passed
- No regressions introduced
- Visualizers module integrated successfully
- All components work together

## Files Created/Modified

### New Files
- `quantum_distortion/ui/visualizers.py` - Reusable plotting utilities for audio visualization
- `tests/test_visualizers.py` - Comprehensive unit tests for visualizers
- `scripts/preview_visualizers.py` - Development script for previewing visualizations

### Modified Files
- None (no changes to existing files)

## Key Features Implemented

### Visualization Utilities
- **Spectrum analysis**: FFT-based magnitude spectrum with optional scale lines
  - Configurable frequency range
  - dB magnitude scale
  - Optional in-key frequency markers
- **Oscilloscope view**: Time-domain waveform visualization
  - Configurable time window
  - Millisecond time axis
  - Clear amplitude display
- **Phase scope**: Lissajous/phase visualization
  - Mono support with pseudo-stereo generation
  - Stereo support with L vs R channels
  - Equal aspect ratio for proper phase display

### Development Tools
- **Preview script**: Quick visualization generation for all pipeline stages
  - Batch processing of all tap buffers
  - PNG export for easy sharing
  - Configurable output directory

## Technical Details

### Python Compatibility
- Python 3.7 compatible (uses typing_extensions for Literal type)
- All functions use matplotlib non-interactive backend for testing
- Comprehensive type hints throughout

### Code Quality
- All functions return matplotlib.figure.Figure objects
- No direct calls to plt.show() - caller controls rendering
- Comprehensive docstrings for all public functions
- No circular imports
- Proper error handling for edge cases

### Testing
- Fast unit tests using synthetic signals
- Tests validate return types (matplotlib.figure.Figure)
- Tests validate function execution without errors
- No visual assertions (structural validation only)
- All tests pass consistently
- Total test suite: 17 tests in ~5 seconds

### Visualization Quality
- High-quality PNG export (120 DPI)
- Proper axis labels and titles
- Clear tap source identification
- Professional appearance suitable for documentation

## Verification

- ✅ All tests pass (17/17)
- ✅ No import errors
- ✅ No regressions introduced
- ✅ Visualizers module integrated successfully
- ✅ Preview script generates all expected PNG files
- ✅ No linting errors
- ✅ Python 3.7 compatible

## Usage Examples

### Programmatic Usage
```python
from quantum_distortion.ui.visualizers import plot_spectrum, plot_oscilloscope, plot_phase_scope

# Generate spectrum plot
fig = plot_spectrum(audio, sr, 'input', key='A', scale='minor', show_scale_lines=True)
fig.savefig('spectrum.png')
plt.close(fig)
```

### Development Script
```bash
python scripts/preview_visualizers.py \
  --infile examples/example_bass.wav \
  --outdir examples/visualizations
```

## Next Steps

The visualization layer is complete and ready for:
1. Streamlit UI integration for interactive visualization
2. Real-time visualization updates during processing
3. Additional visualization types (spectrogram, waterfall, etc.)
4. Parameter automation visualization
5. Comparison views (before/after processing)
6. Export to various formats (SVG, PDF, etc.)

