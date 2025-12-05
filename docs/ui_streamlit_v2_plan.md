# Quantum Distortion Streamlit UI - V2 Plan

## Current UI Structure (MVP)

The current Streamlit interface (`quantum_distortion/ui/app_streamlit.py`) is a single-page MVP with a slider-based control layout. Key characteristics:

### Entry Point
- **File**: `quantum_distortion/ui/app_streamlit.py`
- **Run command**: `streamlit run quantum_distortion/ui/app_streamlit.py`
- **Main function**: `main()` - single-page Streamlit app

### Audio Loading
- Uses `st.file_uploader` widget to accept WAV/AIF/AIFF files
- Automatically converts stereo to mono by averaging channels
- Stores loaded audio in `st.session_state["audio"]` and sample rate in `st.session_state["sr"]`
- Tracks loaded file name to prevent unnecessary reloads on reruns

### Processing Pipeline
- **Import**: `process_audio` is imported from `quantum_distortion.dsp.pipeline`
- **Trigger**: Render button ("Render / Process Audio") at line 196
- **Function call**: `process_audio()` is called with parameters collected from UI sliders/controls
- **Returns**: Tuple of `(processed_audio, taps_dict)` where taps_dict contains intermediate stages
- **Storage**: Results stored in `st.session_state["processed"]` and `st.session_state["taps"]`

### Current Layout (5 Panels)
1. **File & Transport Panel**
   - File uploader widget
   - Original and processed audio playback
   - Download button for processed WAV

2. **Quantization Settings Panel**
   - Key selection (12 keys)
   - Scale selection (6 scales)
   - Snap strength slider (0-100%)
   - Smear slider (0-100%)
   - Bin smoothing checkbox
   - Pre/Post quantization toggles

3. **Distortion Panel**
   - Mode selection (Wavefold or Soft Tube)
   - Mode-specific parameters (fold amount, bias, drive, warmth)

4. **Output Panel**
   - Limiter toggle and ceiling slider
   - Dry/wet mix slider (0-100%)
   - Output gain slider (-12 to +12 dB)

5. **Analysis & Visualization Panel**
   - Tap source selection (input, pre_quant, post_dist, output)
   - Visualization type (Spectrum, Oscilloscope, Phase Scope)
   - Spectrum-specific options (show scale lines, max frequency)

### Pages/Tabs
- **Current**: Single-page layout, no tabs or multi-page navigation
- All controls and visualizations are on one scrollable page

### Session State Management
- Tracks: `audio`, `sr`, `processed`, `taps`, `loaded_file_name`
- Prevents state loss on reruns
- Analysis panel uses local copy of taps to avoid mutations

---

## Planned V2 UI

The V2 UI will maintain the single-page layout but reorganize controls into a more intuitive signal-flow-based structure:

### One-Page Layout
- Single scrollable page (no tabs)
- Organized left-to-right and top-to-bottom following signal flow
- Improved visual hierarchy and grouping

### Signal Flow Overview
- Visual diagram or text-based flow indicator at the top
- Shows: Input → Low Band / High Band → Creative Quantum FX → Output
- Helps users understand the processing chain

### Low Band Panel
- Dedicated section for low-frequency band processing
- Controls for:
  - Crossover frequency
  - Low band drive/saturation
  - Low band-specific quantization settings
  - Low band visualization

### High Band Panel
- Dedicated section for high-frequency band processing
- Controls for:
  - High band drive/saturation
  - High band-specific quantization settings
  - High band visualization

### Creative Quantum FX Panel
- Central section for spectral effects
- Controls for:
  - Quantum FX mode selection
  - FX strength/intensity
  - Spectral manipulation parameters
  - Real-time preview options

### Analysis (Delta, Spectrograms, etc.)
- Enhanced analysis section with:
  - Delta visualization (before/after comparison)
  - Spectrogram views (time-frequency representation)
  - Multi-tap comparison views
  - Export options for analysis data
  - Real-time parameter impact visualization

---

## Migration Notes

- Current MVP uses single-band processing (no multiband split)
- V2 will introduce multiband processing with separate low/high band controls
- Need to maintain backward compatibility with existing presets
- Session state structure may need expansion for multiband taps
- Visualization functions may need updates to handle band-specific displays

