"""
Streamlit entrypoint for Quantum Distortion UI.

This module provides the MVP slider-based interface for the Quantum Distortion
DSP prototype. It allows users to upload audio files, configure quantization
and distortion parameters, process audio through the pipeline, and visualize
results at various tap points.

Key Components:
- Audio loading via file uploader widget (WAV/AIF/AIFF)
- Parameter controls for quantization, distortion, and output settings
- Render button triggers process_audio() from quantum_distortion.dsp.pipeline
- Analysis panel with spectrum, oscilloscope, and phase scope visualizations
- Session state management to preserve render results across UI interactions

The app uses a single-page layout with 5 main panels:
1. File & Transport - file upload and audio playback
2. Quantization Settings - key, scale, snap strength, smear, etc.
3. Distortion - wavefold or soft tube mode with respective parameters
4. Output - limiter, dry/wet mix, output gain
5. Analysis & Visualization - tap point selection and visualization types
"""

from __future__ import annotations


from io import BytesIO
from pathlib import Path
from typing import Any, Dict


import numpy as np
import soundfile as sf
import streamlit as st


from quantum_distortion.dsp.pipeline import process_audio
from quantum_distortion.ui.visualizers import (
    plot_spectrum,
    plot_oscilloscope,
    plot_phase_scope,
)


KEY_OPTIONS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALE_OPTIONS = ["major", "minor", "pentatonic", "dorian", "mixolydian", "harmonic_minor"]
TAP_OPTIONS = ["input", "pre_quant", "post_dist", "output"]
VISUAL_OPTIONS = ["Spectrum", "Oscilloscope", "Phase Scope"]


def build_processing_config_from_session() -> Dict[str, Any]:
    """
    Build a centralized processing configuration dict from Streamlit session state.
    
    Collects all UI settings (low band, high band, quantum FX, etc.) into a single
    nested config structure that can be passed to the DSP pipeline.
    
    Returns
    -------
    dict
        Nested configuration dict with keys:
        - crossover_freq: float
        - quantization: dict with key, scale settings
        - low_band: dict with saturation, mono, output_trim settings
        - high_band: dict with STFT, spectral FX settings
        - quantum_fx: dict with freeze, formant, harmonic locking settings
        - delta_listen: bool
    """
    import streamlit as st
    
    config: Dict[str, Any] = {}
    
    # Quantization settings (key and scale)
    quant_settings = st.session_state.get("quantization_settings", {})
    config["quantization"] = {
        "key": quant_settings.get("key", "D"),
        "scale": quant_settings.get("scale", "minor"),
    }
    
    # Crossover frequency (shared between low and high band)
    low_band_settings = st.session_state.get("low_band_settings", {})
    config["crossover_freq"] = low_band_settings.get("crossover_freq", 300)
    
    # Low band settings
    config["low_band"] = {
        "saturation_amount": low_band_settings.get("saturation_amount", 0.3),
        "saturation_type": low_band_settings.get("saturation_type", "Tube"),
        "mono_strength": low_band_settings.get("mono_strength", 1.0),
        "output_trim_db": low_band_settings.get("output_trim_db", 0),
    }
    
    # High band settings
    high_band_settings = st.session_state.get("high_band_settings", {})
    config["high_band"] = {
        "fft_size": high_band_settings.get("fft_size", 2048),
        "window_type": high_band_settings.get("window_type", "hann"),
        "precision_mode": high_band_settings.get("precision_mode", "Quantized"),
        "mag_decimation": high_band_settings.get("mag_decimation", 0.5),
        "phase_dispersal": high_band_settings.get("phase_dispersal", 0.3),
        "bin_scrambling": high_band_settings.get("bin_scrambling", 0.2),
        "output_trim_db": high_band_settings.get("output_trim_db", 0),
    }
    
    # Quantum FX settings
    quantum_fx_settings = st.session_state.get("quantum_fx_settings", {})
    harmonic_lock_mode = quantum_fx_settings.get("harmonic_lock_mode", "Off")
    custom_fundamental_hz = quantum_fx_settings.get("custom_fundamental_hz", None)
    
    # Map harmonic_lock_mode to fundamental frequency
    fundamental_hz = None
    if harmonic_lock_mode == "F#1":
        fundamental_hz = 46.25
    elif harmonic_lock_mode == "G1":
        fundamental_hz = 49.0
    elif harmonic_lock_mode == "A#1":
        fundamental_hz = 58.27
    elif harmonic_lock_mode == "Custom" and custom_fundamental_hz is not None:
        fundamental_hz = float(custom_fundamental_hz)
    
    config["quantum_fx"] = {
        "spectral_freeze": quantum_fx_settings.get("spectral_freeze", False),
        "formant_shift": quantum_fx_settings.get("formant_shift", 0),
        "harmonic_lock_mode": harmonic_lock_mode,
        "fundamental_hz": fundamental_hz,
    }
    
    # Delta listen setting
    config["delta_listen"] = st.session_state.get("delta_listen", False)
    
    return config


def _audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """
    Convert mono audio buffer to WAV bytes suitable for st.audio().
    """
    buf = BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _init_session_state() -> None:
    if "audio" not in st.session_state:
        st.session_state["audio"] = None  # np.ndarray or None
    if "sr" not in st.session_state:
        st.session_state["sr"] = None  # int or None
    if "processed" not in st.session_state:
        st.session_state["processed"] = None  # np.ndarray or None
    if "taps" not in st.session_state:
        st.session_state["taps"] = None  # Dict[str, np.ndarray] or None

    # Track which file is currently loaded so we don't reset state on every rerun
    if "loaded_file_name" not in st.session_state:
        st.session_state["loaded_file_name"] = None  # str | None


def render_v2_ui() -> None:
    """Render the new V2 single-page layout with signal-flow-based organization."""
    _init_session_state()

    st.title("Quantum Distortion V2")
    st.caption("Spectral Quantized Distortion · Signal-Flow Based Interface")

    # ===========================
    # Top Bar — File Upload, Render, Delta Toggle
    # ===========================
    with st.container():
        st.markdown("### 🎛 Transport")
        col_upload, col_render, col_delta = st.columns([3, 2, 1], gap="medium")

        with col_upload:
            uploaded = st.file_uploader(
                "Load audio (WAV / AIF / AIFF)",
                type=["wav", "aif", "aiff"],
            )

            if uploaded is not None:
                # Only reload and reset state if the file has actually changed.
                last_name = st.session_state.get("loaded_file_name", None)
                if uploaded.name != last_name:
                    data, sr = sf.read(uploaded, always_2d=False)

                    audio = np.asarray(data, dtype=np.float32)
                    if audio.ndim == 2:
                        audio = audio.mean(axis=1).astype(np.float32)

                    st.session_state["audio"] = audio
                    st.session_state["sr"] = int(sr)

                    # Reset DSP results because we have a new input file
                    st.session_state["processed"] = None
                    st.session_state["taps"] = None
                    st.session_state["loaded_file_name"] = uploaded.name

                    st.success(f"Loaded file — {len(audio)} samples @ {sr} Hz")

        with col_render:
            st.markdown("<br>", unsafe_allow_html=True)  # Vertical spacing
            if st.button("Render", type="primary", use_container_width=True):
                audio = st.session_state.get("audio")
                sr = st.session_state.get("sr")

                if audio is None or sr is None:
                    st.error("Please load an audio file first.")
                else:
                    # Build centralized config from session state
                    config = build_processing_config_from_session()
                    
                    # Show warnings for features with limited DSP support
                    low_band = config.get("low_band", {})
                    if low_band.get("saturation_type") != "Tube":
                        st.warning(f"Saturation type '{low_band.get('saturation_type')}' not yet implemented in DSP. Using Tube mode.")

                    high_band = config.get("high_band", {})
                    if high_band.get("fft_size", 2048) != 2048:
                        st.warning(f"FFT size {high_band.get('fft_size')} not yet implemented in DSP. Using default 2048.")
                    if high_band.get("window_type", "hann") != "hann":
                        st.warning(f"Window type '{high_band.get('window_type')}' not yet implemented in DSP. Using Hann window.")
                    
                    # Process with config
                    processed, taps = process_audio(
                        audio=audio,
                        sr=sr,
                        config=config,
                    )
                    
                    st.session_state["processed"] = processed
                    st.session_state["taps"] = taps
                    
                    # Compute and store delta for analysis
                    min_len = min(len(audio), len(processed))
                    delta = processed[:min_len] - audio[:min_len]
                    st.session_state["delta"] = delta
                    
                    st.success("Processing complete!")

        with col_delta:
            st.markdown("<br>", unsafe_allow_html=True)  # Vertical spacing
            delta_listen = st.checkbox(
                "Listen to Delta (differences only)",
                value=st.session_state.get("delta_listen", False),
                key="delta_listen_top",
            )
            # Store in session state
            st.session_state["delta_listen"] = delta_listen

    # Audio playback section
    with st.container():
        st.markdown("### 🎧 Playback")
        audio = st.session_state.get("audio")
        sr = st.session_state.get("sr")
        processed = st.session_state.get("processed")
        delta_listen = st.session_state.get("delta_listen", False)

        if audio is not None and sr is not None:
            col_orig, col_proc = st.columns(2, gap="medium")
            with col_orig:
                st.markdown("**Original Audio**")
                st.audio(_audio_to_wav_bytes(audio, sr), format="audio/wav")
            with col_proc:
                if processed is not None:
                    # Determine what to play based on delta_listen setting
                    if delta_listen:
                        # Compute delta if not already computed
                        delta = st.session_state.get("delta")
                        if delta is None:
                            # Compute delta on the fly
                            min_len = min(len(audio), len(processed))
                            delta = processed[:min_len] - audio[:min_len]
                            st.session_state["delta"] = delta
                        
                        st.markdown("**Delta Signal (Differences Only)**")
                        st.audio(_audio_to_wav_bytes(delta, sr), format="audio/wav")
                        dl_bytes = _audio_to_wav_bytes(delta, sr)
                        st.download_button(
                            "Download Delta WAV",
                            data=dl_bytes,
                            file_name="quantum_distortion_delta.wav",
                            mime="audio/wav",
                        )
                    else:
                        st.markdown("**Processed Audio**")
                        st.audio(_audio_to_wav_bytes(processed, sr), format="audio/wav")
                        dl_bytes = _audio_to_wav_bytes(processed, sr)
                        st.download_button(
                            "Download Processed WAV",
                            data=dl_bytes,
                            file_name="quantum_distortion_processed.wav",
                            mime="audio/wav",
                        )
                else:
                    st.info("Click Render to process audio.")
    
    st.markdown("---")

    # ===========================
    # Signal Flow Overview
    # ===========================
    with st.expander("📊 Signal Flow Overview", expanded=True):
        # Block diagram using columns and markdown
        # Using dark theme colors that work with Streamlit's default theme
        col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1.2, 1, 1.2], gap="small")
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 15px; background-color: #1e1e1e; border-radius: 8px; border: 2px solid #4a9eff; color: #ffffff;">
                <strong>🎵 Input</strong><br>
                <small style="color: #cccccc;">Audio Signal</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 10px; color: #ffffff;">
                <strong>→</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 15px; background-color: #1e1e1e; border-radius: 8px; border: 2px solid #ffa500; color: #ffffff;">
                <strong>🎚 Split Band</strong><br>
                <small style="color: #cccccc;">Linkwitz-Riley<br>Crossover</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="text-align: center; padding: 10px; color: #ffffff;">
                <strong>→</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div style="text-align: center; padding: 15px; background-color: #1e1e1e; border-radius: 8px; border: 2px solid #4caf50; color: #ffffff;">
                <strong>🎛 Mixer</strong><br>
                <small style="color: #cccccc;">Recombine</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Two paths below the split
        col_path1, col_path2 = st.columns(2, gap="medium")
        
        with col_path1:
            st.markdown("""
            <div style="text-align: center; padding: 10px; color: #ffffff;">
                <strong>↓</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: center; padding: 15px; background-color: #1e1e1e; border-radius: 8px; border: 2px solid #4a9eff; color: #ffffff;">
                <strong>🔊 Low Path (Body)</strong><br>
                <small style="color: #cccccc;">Time Domain</small><br>
                <small style="color: #999999;">Saturation + Mono</small>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📍 Go to Low Band Panel", key="goto_low", use_container_width=True):
                st.session_state["highlight_low_band"] = True
                st.rerun()
        
        with col_path2:
            st.markdown("""
            <div style="text-align: center; padding: 10px; color: #ffffff;">
                <strong>↓</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: center; padding: 15px; background-color: #1e1e1e; border-radius: 8px; border: 2px solid #ffa500; color: #ffffff;">
                <strong>✨ High Path (Texture)</strong><br>
                <small style="color: #cccccc;">Spectral Quantum</small><br>
                <small style="color: #999999;">STFT Pipeline</small>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📍 Go to High Band Panel", key="goto_high", use_container_width=True):
                st.session_state["highlight_high_band"] = True
                st.rerun()
        
        # Final output
        st.markdown("""
        <div style="text-align: center; padding: 10px; color: #ffffff;">
            <strong>↓</strong>
        </div>
        """, unsafe_allow_html=True)
        
        col_out1, col_out2, col_out3 = st.columns([1, 1, 1])
        with col_out2:
            st.markdown("""
            <div style="text-align: center; padding: 15px; background-color: #1e1e1e; border-radius: 8px; border: 2px solid #4caf50; color: #ffffff;">
                <strong>🎧 Output</strong><br>
                <small style="color: #cccccc;">Processed Audio</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Explanatory text - dark theme compatible
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background-color: #262730; border-radius: 8px; border-left: 4px solid #4a9eff; color: #ffffff;">
            <p style="margin: 0; color: #ffffff;"><strong>Architecture Overview:</strong></p>
            <p style="margin: 5px 0 0 0; color: #cccccc;">
                The multiband architecture splits audio at a crossover frequency (typically 300 Hz). 
                The <strong style="color: #ffffff;">low band (body)</strong> is processed entirely in the <strong style="color: #ffffff;">time domain</strong> using saturation 
                and mono-making for optimal transient response. The <strong style="color: #ffffff;">high band (texture)</strong> uses the 
                <strong style="color: #ffffff;">spectral quantum pipeline</strong> with STFT-based quantization, distortion, and creative FX 
                for harmonic shaping. Both paths are then recombined at the mixer stage.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ===========================
    # Low Band (Body) Section
    # ===========================
    # Show highlight if navigated from Signal Flow Overview
    if st.session_state.get("highlight_low_band", False):
        st.info("📍 **Low Band (Body) Panel** - Time-domain processing for bass frequencies")
        st.session_state["highlight_low_band"] = False  # Clear flag after showing
    
    with st.container():
        st.subheader("🦴 Low Band (Body)")
        st.caption("Sub weight & punch (time-domain).")
    
    with st.expander("🦴 Low Band (Body)", expanded=True):
        # Initialize low band settings in session state if not present
        if "low_band_settings" not in st.session_state:
            st.session_state["low_band_settings"] = {
                "crossover_freq": 300,
                "saturation_amount": 0.3,
                "saturation_type": "Tube",
                "mono_strength": 1.0,
                "output_trim_db": 0,
            }
        
        # Get current settings
        settings = st.session_state["low_band_settings"]
        
        # Controls layout - use columns for better organization
        col_left, col_right = st.columns(2, gap="medium")
        
        with col_left:
            # Crossover Frequency
            crossover_freq = st.slider(
                "Crossover Frequency (Hz)",
                min_value=80,
                max_value=600,
                value=settings.get("crossover_freq", 300),
                step=10,
                key="v2_low_crossover_freq",
            )
            settings["crossover_freq"] = crossover_freq
            # Store in session state for reuse in High Band panel
            st.session_state["crossover_freq"] = crossover_freq
            
            # Saturation Amount
            low_saturation_amount = st.slider(
                "Saturation Amount",
                min_value=0.0,
                max_value=1.0,
                value=settings.get("saturation_amount", 0.3),
                step=0.05,
                key="v2_low_saturation_amount",
            )
            settings["saturation_amount"] = low_saturation_amount
            
            # Saturation Type
            low_saturation_type = st.radio(
                "Saturation Type",
                ["Tube", "Clip"],
                index=0 if settings.get("saturation_type", "Tube") == "Tube" else 1,
                horizontal=True,
                key="v2_low_saturation_type",
            )
            settings["saturation_type"] = low_saturation_type
        
        with col_right:
            # Mono Maker Strength
            low_mono_strength = st.slider(
                "Mono Maker Strength",
                min_value=0.0,
                max_value=1.0,
                value=settings.get("mono_strength", 1.0),
                step=0.1,
                key="v2_low_mono_strength",
            )
            settings["mono_strength"] = low_mono_strength
            
            # Low Band Output Trim
            low_output_trim_db = st.slider(
                "Output Trim (dB)",
                min_value=-12,
                max_value=12,
                value=settings.get("output_trim_db", 0),
                step=1,
                key="v2_low_output_trim",
            )
            settings["output_trim_db"] = low_output_trim_db
            
            # Update session state
            st.session_state["low_band_settings"] = settings
    
    st.markdown("---")

    # ===========================
    # High Band (Texture) Section
    # ===========================
    # Show highlight if navigated from Signal Flow Overview
    if st.session_state.get("highlight_high_band", False):
        st.info("📍 **High Band (Texture) Panel** - Spectral quantum pipeline for harmonic shaping")
        st.session_state["highlight_high_band"] = False  # Clear flag after showing
    
    with st.container():
        st.subheader("⚡ High Band (Texture)")
        st.caption("Texture & character (spectral domain).")
    
    with st.expander("⚡ High Band (Texture)", expanded=True):
        # Initialize high band settings in session state if not present
        if "high_band_settings" not in st.session_state:
            st.session_state["high_band_settings"] = {
                "fft_size": 2048,
                "window_type": "hann",
                "precision_mode": "Quantized",
                "mag_decimation": 0.5,
                "phase_dispersal": 0.3,
                "bin_scrambling": 0.2,
                "output_trim_db": 0,
            }
        
        # Initialize quantization settings (key/scale) if not present
        if "quantization_settings" not in st.session_state:
            st.session_state["quantization_settings"] = {
                "key": "D",
                "scale": "minor",
            }
        
        # Get current settings
        settings = st.session_state["high_band_settings"]
        quant_settings = st.session_state["quantization_settings"]
        
        # Quantization Settings (Key & Scale) - at the top
        st.markdown("**Quantization Settings**")
        col_key, col_scale = st.columns(2, gap="medium")
        with col_key:
            key = st.selectbox(
                "Key",
                KEY_OPTIONS,
                index=KEY_OPTIONS.index(quant_settings.get("key", "D")) if quant_settings.get("key", "D") in KEY_OPTIONS else 3,
                key="v2_quantization_key",
            )
            quant_settings["key"] = key
        with col_scale:
            scale = st.selectbox(
                "Scale",
                SCALE_OPTIONS,
                index=SCALE_OPTIONS.index(quant_settings.get("scale", "minor")) if quant_settings.get("scale", "minor") in SCALE_OPTIONS else 1,
                key="v2_quantization_scale",
            )
            quant_settings["scale"] = scale
        st.session_state["quantization_settings"] = quant_settings
        
        st.markdown("---")
        
        # Controls layout - organized in columns
        col_stft, col_fx = st.columns(2, gap="medium")
        
        with col_stft:
            st.markdown("**STFT Settings**")
            fft_options = [512, 1024, 2048, 4096]
            current_fft = settings.get("fft_size", 2048)
            fft_index = fft_options.index(current_fft) if current_fft in fft_options else 2
            fft_size = st.selectbox(
                "FFT Size",
                fft_options,
                index=fft_index,
                key="v2_high_fft_size",
            )
            settings["fft_size"] = fft_size
            
            window_type = st.selectbox(
                "Window Type",
                ["hann", "blackmanharris"],
                index=0 if settings.get("window_type", "hann") == "hann" else 1,
                key="v2_high_window_type",
            )
            settings["window_type"] = window_type
            
            precision_options = ["Clean", "Quantized", "Brutal"]
            current_precision = settings.get("precision_mode", "Quantized")
            precision_index = precision_options.index(current_precision) if current_precision in precision_options else 1
            precision_mode = st.selectbox(
                "Precision Mode",
                precision_options,
                index=precision_index,
                key="v2_high_precision_mode",
            )
            settings["precision_mode"] = precision_mode
        
        with col_fx:
            st.markdown("**Spectral Distortion**")
            mag_decimation = st.slider(
                "Magnitude Decimation",
                min_value=0.0,
                max_value=1.0,
                value=settings.get("mag_decimation", 0.5),
                step=0.05,
                key="v2_high_mag_decimation",
            )
            settings["mag_decimation"] = mag_decimation
            
            phase_dispersal = st.slider(
                "Phase Dispersal",
                min_value=0.0,
                max_value=1.0,
                value=settings.get("phase_dispersal", 0.3),
                step=0.05,
                key="v2_high_phase_dispersal",
            )
            settings["phase_dispersal"] = phase_dispersal
            
            bin_scrambling = st.slider(
                "Bin Scrambling",
                min_value=0.0,
                max_value=1.0,
                value=settings.get("bin_scrambling", 0.2),
                step=0.05,
                key="v2_high_bin_scrambling",
            )
            settings["bin_scrambling"] = bin_scrambling
            
            high_output_trim_db = st.slider(
                "Output Trim (dB)",
                min_value=-12,
                max_value=12,
                value=settings.get("output_trim_db", 0),
                step=1,
                key="v2_high_output_trim",
            )
            settings["output_trim_db"] = high_output_trim_db
        
        # Update session state
        st.session_state["high_band_settings"] = settings

    # ===========================
    # Creative Quantum FX Section
    # ===========================
    with st.container():
        st.subheader("🧪 Creative Quantum FX")
        st.caption("Experimental quantum-style transformations.")
    
    with st.expander("🧪 Creative Quantum FX", expanded=True):
        # Initialize quantum FX settings in session state if not present
        if "quantum_fx_settings" not in st.session_state:
            st.session_state["quantum_fx_settings"] = {
                "spectral_freeze": False,
                "formant_shift": 0,
                "harmonic_lock_mode": "Off",
                "custom_fundamental_hz": None,
            }
        
        # Get current settings
        settings = st.session_state["quantum_fx_settings"]
        
        # Controls - organized in columns
        col_fx1, col_fx2 = st.columns(2, gap="medium")
        
        with col_fx1:
            spectral_freeze = st.checkbox(
                "Spectral Freeze",
                value=settings.get("spectral_freeze", False),
                help="Hold first-frame spectral texture for entire clip",
                key="v2_quantum_spectral_freeze",
            )
            settings["spectral_freeze"] = spectral_freeze
            
            formant_shift = st.slider(
                "Formant Shift (%)",
                min_value=-100,
                max_value=100,
                value=settings.get("formant_shift", 0),
                step=5,
                key="v2_quantum_formant_shift",
            )
            settings["formant_shift"] = formant_shift
        
        with col_fx2:
            harmonic_lock_options = ["Off", "F#1", "G1", "A#1", "Custom"]
            current_lock_mode = settings.get("harmonic_lock_mode", "Off")
            lock_index = harmonic_lock_options.index(current_lock_mode) if current_lock_mode in harmonic_lock_options else 0
            harmonic_lock_mode = st.selectbox(
                "Harmonic Locking",
                harmonic_lock_options,
                index=lock_index,
                key="v2_quantum_harmonic_lock",
            )
            settings["harmonic_lock_mode"] = harmonic_lock_mode
            
            # Custom fundamental input (only shown if "Custom" is selected)
            custom_fundamental_hz = None
            if harmonic_lock_mode == "Custom":
                custom_fundamental_hz = st.number_input(
                    "Custom Fundamental (Hz)",
                    min_value=20.0,
                    max_value=2000.0,
                    value=float(settings.get("custom_fundamental_hz", 55.0)),
                    step=1.0,
                    key="v2_quantum_custom_fundamental",
                )
                settings["custom_fundamental_hz"] = custom_fundamental_hz
            else:
                settings["custom_fundamental_hz"] = None
        
        # Update session state
        st.session_state["quantum_fx_settings"] = settings
        
        # Explanatory text
        with st.expander("💡 What do these do?", expanded=False):
            st.markdown("""
            - **Spectral Freeze:** Captures and holds the current spectral texture, creating a sustained "frozen" harmonic character. Perfect for evolving pads or glitchy stutter effects.
            - **Formant Shift:** Shifts formant frequencies (vocal character) up or down without changing pitch. Positive values add brightness, negative values add warmth.
            - **Harmonic Locking:** Forces the spectrum to lock onto specific fundamental frequencies. Creates a strong harmonic foundation for bass design.
            """)
    
    st.markdown("---")

    # ===========================
    # Analysis Tools Section
    # ===========================
    with st.container():
        st.subheader("🔬 Analysis Tools")
        st.caption("Verification and Delta listening.")
    
    with st.expander("🔬 Analysis Tools", expanded=False):
        audio = st.session_state.get("audio")
        processed = st.session_state.get("processed")
        sr = st.session_state.get("sr")
        
        if audio is None or processed is None or sr is None:
            st.info("Render audio to enable analysis tools.")
        else:
            # Compute delta = output - input
            # Ensure both signals are the same length
            min_len = min(len(audio), len(processed))
            audio_aligned = audio[:min_len]
            processed_aligned = processed[:min_len]
            delta = processed_aligned - audio_aligned
            
            # Compute summary statistics
            delta_peak = np.max(np.abs(delta))
            delta_peak_db = 20.0 * np.log10(delta_peak) if delta_peak > 0.0 else float('-inf')
            delta_rms = np.sqrt(np.mean(delta ** 2))
            delta_rms_db = 20.0 * np.log10(delta_rms) if delta_rms > 0.0 else float('-inf')
            delta_length_samples = len(delta)
            delta_length_seconds = delta_length_samples / float(sr)
            
            # Store delta in session state for playback/download
            st.session_state["delta"] = delta
            
            # Summary section
            st.markdown("### Delta Signal Summary")
            col_sum1, col_sum2 = st.columns(2)
            with col_sum1:
                st.metric("Length", f"{delta_length_seconds:.2f} s ({delta_length_samples:,} samples)")
                st.metric("Peak Level", f"{delta_peak_db:.2f} dB" if delta_peak_db > float('-inf') else "-∞ dB")
            with col_sum2:
                st.metric("RMS Level", f"{delta_rms_db:.2f} dB" if delta_rms_db > float('-inf') else "-∞ dB")
                st.metric("Peak Amplitude", f"{delta_peak:.6f}")
            
            # Explanatory text
            st.info("""
            **Delta Signal:** The difference between processed and original audio. 
            This reveals only what the processor is adding or changing, making it easier 
            to hear the effect in isolation.
            """)
            
            # Spectrogram placeholder
            st.markdown("### Spectrograms")
            st.info("Spectrogram visualization will be available here after render. Placeholder for future implementation.")
            # TODO: Generate and display spectrograms for input, output, and delta
            # Can use existing visualizers.py functions or create new spectrogram plotting function
            
            # Download Delta Audio button
            st.markdown("### Export")
            delta_bytes = _audio_to_wav_bytes(delta, sr)
            st.download_button(
                "Download Delta Audio (WAV)",
                data=delta_bytes,
                file_name="quantum_distortion_delta.wav",
                mime="audio/wav",
            )


def main() -> None:
    """Main entry point for the Quantum Distortion Streamlit UI."""
    st.set_page_config(page_title="Quantum Distortion", layout="wide")
    render_v2_ui()


if __name__ == "__main__":
    main()

