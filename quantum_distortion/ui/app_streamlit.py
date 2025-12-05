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


import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict


# Add project root to Python path for Streamlit
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Feature flag for V2 UI (can be overridden via environment variable)
USE_V2_UI = os.getenv("QD_USE_V2_UI", "true").lower() in ("true", "1", "yes")


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
        - low_band: dict with saturation, mono, output_trim settings
        - high_band: dict with STFT, spectral FX settings
        - quantum_fx: dict with freeze, formant, harmonic locking settings
        - delta_listen: bool
    """
    import streamlit as st
    
    config: Dict[str, Any] = {}
    
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


def render_v1_ui() -> None:
    """Render the original MVP slider-based UI layout."""
    _init_session_state()

    st.title("Quantum Distortion — MVP Prototype")
    st.caption("Spectral Quantized Distortion · Python DSP Prototype")

    # ===========================
    # Panel 1 — File & Transport
    # ===========================
    with st.container():
        st.subheader("1. File & Transport")

        col_file, col_play = st.columns([2, 2], gap="large")

        with col_file:
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

        with col_play:
            audio = st.session_state["audio"]
            sr = st.session_state["sr"]

            if audio is not None and sr is not None:
                st.markdown("**Original Audio**")
                st.audio(_audio_to_wav_bytes(audio, sr), format="audio/wav")

                if st.session_state["processed"] is not None:
                    st.markdown("**Processed Audio**")
                    processed = st.session_state["processed"]
                    st.audio(_audio_to_wav_bytes(processed, sr), format="audio/wav")

                    # Download processed
                    dl_bytes = _audio_to_wav_bytes(processed, sr)
                    st.download_button(
                        "Download Processed WAV",
                        data=dl_bytes,
                        file_name="quantum_distortion_processed.wav",
                        mime="audio/wav",
                    )
            else:
                st.info("Upload an audio file to begin.")

    st.markdown("---")

    audio = st.session_state["audio"]
    sr = st.session_state["sr"]
    if audio is None or sr is None:
        st.stop()

    # ===========================
    # Panel 2 — Quantization
    # ===========================
    with st.container():
        st.subheader("2. Quantization Settings")

        col_q1, col_q2 = st.columns(2)

        with col_q1:
            key = st.selectbox("Key", KEY_OPTIONS, index=0)
            scale = st.selectbox("Scale", SCALE_OPTIONS, index=1)  # default minor

            snap_percent = st.slider("Snap Strength (%)", min_value=0, max_value=100, value=80)
            smear_percent = st.slider("Smear (%)", min_value=0, max_value=100, value=30)

        with col_q2:
            bin_smoothing = st.checkbox("Bin Smoothing (ON)", value=True)
            pre_quant = st.checkbox("Pre-Quantize", value=True)
            post_quant = st.checkbox("Post-Quantize", value=True)

    st.markdown("---")

    # ===========================
    # Panel 3 — Distortion
    # ===========================
    with st.container():
        st.subheader("3. Distortion")

        mode = st.radio("Mode", options=["Wavefold", "Soft Tube"], index=0, horizontal=True)

        if mode == "Wavefold":
            fold_amount = st.slider("Fold Amount", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
            bias = st.slider("Symmetry / Bias", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)
            drive = 1.0
            warmth = 0.5
        else:
            drive = st.slider("Drive", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
            warmth = st.slider("Warmth", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            fold_amount = 1.0
            bias = 0.0

    st.markdown("---")

    # ===========================
    # Panel 4 — Output
    # ===========================
    with st.container():
        st.subheader("4. Output")

        col_o1, col_o2 = st.columns(2)

        with col_o1:
            limiter_on = st.checkbox("Limiter ON", value=True)
            ceiling_db = st.slider("Limiter Ceiling (dBFS)", min_value=-12.0, max_value=-0.1, value=-1.0, step=0.1)

        with col_o2:
            dry_wet_percent = st.slider("Dry / Wet (%)", min_value=0, max_value=100, value=100)
            output_gain_db = st.slider("Output Gain (dB)", min_value=-12.0, max_value=12.0, value=0.0, step=0.5)

    # ===========================
    # Render Button
    # ===========================
    st.markdown("### Render")

    if st.button("Render / Process Audio", type="primary"):
        snap_strength = snap_percent / 100.0
        smear = smear_percent / 100.0
        dry_wet = dry_wet_percent / 100.0

        distortion_mode = "wavefold" if mode == "Wavefold" else "tube"
        distortion_params: Dict[str, float] = {
            "fold_amount": float(fold_amount),
            "bias": float(bias),
            "drive": float(drive),
            "warmth": float(warmth),
        }

        processed, taps = process_audio(
            audio=audio,
            sr=sr,
            key=key,
            scale=scale,
            snap_strength=snap_strength,
            smear=smear,
            bin_smoothing=bin_smoothing,
            pre_quant=pre_quant,
            post_quant=post_quant,
            distortion_mode=distortion_mode,
            distortion_params=distortion_params,
            limiter_on=limiter_on,
            limiter_ceiling_db=ceiling_db,
            dry_wet=dry_wet,
        )

        # Apply output gain (for playback/visualization only)
        gain_lin = 10.0 ** (output_gain_db / 20.0)
        processed = (processed * gain_lin).astype(np.float32)

        st.session_state["processed"] = processed
        st.session_state["taps"] = taps
        st.success("Processing complete. Scroll down to Analysis Panel.")

    st.markdown("---")

    # ===========================
    # Panel 5 — Analysis
    # ===========================
    st.subheader("5. Analysis & Visualization")

    taps = st.session_state["taps"]
    processed = st.session_state["processed"]

    if taps is None or processed is None:
        st.info("Render the audio to enable analysis.")
        return

    # Work on a copy so we don't mutate session_state
    taps_local = dict(taps)
    taps_local["output"] = processed

    col_a1, col_a2 = st.columns([1, 3])

    with col_a1:
        tap_source = st.selectbox("Tap Source", TAP_OPTIONS, index=3)
        visual_type = st.radio("View", VISUAL_OPTIONS, index=0)

        show_scale_lines = st.checkbox("Show In-Key Lines (Spectrum)", value=True)
        max_freq = st.slider(
            "Max Frequency (Hz)",
            min_value=2000,
            max_value=int(sr // 2),
            value=int(sr // 2),
            step=1000,
        )

    with col_a2:
        buf = taps_local[tap_source]

        if visual_type == "Spectrum":
            fig = plot_spectrum(
                audio=buf,
                sr=sr,
                tap_source=tap_source,  # type: ignore[arg-type]
                key=key,
                scale=scale,
                show_scale_lines=show_scale_lines,
                max_freq=float(max_freq),
            )
            st.pyplot(fig, clear_figure=True)

        elif visual_type == "Oscilloscope":
            fig = plot_oscilloscope(
                audio=buf,
                sr=sr,
                tap_source=tap_source,  # type: ignore[arg-type]
                duration=0.02,
            )
            st.pyplot(fig, clear_figure=True)

        else:  # "Phase Scope"
            fig = plot_phase_scope(
                audio=buf,
                sr=sr,
                tap_source=tap_source,  # type: ignore[arg-type]
            )
            st.pyplot(fig, clear_figure=True)


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
                    
                    # Show warnings for features not yet implemented
                    low_band = config.get("low_band", {})
                    if low_band.get("saturation_type") != "Tube":
                        st.warning(f"Saturation type '{low_band.get('saturation_type')}' not yet implemented in DSP. Using Tube mode.")
                    if low_band.get("mono_strength", 1.0) < 1.0:
                        st.warning(f"Mono strength {low_band.get('mono_strength')} not yet implemented in DSP. Using full mono (1.0).")
                    if low_band.get("output_trim_db", 0) != 0:
                        st.warning(f"Low band output trim ({low_band.get('output_trim_db')} dB) not yet implemented in DSP.")
                    
                    high_band = config.get("high_band", {})
                    if high_band.get("fft_size", 2048) != 2048:
                        st.warning(f"FFT size {high_band.get('fft_size')} not yet implemented in DSP. Using default 2048.")
                    if high_band.get("window_type", "hann") != "hann":
                        st.warning(f"Window type '{high_band.get('window_type')}' not yet implemented in DSP. Using Hann window.")
                    if high_band.get("precision_mode", "Quantized") != "Quantized":
                        st.info(f"Precision mode '{high_band.get('precision_mode')}' not yet implemented. Using default quantization.")
                    if high_band.get("output_trim_db", 0) != 0:
                        st.warning(f"High band output trim ({high_band.get('output_trim_db')} dB) not yet implemented in DSP.")
                    
                    quantum_fx = config.get("quantum_fx", {})
                    if quantum_fx.get("spectral_freeze", False):
                        st.info("Spectral Freeze: Feature not yet implemented in DSP. Will hold current spectral texture when available.")
                    if quantum_fx.get("formant_shift", 0) != 0:
                        st.info(f"Formant Shift ({quantum_fx.get('formant_shift')}%): Feature not yet implemented in DSP. Will shift vocal character when available.")
                    if quantum_fx.get("fundamental_hz") is not None:
                        st.info(f"Harmonic Locking ({quantum_fx.get('fundamental_hz'):.2f} Hz): Feature not yet implemented in DSP. Will lock harmonics when available.")
                    
                    # Process with config
                    processed, taps = process_audio(
                        audio=audio,
                        sr=sr,
                        config=config,
                    )
                    
                    # TODO: Apply low_output_trim_db gain if implemented
                    # For now, this is a placeholder
                    # if output_trim_db != 0:
                    #     gain_lin = 10.0 ** (output_trim_db / 20.0)
                    #     # Apply to low band only - would need access to low_processed from taps
                    
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
    with st.container():
        st.subheader("📊 Signal Flow Overview")
        
        # Block diagram using columns and markdown
        col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1.2, 1, 1.2], gap="small")
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 15px; background-color: #f0f2f6; border-radius: 8px; border: 2px solid #1f77b4;">
                <strong>🎵 Input</strong><br>
                <small>Audio Signal</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 10px;">
                <strong>→</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 15px; background-color: #f0f2f6; border-radius: 8px; border: 2px solid #ff7f0e;">
                <strong>🎚 Split Band</strong><br>
                <small>Linkwitz-Riley<br>Crossover</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="text-align: center; padding: 10px;">
                <strong>→</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div style="text-align: center; padding: 15px; background-color: #f0f2f6; border-radius: 8px; border: 2px solid #2ca02c;">
                <strong>🎛 Mixer</strong><br>
                <small>Recombine</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Two paths below the split
        col_path1, col_path2 = st.columns(2, gap="medium")
        
        with col_path1:
            st.markdown("""
            <div style="text-align: center; padding: 10px;">
                <strong>↓</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: center; padding: 15px; background-color: #e8f4f8; border-radius: 8px; border: 2px solid #1f77b4;">
                <strong>🔊 Low Path (Body)</strong><br>
                <small>Time Domain</small><br>
                <small style="color: #666;">Saturation + Mono</small>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📍 Go to Low Band Panel", key="goto_low", use_container_width=True):
                st.session_state["highlight_low_band"] = True
                st.rerun()
        
        with col_path2:
            st.markdown("""
            <div style="text-align: center; padding: 10px;">
                <strong>↓</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: center; padding: 15px; background-color: #fff4e6; border-radius: 8px; border: 2px solid #ff7f0e;">
                <strong>✨ High Path (Texture)</strong><br>
                <small>Spectral Quantum</small><br>
                <small style="color: #666;">STFT Pipeline</small>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📍 Go to High Band Panel", key="goto_high", use_container_width=True):
                st.session_state["highlight_high_band"] = True
                st.rerun()
        
        # Final output
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <strong>↓</strong>
        </div>
        """, unsafe_allow_html=True)
        
        col_out1, col_out2, col_out3 = st.columns([1, 1, 1])
        with col_out2:
            st.markdown("""
            <div style="text-align: center; padding: 15px; background-color: #f0f2f6; border-radius: 8px; border: 2px solid #2ca02c;">
                <strong>🎧 Output</strong><br>
                <small>Processed Audio</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Explanatory text
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 8px; border-left: 4px solid #1f77b4;">
            <p style="margin: 0;"><strong>Architecture Overview:</strong></p>
            <p style="margin: 5px 0 0 0; color: #555;">
                The multiband architecture splits audio at a crossover frequency (typically 300 Hz). 
                The <strong>low band (body)</strong> is processed entirely in the <strong>time domain</strong> using saturation 
                and mono-making for optimal transient response. The <strong>high band (texture)</strong> uses the 
                <strong>spectral quantum pipeline</strong> with STFT-based quantization, distortion, and creative FX 
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
            )
            settings["saturation_amount"] = low_saturation_amount
            
            # Saturation Type
            low_saturation_type = st.radio(
                "Saturation Type",
                ["Tube", "Clip"],
                index=0 if settings.get("saturation_type", "Tube") == "Tube" else 1,
                horizontal=True,
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
            )
            settings["mono_strength"] = low_mono_strength
            
            # Low Band Output Trim
            low_output_trim_db = st.slider(
                "Output Trim (dB)",
                min_value=-12,
                max_value=12,
                value=settings.get("output_trim_db", 0),
                step=1,
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
        
        # Get current settings
        settings = st.session_state["high_band_settings"]
        
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
            )
            settings["fft_size"] = fft_size
            
            window_type = st.selectbox(
                "Window Type",
                ["hann", "blackmanharris"],
                index=0 if settings.get("window_type", "hann") == "hann" else 1,
            )
            settings["window_type"] = window_type
            
            precision_options = ["Clean", "Quantized", "Brutal"]
            current_precision = settings.get("precision_mode", "Quantized")
            precision_index = precision_options.index(current_precision) if current_precision in precision_options else 1
            precision_mode = st.selectbox(
                "Precision Mode",
                precision_options,
                index=precision_index,
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
            )
            settings["mag_decimation"] = mag_decimation
            
            phase_dispersal = st.slider(
                "Phase Dispersal",
                min_value=0.0,
                max_value=1.0,
                value=settings.get("phase_dispersal", 0.3),
                step=0.05,
            )
            settings["phase_dispersal"] = phase_dispersal
            
            bin_scrambling = st.slider(
                "Bin Scrambling",
                min_value=0.0,
                max_value=1.0,
                value=settings.get("bin_scrambling", 0.2),
                step=0.05,
            )
            settings["bin_scrambling"] = bin_scrambling
            
            high_output_trim_db = st.slider(
                "Output Trim (dB)",
                min_value=-12,
                max_value=12,
                value=settings.get("output_trim_db", 0),
                step=1,
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
                help="Hold current texture (not yet implemented)",
            )
            settings["spectral_freeze"] = spectral_freeze
            
            formant_shift = st.slider(
                "Formant Shift (%)",
                min_value=-100,
                max_value=100,
                value=settings.get("formant_shift", 0),
                step=5,
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
    """Main entry point that routes to V1 or V2 UI based on feature flag."""
    # Set page config first (must be first Streamlit command, before any other Streamlit calls)
    st.set_page_config(page_title="Quantum Distortion", layout="wide")
    
    # Dev-only sidebar toggle (overrides environment variable when app is running)
    # This allows easy switching during development without restarting
    # Environment variable QD_USE_V2_UI sets the default, but sidebar can override
    use_v2_ui = st.sidebar.checkbox(
        "Use V2 UI (dev)",
        value=USE_V2_UI,
        help="Toggle between V1 (MVP) and V2 (new layout) interfaces. Default from QD_USE_V2_UI env var.",
    )
    
    if use_v2_ui:
        render_v2_ui()
    else:
        render_v1_ui()


if __name__ == "__main__":
    main()

