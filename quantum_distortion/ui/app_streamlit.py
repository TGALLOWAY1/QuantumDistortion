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
from typing import Dict


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
    st.set_page_config(page_title="Quantum Distortion MVP", layout="wide")
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
    st.set_page_config(page_title="Quantum Distortion V2", layout="wide")
    _init_session_state()

    st.title("Quantum Distortion V2")
    st.caption("Spectral Quantized Distortion · Signal-Flow Based Interface")

    # ===========================
    # Top Bar — File Upload, Render, Delta Toggle
    # ===========================
    with st.container():
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
                    # For now, use default parameters from V1 UI
                    # This will be replaced with V2 controls later
                    processed, taps = process_audio(
                        audio=audio,
                        sr=sr,
                        # Using default values for now
                    )

                    st.session_state["processed"] = processed
                    st.session_state["taps"] = taps
                    st.success("Processing complete!")

        with col_delta:
            st.markdown("<br>", unsafe_allow_html=True)  # Vertical spacing
            listen_to_delta = st.checkbox("Listen to Delta", value=False)
            # Store in session state for future use
            st.session_state["listen_to_delta"] = listen_to_delta

    # Audio playback section
    audio = st.session_state.get("audio")
    sr = st.session_state.get("sr")
    processed = st.session_state.get("processed")

    if audio is not None and sr is not None:
        col_orig, col_proc = st.columns(2, gap="medium")
        with col_orig:
            st.markdown("**Original Audio**")
            st.audio(_audio_to_wav_bytes(audio, sr), format="audio/wav")
        with col_proc:
            if processed is not None:
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
        st.subheader("Signal Flow Overview")
        
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
    
    with st.expander("Low Band (Body)", expanded=True):
        st.write("Low band processing controls will be added here.")
        st.write("This section will include:")
        st.write("- Crossover frequency settings")
        st.write("- Low band drive/saturation")
        st.write("- Low band-specific quantization")
        st.write("- Low band visualization")

    # ===========================
    # High Band (Texture) Section
    # ===========================
    # Show highlight if navigated from Signal Flow Overview
    if st.session_state.get("highlight_high_band", False):
        st.info("📍 **High Band (Texture) Panel** - Spectral quantum pipeline for harmonic shaping")
        st.session_state["highlight_high_band"] = False  # Clear flag after showing
    
    with st.expander("High Band (Texture)", expanded=True):
        st.write("High band processing controls will be added here.")
        st.write("This section will include:")
        st.write("- High band drive/saturation")
        st.write("- High band-specific quantization")
        st.write("- High band visualization")

    # ===========================
    # Creative Quantum FX Section
    # ===========================
    with st.expander("Creative Quantum FX", expanded=True):
        st.write("Creative spectral effects controls will be added here.")
        st.write("This section will include:")
        st.write("- Quantum FX mode selection")
        st.write("- FX strength/intensity")
        st.write("- Spectral manipulation parameters")
        st.write("- Real-time preview options")

    # ===========================
    # Analysis Tools Section
    # ===========================
    with st.expander("Analysis Tools", expanded=False):
        st.write("Enhanced analysis and visualization tools will be added here.")
        st.write("This section will include:")
        st.write("- Delta visualization (before/after comparison)")
        st.write("- Spectrogram views")
        st.write("- Multi-tap comparison")
        st.write("- Export options")

        # Show basic info if processing has been done
        processed = st.session_state.get("processed")
        taps = st.session_state.get("taps")
        if processed is not None and taps is not None:
            st.success("Analysis data available. Full visualization tools coming soon.")


def main() -> None:
    """Main entry point that routes to V1 or V2 UI based on feature flag."""
    if USE_V2_UI:
        render_v2_ui()
    else:
        render_v1_ui()


if __name__ == "__main__":
    main()

