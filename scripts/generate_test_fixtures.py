"""
Generate synthetic audio test fixtures for Quantum Distortion.

This script is part of M0 – Baseline & Audio Test Harness.

It generates reproducible, synthetic audio files that can be used for:
- Regression testing
- Listening tests
- Performance benchmarking
- Integration tests

Run this script whenever you need to regenerate the test fixtures:
    python scripts/generate_test_fixtures.py

The fixtures are saved to tests/data/ and are designed to be short (5 seconds)
and cover various frequency ranges and signal characteristics relevant to bass
processing and quantum distortion effects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


# Configuration constants
SAMPLE_RATE = 48000
DURATION_SECONDS = 5.0
OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "data"


def generate_sub_sweep(output_path: Path, sr: int, duration: float) -> None:
    """
    Generate a frequency sweep from ~30 Hz to ~120 Hz.
    
    Uses a linear frequency sweep (chirp) in the sub-bass range.
    """
    n_samples = int(sr * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    
    # Linear frequency sweep: f(t) = f0 + (f1 - f0) * t / duration
    f0 = 30.0  # Start frequency (Hz)
    f1 = 120.0  # End frequency (Hz)
    
    # Phase accumulation for linear chirp
    # phase(t) = 2π * ∫[f0 + (f1-f0)*τ/duration] dτ from 0 to t
    #         = 2π * [f0*t + (f1-f0)*t²/(2*duration)]
    phase = 2.0 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * duration))
    
    # Generate sine wave with amplitude envelope to avoid clicks
    amplitude = 0.3 * np.ones_like(t)
    # Fade in/out to avoid clicks
    fade_samples = int(0.05 * sr)  # 50ms fade
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        amplitude[:fade_samples] *= fade_in
        amplitude[-fade_samples:] *= fade_out
    
    audio = amplitude * np.sin(phase).astype(np.float32)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr)
    print(f"Generated: {output_path.name} ({duration:.1f}s, {sr} Hz)")


def generate_wobble_bass(output_path: Path, sr: int, duration: float) -> None:
    """
    Generate a bass tone with LFO-like amplitude modulation (wobble effect).
    
    Uses a fundamental bass frequency (~60 Hz) with a slow amplitude modulation
    to create a "wobble" effect typical of bass music.
    """
    n_samples = int(sr * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    
    # Fundamental bass frequency
    bass_freq = 60.0  # Hz
    
    # LFO for wobble (slow modulation)
    lfo_freq = 2.0  # Hz (2 wobbles per second)
    lfo_depth = 0.4  # Modulation depth (0-1)
    lfo = 1.0 - lfo_depth + lfo_depth * (1.0 + np.sin(2.0 * np.pi * lfo_freq * t)) / 2.0
    
    # Generate bass tone with amplitude modulation
    bass_phase = 2.0 * np.pi * bass_freq * t
    audio = 0.3 * lfo * np.sin(bass_phase).astype(np.float32)
    
    # Fade in/out
    fade_samples = int(0.05 * sr)
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr)
    print(f"Generated: {output_path.name} ({duration:.1f}s, {sr} Hz)")


def generate_kick_sub_combo(output_path: Path, sr: int, duration: float) -> None:
    """
    Generate alternating kick bursts and sustained sub tones.
    
    Creates a simple pattern: kick (short burst) -> sub (sustained) -> kick -> sub...
    """
    n_samples = int(sr * duration)
    audio = np.zeros(n_samples, dtype=np.float32)
    
    # Pattern timing
    pattern_duration = 1.0  # seconds per pattern (kick + sub)
    kick_duration = 0.1  # seconds for kick
    sub_duration = pattern_duration - kick_duration  # remaining time for sub
    
    kick_samples = int(sr * kick_duration)
    sub_samples = int(sr * sub_duration)
    pattern_samples = kick_samples + sub_samples
    
    n_patterns = int(np.ceil(duration / pattern_duration))
    
    for i in range(n_patterns):
        start_idx = i * pattern_samples
        if start_idx >= n_samples:
            break
        
        # Generate kick: short burst with exponential decay
        kick_end = min(start_idx + kick_samples, n_samples)
        kick_length = kick_end - start_idx
        if kick_length > 0:
            # Kick: short burst at ~60 Hz with fast decay
            kick_t = np.linspace(0.0, kick_duration, kick_length, endpoint=False)
            kick_freq = 60.0
            kick_phase = 2.0 * np.pi * kick_freq * kick_t
            # Exponential decay envelope
            decay = np.exp(-kick_t * 20.0)  # Fast decay
            kick_signal = 0.5 * decay * np.sin(kick_phase)
            audio[start_idx:kick_end] = kick_signal.astype(np.float32)
        
        # Generate sub: sustained tone
        sub_start = kick_end
        sub_end = min(sub_start + sub_samples, n_samples)
        sub_length = sub_end - sub_start
        if sub_length > 0:
            sub_t = np.linspace(0.0, sub_duration, sub_length, endpoint=False)
            sub_freq = 40.0  # Hz
            sub_phase = 2.0 * np.pi * sub_freq * sub_t
            # Gentle fade in for sub
            fade_in = np.linspace(0.0, 1.0, min(int(0.02 * sr), sub_length))
            fade_out = np.linspace(1.0, 0.0, min(int(0.02 * sr), sub_length))
            sub_envelope = np.ones(sub_length)
            if len(fade_in) > 0:
                sub_envelope[:len(fade_in)] = fade_in
            if len(fade_out) > 0:
                sub_envelope[-len(fade_out):] = fade_out
            sub_signal = 0.25 * sub_envelope * np.sin(sub_phase)
            audio[sub_start:sub_end] = sub_signal.astype(np.float32)
    
    # Ensure we don't exceed duration
    audio = audio[:n_samples]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr)
    print(f"Generated: {output_path.name} ({duration:.1f}s, {sr} Hz)")


def generate_midrange_growl_like(output_path: Path, sr: int, duration: float) -> None:
    """
    Generate a midrange tone with basic distortion/modulation for "growl-ish" texture.
    
    Uses a midrange fundamental with some harmonics and amplitude modulation
    to create a harmonically rich, modulated signal.
    """
    n_samples = int(sr * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    
    # Fundamental in midrange
    fundamental = 200.0  # Hz
    
    # Generate signal with harmonics
    signal = np.zeros_like(t)
    
    # Fundamental
    phase = 2.0 * np.pi * fundamental * t
    signal += 0.3 * np.sin(phase)
    
    # Add some harmonics for richness
    signal += 0.15 * np.sin(2.0 * phase)  # 2nd harmonic
    signal += 0.1 * np.sin(3.0 * phase)   # 3rd harmonic
    
    # Apply amplitude modulation for "growl" effect
    mod_freq = 5.0  # Hz (modulation rate)
    mod_depth = 0.3
    modulation = 1.0 - mod_depth + mod_depth * (1.0 + np.sin(2.0 * np.pi * mod_freq * t)) / 2.0
    signal *= modulation
    
    # Apply basic soft clipping distortion (tanh) for growl texture
    signal = np.tanh(signal * 1.5) * 0.4
    
    # Fade in/out
    fade_samples = int(0.05 * sr)
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        signal[:fade_samples] *= fade_in
        signal[-fade_samples:] *= fade_out
    
    audio = signal.astype(np.float32)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr)
    print(f"Generated: {output_path.name} ({duration:.1f}s, {sr} Hz)")


def main() -> None:
    """Generate all test fixtures."""
    print(f"Generating test fixtures to {OUTPUT_DIR}")
    print(f"Sample rate: {SAMPLE_RATE} Hz, Duration: {DURATION_SECONDS} s")
    print()
    
    generate_sub_sweep(
        OUTPUT_DIR / "sub_sweep.wav",
        SAMPLE_RATE,
        DURATION_SECONDS,
    )
    
    generate_wobble_bass(
        OUTPUT_DIR / "wobble_bass.wav",
        SAMPLE_RATE,
        DURATION_SECONDS,
    )
    
    generate_kick_sub_combo(
        OUTPUT_DIR / "kick_sub_combo.wav",
        SAMPLE_RATE,
        DURATION_SECONDS,
    )
    
    generate_midrange_growl_like(
        OUTPUT_DIR / "midrange_growl_like.wav",
        SAMPLE_RATE,
        DURATION_SECONDS,
    )
    
    print()
    print("All fixtures generated successfully!")


if __name__ == "__main__":
    main()

