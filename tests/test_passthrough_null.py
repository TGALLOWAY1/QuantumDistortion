"""
Unit test for OLA passthrough transparency verification.

Tests that passthrough_test mode reconstructs input signals transparently,
verifying that the OLA-compliant STFT/ISTFT roundtrip is correct.
"""

from typing import Tuple

import numpy as np

from quantum_distortion.dsp.pipeline import process_audio


def _make_sine_sweep(
    freq_start: float,
    freq_end: float,
    sr: int = 44100,
    seconds: float = 0.5,
) -> Tuple[np.ndarray, int]:
    """
    Generate a simple linear sine sweep.
    
    Args:
        freq_start: Starting frequency in Hz
        freq_end: Ending frequency in Hz
        sr: Sample rate
        seconds: Duration in seconds
    
    Returns:
        (audio, sample_rate)
    """
    n_samples = int(sr * seconds)
    t = np.linspace(0.0, seconds, n_samples, endpoint=False)
    
    # Linear frequency sweep
    freq = freq_start + (freq_end - freq_start) * t / seconds
    
    # Generate sweep: integrate frequency to get phase
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    
    # Normalize amplitude
    x = 0.1 * np.sin(phase).astype(np.float32)
    
    return x, sr


def _compute_rms_db(signal: np.ndarray) -> float:
    """Compute RMS in dB."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms == 0.0:
        return float('-inf')
    return 20.0 * np.log10(rms)


def test_passthrough_ola_transparency() -> None:
    """
    Test that passthrough_test mode reconstructs input transparently.
    
    This verifies that the OLA-compliant STFT/ISTFT roundtrip is correct.
    Success threshold: RMS of delta < -80 dB (CI-safe threshold).
    """
    # Generate a simple sine sweep (100 Hz to 2000 Hz over 0.5 seconds)
    # A sweep is better than a pure tone because it exercises the full
    # frequency range and temporal structure of the STFT
    input_audio, sr = _make_sine_sweep(
        freq_start=100.0,
        freq_end=2000.0,
        sr=44100,
        seconds=0.5,
    )
    
    # Run through pipeline with passthrough_test=True
    # This should perform transparent STFT->ISTFT roundtrip
    output_audio, taps = process_audio(
        input_audio,
        sr=sr,
        passthrough_test=True,  # Bypass all FX, transparent STFT->ISTFT
    )
    
    # Verify output shape matches input
    assert output_audio.shape == input_audio.shape, \
        f"Output shape {output_audio.shape} != input shape {input_audio.shape}"
    
    # Align lengths (should already match, but be safe)
    min_length = min(len(input_audio), len(output_audio))
    input_aligned = input_audio[:min_length]
    output_aligned = output_audio[:min_length]
    
    # Compute delta = output - input
    delta = output_aligned - input_aligned
    
    # Compute RMS of delta in dB
    rms_delta = np.sqrt(np.mean(delta ** 2))
    rms_delta_db = _compute_rms_db(delta)
    
    # CI-safe threshold: RMS < -80 dB
    # This is more lenient than the script threshold (-90 dB) to account for
    # potential CI environment differences, but still very strict
    threshold_db = -80.0
    
    assert rms_delta_db < threshold_db, \
        f"OLA reconstruction failed: RMS delta = {rms_delta_db:.2f} dB " \
        f"(threshold: {threshold_db} dB). " \
        f"RMS = {rms_delta:.6e}, Peak = {np.max(np.abs(delta)):.6e}"
    
    # Verify taps structure
    assert set(taps.keys()) == {"input", "pre_quant", "post_dist", "output"}, \
        f"Unexpected taps keys: {set(taps.keys())}"


def test_passthrough_simple_sine() -> None:
    """
    Additional test with a simple sine wave (simpler signal).
    
    This provides a baseline test with a pure tone.
    """
    sr = 44100
    seconds = 0.2
    freq = 440.0
    
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    input_audio = 0.1 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    
    # Run through pipeline with passthrough_test=True
    output_audio, taps = process_audio(
        input_audio,
        sr=sr,
        passthrough_test=True,
    )
    
    # Align lengths
    min_length = min(len(input_audio), len(output_audio))
    input_aligned = input_audio[:min_length]
    output_aligned = output_audio[:min_length]
    
    # Compute delta
    delta = output_aligned - input_aligned
    
    # Compute RMS of delta in dB
    rms_delta_db = _compute_rms_db(delta)
    
    # CI-safe threshold
    threshold_db = -80.0
    
    assert rms_delta_db < threshold_db, \
        f"OLA reconstruction failed for simple sine: RMS delta = {rms_delta_db:.2f} dB " \
        f"(threshold: {threshold_db} dB)"

