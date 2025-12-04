"""
Audio test utilities for Quantum Distortion.

Provides reusable functions for null tests and audio comparison,
to quantify how close processed audio is to the input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np

from quantum_distortion.io.audio_io import load_audio as _load_audio


def load_audio(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """
    Load audio file using the project's existing audio IO helper.
    
    This is a convenience wrapper around quantum_distortion.io.audio_io.load_audio
    for use in test utilities.
    
    Args:
        path: Path to audio file
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    return _load_audio(path)


def null_test(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the RMS (in dB) of the residual signal (a - b).
    
    This measures how different two audio signals are. A more negative value
    indicates the signals are more similar (lower residual energy).
    
    Args:
        a: First audio signal (1D or 2D array)
        b: Second audio signal (1D or 2D array)
        
    Returns:
        Residual RMS in dBFS (or dB relative), where more negative is better.
        Typical values:
        - Identical signals: < -80 dB
        - Very similar: -60 to -80 dB
        - Noticeably different: > -40 dB
        
    Notes:
        - If signals have different lengths, both are truncated to the minimum length.
        - For multi-channel signals (2D arrays), the residual is computed channel-wise
          and then averaged across channels.
        - Uses a small epsilon to avoid log-of-zero issues.
    """
    # Ensure arrays are numpy arrays
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    # Handle length mismatches: truncate both to minimum length
    min_len = min(a.shape[0], b.shape[0])
    a = a[:min_len]
    b = b[:min_len]
    
    # Handle multi-channel signals
    if a.ndim == 2 and b.ndim == 2:
        # Both are multi-channel: compute residual per channel, then average
        if a.shape[1] != b.shape[1]:
            # Different number of channels: use minimum
            min_channels = min(a.shape[1], b.shape[1])
            a = a[:, :min_channels]
            b = b[:, :min_channels]
        residual = a - b
        # Compute RMS per channel, then average
        rms_per_channel = np.sqrt(np.mean(residual ** 2, axis=0))
        rms = np.mean(rms_per_channel)
    elif a.ndim == 2 or b.ndim == 2:
        # One is multi-channel, one is mono: flatten multi-channel to mono
        if a.ndim == 2:
            a = np.mean(a, axis=1)
        if b.ndim == 2:
            b = np.mean(b, axis=1)
        residual = a - b
        rms = np.sqrt(np.mean(residual ** 2))
    else:
        # Both are mono (1D)
        residual = a - b
        rms = np.sqrt(np.mean(residual ** 2))
    
    # Convert to dB with epsilon to avoid log-of-zero
    epsilon = 1e-10
    rms_db = 20.0 * np.log10(max(rms, epsilon))
    
    return float(rms_db)


def print_null_test(label: str, a: np.ndarray, b: np.ndarray) -> None:
    """
    Print null test result in a nicely formatted way.
    
    Args:
        label: Label to print before the result
        a: First audio signal
        b: Second audio signal
    """
    residual_db = null_test(a, b)
    print(f"{label}: {residual_db:.2f} dB")

