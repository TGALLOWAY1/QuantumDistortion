from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, sosfilt


def design_linkwitz_riley_sos(
    sr: int,
    crossover_hz: float,
    order_per_side: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Linkwitz-Riley crossover as cascaded Butterworth filters.
    
    A Linkwitz-Riley crossover is created by cascading two Butterworth filters
    of the same order. This results in a 4th-order crossover (when order_per_side=2)
    that provides perfect reconstruction: low_band + high_band = original signal.
    
    Parameters
    ----------
    sr : int
        Sample rate in Hz.
    crossover_hz : float
        Crossover frequency in Hz.
    order_per_side : int, optional
        Order of each Butterworth filter side (default: 2).
        Total crossover order will be 2 * order_per_side.
    
    Returns
    -------
    sos_low : np.ndarray
        Second-order sections (SOS) for the low-pass filter.
    sos_high : np.ndarray
        Second-order sections (SOS) for the high-pass filter.
    """
    # Normalize frequency to Nyquist (0-1 range)
    nyquist = sr / 2.0
    normalized_freq = crossover_hz / nyquist
    
    if normalized_freq <= 0.0 or normalized_freq >= 1.0:
        raise ValueError(
            f"Crossover frequency {crossover_hz} Hz must be between 0 and Nyquist ({nyquist} Hz)"
        )
    
    # Design Butterworth low-pass filter
    sos_lp_single = butter(
        N=order_per_side,
        Wn=normalized_freq,
        btype="low",
        output="sos",
    )
    
    # Design Butterworth high-pass filter
    sos_hp_single = butter(
        N=order_per_side,
        Wn=normalized_freq,
        btype="high",
        output="sos",
    )
    
    # Cascade two identical filters to create Linkwitz-Riley
    # Concatenate SOS arrays along the first axis
    sos_low = np.concatenate([sos_lp_single, sos_lp_single], axis=0)
    sos_high = np.concatenate([sos_hp_single, sos_hp_single], axis=0)
    
    return sos_low, sos_high


def linkwitz_riley_split(
    audio: np.ndarray,
    sr: int,
    crossover_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split audio into (low_band, high_band) using a Linkwitz-Riley crossover.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio signal. Shape can be (num_samples,) for mono or
        (num_samples, num_channels) for multi-channel.
    sr : int
        Sample rate in Hz.
    crossover_hz : float
        Crossover frequency in Hz.
    
    Returns
    -------
    low_band : np.ndarray
        Low-frequency band, same shape as input.
    high_band : np.ndarray
        High-frequency band, same shape as input.
    """
    audio = np.asarray(audio, dtype=np.float32)
    
    # Design the crossover filters
    sos_low, sos_high = design_linkwitz_riley_sos(sr, crossover_hz)
    
    # Handle mono vs multi-channel
    if audio.ndim == 1:
        # Mono: apply filters directly
        low_band = sosfilt(sos_low, audio).astype(np.float32)
        high_band = sosfilt(sos_high, audio).astype(np.float32)
    elif audio.ndim == 2:
        # Multi-channel: apply filters to each channel
        num_channels = audio.shape[1]
        low_band = np.zeros_like(audio, dtype=np.float32)
        high_band = np.zeros_like(audio, dtype=np.float32)
        
        for ch in range(num_channels):
            low_band[:, ch] = sosfilt(sos_low, audio[:, ch]).astype(np.float32)
            high_band[:, ch] = sosfilt(sos_high, audio[:, ch]).astype(np.float32)
    else:
        raise ValueError(f"Audio must be 1D (mono) or 2D (multi-channel), got shape {audio.shape}")
    
    return low_band, high_band

