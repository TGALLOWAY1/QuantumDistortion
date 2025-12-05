"""
Frequency-domain creative FX for high-band STFT processing.

This module provides spectral effects that operate on magnitude and phase
spectra, including bitcrush, phase dispersal, and bin scrambling. These
effects are designed to be applied to the high-band STFT in the quantum
distortion pipeline.
"""

from __future__ import annotations

import numpy as np


def bitcrush(
    mag: np.ndarray,
    phase: np.ndarray,
    *,
    method: str = "uniform",
    step: float = 0.02,
    step_db: float = 1.5,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply spectral bitcrush / decimation in the magnitude domain.

    Parameters
    ----------
    mag : np.ndarray
        Magnitude spectrum for a single frame (non-negative).
    phase : np.ndarray
        Phase spectrum for a single frame, in radians.
    method : {"uniform", "log"}
        Quantization mode:
        - "uniform": linear step in magnitude
        - "log": step in dB space
    step : float
        Step size for uniform magnitude quantization (0..1-ish).
    step_db : float
        Step size in dB for log-space quantization.
    threshold : float | None
        Optional magnitude threshold below which bins may be zeroed.

    Returns
    -------
    mag_out, phase_out : np.ndarray, np.ndarray
        Processed magnitude and phase. Phase is typically unchanged.
    """
    eps = 1e-12
    mag = np.asarray(mag, dtype=float).copy()
    phase = np.asarray(phase, dtype=float)

    # Ensure non-negative magnitudes
    mag = np.clip(mag, 0.0, None)

    if method == "uniform":
        if step <= 0:
            mag_out = mag.copy()
        else:
            mag_out = np.round(mag / step) * step
            mag_out = np.clip(mag_out, 0.0, None)
    elif method == "log":
        if step_db <= 0:
            mag_out = mag.copy()
        else:
            mag_db = 20.0 * np.log10(np.clip(mag, eps, None))
            mag_db_q = np.round(mag_db / step_db) * step_db
            mag_out = 10.0 ** (mag_db_q / 20.0)
    else:
        # Unknown method, return unchanged
        mag_out = mag.copy()

    # Apply threshold if specified
    if threshold is not None and threshold > 0.0:
        mag_out = np.where(mag_out < threshold, 0.0, mag_out)

    return mag_out, phase


def phase_dispersal(
    mag: np.ndarray,
    phase: np.ndarray,
    *,
    thresh: float = 0.01,
    amount: float = 0.5,
    randomized: bool = False,
    rand_amt: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate phase for louder bins to create 'laser' / 'zap' textures.

    Parameters
    ----------
    mag : np.ndarray
        Magnitude spectrum for a single frame.
    phase : np.ndarray
        Phase spectrum in radians.
    thresh : float
        Magnitude threshold above which bins are eligible for rotation.
    amount : float
        Base rotation amount in radians to apply to selected bins.
    randomized : bool
        If True, add random jitter to the rotation.
    rand_amt : float
        Maximum random jitter (in radians) when randomized is True.

    Returns
    -------
    mag_out, phase_out : np.ndarray, np.ndarray
    """
    return mag, phase


def bin_scramble(
    mag: np.ndarray,
    phase: np.ndarray,
    *,
    window: int = 5,
    mode: str = "random_pick",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locally scramble magnitude bins to blur texture.

    Parameters
    ----------
    mag : np.ndarray
        Magnitude spectrum for a single frame.
    phase : np.ndarray
        Phase spectrum in radians (typically unchanged).
    window : int
        Odd window size for local neighborhoods (e.g. 3, 5, 7).
    mode : {"random_pick", "swap"}
        Scramble strategy:
        - "random_pick": new mag[i] is random neighbor in window
        - "swap": occasionally swap neighboring bins

    Returns
    -------
    mag_out, phase_out : np.ndarray, np.ndarray
    """
    return mag, phase

