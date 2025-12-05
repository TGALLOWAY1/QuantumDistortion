"""
Frequency-domain creative FX for high-band STFT processing.

This module provides spectral effects that operate on magnitude and phase
spectra, including bitcrush, phase dispersal, and bin scrambling. These
effects are designed to be applied to the high-band STFT in the quantum
distortion pipeline.

Design Philosophy:
------------------
These FX operate on high-band STFT only (not low band). This design allows
the low band to stay tight and clean while the high band gets "quantum"
texture. This is particularly useful for:

- Dubstep / Neuro bass: Keep sub frequencies clean and punchy while adding
  grit and movement to the mid/high frequencies
- Drum & Bass: Maintain tight low-end while adding creative texture to
  high-frequency content
- General bass processing: Preserve low-end clarity while adding spectral
  interest to the upper frequencies

The multiband architecture ensures that bass frequencies remain untouched,
preventing muddiness and phase issues in the critical low-end range.
"""

from __future__ import annotations

import numpy as np


# Preset definitions for common bass/FX use-cases.
# These are intentionally simple dictionaries so they can
# be surfaced in a UI or config later.
SPECTRAL_FX_PRESETS: dict[str, dict] = {
    "sub_safe_glue": {
        "mode": "bitcrush",
        "distortion_strength": 0.35,
        "params": {
            "method": "log",
            "step_db": 2.0,
            "threshold": 0.0,
        },
        "description": "Sub-safe subtle log-domain bitcrush for main bass growls.",
    },
    "digital_growl": {
        "mode": "bitcrush",
        "distortion_strength": 0.55,
        "params": {
            "method": "log",
            "step_db": 3.0,
            # threshold is computed dynamically from mag.max() if omitted
        },
        "description": "More obvious digital grit for aggressive growls.",
    },
    "hard_crush_fx": {
        "mode": "bitcrush",
        "distortion_strength": 0.8,
        "params": {
            "method": "uniform",
            "step": 0.07,
            # threshold often pushes small bins to zero
        },
        "description": "Heavy bitcrush for stabs and FX, not main bass.",
    },
    "gentle_movement": {
        "mode": "phase_dispersal",
        "distortion_strength": 0.25,
        "params": {
            "randomized": False,
        },
        "description": "Small phase rotation for subtle shimmer.",
    },
    "laser_zap": {
        "mode": "phase_dispersal",
        "distortion_strength": 0.6,
        "params": {
            "randomized": True,
        },
        "description": "Laser/zap phase dispersal for neuro-style tops.",
    },
    "phase_chaos_fx": {
        "mode": "phase_dispersal",
        "distortion_strength": 0.9,
        "params": {
            "randomized": True,
        },
        "description": "Extreme phase chaos for risers and FX beds.",
    },
    "stereo_smear": {
        "mode": "bin_scramble",
        "distortion_strength": 0.3,
        "params": {
            "mode": "swap",
        },
        "description": "Subtle local swaps for smeared high-end texture.",
    },
    "grainy_top": {
        "mode": "bin_scramble",
        "distortion_strength": 0.55,
        "params": {
            "mode": "random_pick",
        },
        "description": "Noticeable granular smear on the high band.",
    },
    "granular_shred": {
        "mode": "bin_scramble",
        "distortion_strength": 0.85,
        "params": {
            "mode": "random_pick",
        },
        "description": "Heavily shredded high band for FX-only usage.",
    },
}


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
    mag = np.asarray(mag, dtype=float).copy()
    phase = np.asarray(phase, dtype=float)

    # Early return if amount <= 0 and not randomized
    if amount <= 0 and not randomized:
        return mag, phase

    # Compute boolean mask
    if thresh is not None and thresh > 0.0:
        mask = mag > thresh
    else:
        mask = np.ones_like(mag, dtype=bool)

    # Base rotation
    phase_out = np.array(phase, copy=True)
    rotation = amount * (mag / (np.max(mag) + 1e-12))
    rotation = np.where(mask, rotation, 0.0)

    # Add random jitter if requested
    if randomized:
        random_jitter = (np.random.rand(*phase.shape) * 2.0 - 1.0) * rand_amt
        random_jitter = np.where(mask, random_jitter, 0.0)
        rotation = rotation + random_jitter

    # Apply rotation
    phase_out = phase_out + rotation
    # Wrap to [-pi, pi]
    phase_out = (phase_out + np.pi) % (2.0 * np.pi) - np.pi

    return mag, phase_out


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
    mag = np.asarray(mag, dtype=float)
    phase = np.asarray(phase, dtype=float)

    # Validate and clamp window to odd >= 3
    if window < 2 or window % 2 == 0:
        window = max(3, window if window % 2 == 1 else window + 1)

    if mode == "random_pick":
        mag_out = np.zeros_like(mag)
        half = window // 2
        for i in range(mag.shape[0]):
            start = max(0, i - half)
            end = min(mag.shape[0], i + half + 1)
            # Choose random index in [start, end)
            if end > start:
                idx = np.random.randint(start, end)
                mag_out[i] = mag[idx]
            else:
                mag_out[i] = mag[i]
    elif mode == "swap":
        mag_out = mag.copy()
        for i in range(mag.size - 1):
            if np.random.rand() < 0.25:
                # Swap mag_out[i] and mag_out[i+1]
                mag_out[i], mag_out[i + 1] = mag_out[i + 1], mag_out[i]
    else:
        # Unknown mode, return unchanged
        mag_out = mag.copy()

    # Maintain gross energy
    scale = np.sum(mag) / (np.sum(mag_out) + 1e-12)
    mag_out = mag_out * scale

    return mag_out, phase

