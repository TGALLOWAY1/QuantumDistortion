from __future__ import annotations


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


import numpy as np


DistortionMode = Literal["wavefold", "tube"]


def _ensure_mono_float(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1:
        raise ValueError("Distortion currently expects mono (1D) audio")
    return audio


def wavefold(
    audio: np.ndarray,
    fold_amount: float = 1.0,
    bias: float = 0.0,
    threshold: float = 1.0,
) -> np.ndarray:
    """
    Simple wavefolding distortion.

    Steps:
    - Apply bias (DC offset).
    - Apply fold_amount as input gain.
    - Fold around +/- threshold using mirrored clipping (single fold pass).

    This is not a mathematically perfect multi-fold circuit emulator,
    but is sufficient for MVP to generate rich harmonics.
    """
    x = _ensure_mono_float(audio)

    # Apply bias + gain
    x = (x + bias) * fold_amount

    if threshold <= 0.0:
        threshold = 1.0

    # Mirrored clipping fold:
    # First handle positive side, then negative
    y = x.copy()

    # Positive side
    over_pos = y > threshold
    y[over_pos] = 2.0 * threshold - y[over_pos]

    # Negative side
    over_neg = y < -threshold
    y[over_neg] = -2.0 * threshold - y[over_neg]

    # Optional post-normalization: try to keep in [-threshold, threshold]
    y = np.clip(y, -threshold, threshold)

    return y.astype(np.float32)


def soft_tube(
    audio: np.ndarray,
    drive: float = 1.0,
    warmth: float = 0.5,
) -> np.ndarray:
    """
    Soft-tube style saturation using a tanh waveshaper.

    Parameters
    ----------
    drive : float
        Input gain. >1.0 increases distortion.
    warmth : float (0..1)
        Controls curve steepness; higher = more aggressive.
    """
    x = _ensure_mono_float(audio)

    # Input gain
    x = x * max(drive, 0.0)

    # Map warmth (0..1) → shape parameter a (1..5)
    warmth = float(np.clip(warmth, 0.0, 1.0))
    a = 1.0 + 4.0 * warmth

    # Shape with tanh, normalize so that max slope is ~1 around zero
    y = np.tanh(a * x)
    # Normalize to keep output roughly in [-1,1]
    y /= np.tanh(a) if a != 0.0 else 1.0

    return y.astype(np.float32)


def apply_distortion(
    audio: np.ndarray,
    mode: DistortionMode,
    fold_amount: float = 1.0,
    bias: float = 0.0,
    drive: float = 1.0,
    warmth: float = 0.5,
) -> np.ndarray:
    """
    Convenience wrapper that chooses distortion mode.

    Parameters
    ----------
    mode : "wavefold" or "tube"
    Other params are routed to the corresponding function.
    """
    if mode == "wavefold":
        return wavefold(audio, fold_amount=fold_amount, bias=bias, threshold=1.0)
    elif mode == "tube":
        return soft_tube(audio, drive=drive, warmth=warmth)
    else:
        raise ValueError(f"Unsupported distortion mode: {mode}")

