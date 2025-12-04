from __future__ import annotations

import numpy as np


def soft_tube(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """
    Simple tube-like saturation. Assumes audio in [-1, 1].
    
    Uses a tanh waveshaper with configurable drive. Higher drive values
    increase saturation and harmonic content.
    
    Parameters
    ----------
    x : np.ndarray
        Input audio signal. Can be mono (1D) or multi-channel (2D).
        Assumes values are in the range [-1, 1].
    drive : float, optional
        Input gain. >1.0 increases distortion. Defaults to 1.0.
    
    Returns
    -------
    np.ndarray
        Saturated audio, same shape as input.
    """
    x = np.asarray(x, dtype=np.float32)
    original_shape = x.shape
    
    # Handle mono vs multi-channel
    if x.ndim == 1:
        # Mono: process directly
        x_flat = x
        is_mono = True
    elif x.ndim == 2:
        # Multi-channel: flatten for processing, will reshape later
        n_samples, n_channels = x.shape
        x_flat = x.flatten()
        is_mono = False
    else:
        raise ValueError(f"Audio must be 1D (mono) or 2D (multi-channel), got shape {x.shape}")
    
    # Input gain
    x_flat = x_flat * max(float(drive), 0.0)
    
    # Apply tanh saturation
    # Using a fixed shape parameter (a=3) for consistent saturation curve
    # This gives a good balance between subtle and aggressive saturation
    a = 3.0
    y_flat = np.tanh(a * x_flat)
    
    # Normalize to keep output roughly in [-1, 1]
    # Divide by tanh(a) to normalize the maximum slope
    y_flat = y_flat / np.tanh(a) if a != 0.0 else y_flat
    
    # Reshape if needed
    if is_mono:
        y = y_flat.astype(np.float32)
    else:
        y = y_flat.reshape(n_samples, n_channels).astype(np.float32)
    
    return y


def make_mono_lowband(x: np.ndarray) -> np.ndarray:
    """
    Force low band to mono by averaging channels:
    - If mono, returns x unchanged.
    - If stereo/2D: averages channels and duplicates back to both channels.
    
    This ensures the low band is mono (important for bass frequencies) while
    preserving the output shape for recombination with the high band.
    
    Parameters
    ----------
    x : np.ndarray
        Input audio signal. Can be mono (1D) or multi-channel (2D).
    
    Returns
    -------
    np.ndarray
        Mono audio. If input was 2D, returns 2D with identical channels.
        If input was 1D, returns 1D unchanged.
    """
    x = np.asarray(x, dtype=np.float32)
    
    if x.ndim == 1:
        # Already mono, return unchanged
        return x
    elif x.ndim == 2:
        # Multi-channel: average channels
        mono = np.mean(x, axis=1, keepdims=True)
        # Duplicate to all channels to preserve shape
        n_channels = x.shape[1]
        mono_duplicated = np.repeat(mono, n_channels, axis=1)
        return mono_duplicated.astype(np.float32)
    else:
        raise ValueError(f"Audio must be 1D (mono) or 2D (multi-channel), got shape {x.shape}")

