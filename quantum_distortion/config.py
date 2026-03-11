from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------
ScaleName = Literal["major", "minor", "pentatonic", "dorian", "mixolydian", "harmonic_minor"]
TapSource = Literal["input", "pre_quant", "post_dist", "output"]


# ---------------------------------------------------------------------------
# Shared audio utility
# ---------------------------------------------------------------------------
def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono float32. Averages channels if stereo."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32)
    return x


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 48000

DEFAULT_KEY = "D"
DEFAULT_SCALE = "minor"

DEFAULT_SNAP_STRENGTH = 1.0
DEFAULT_SMEAR = 0.1
DEFAULT_BIN_SMOOTHING = True

DEFAULT_DISTORTION_MODE = "wavefold"

DEFAULT_LIMITER_ON = True
DEFAULT_LIMITER_CEILING_DB = -1.0
DEFAULT_DRY_WET = 1.0

# Preview render mode: limits processing to first N seconds for faster iteration
# Set DSP_PREVIEW_MODE=1 environment variable to enable, or pass preview_enabled=True
PREVIEW_ENABLED_DEFAULT = False  # Default to full render for production
PREVIEW_MAX_SECONDS = 10.0  # Process only first 10 seconds in preview mode
