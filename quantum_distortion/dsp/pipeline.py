import numpy as np

from typing import Any, Dict, Tuple, Union


from quantum_distortion.config import (
    DEFAULT_KEY,
    DEFAULT_SCALE,
    DEFAULT_SNAP_STRENGTH,
    DEFAULT_SMEAR,
    DEFAULT_BIN_SMOOTHING,
    DEFAULT_DISTORTION_MODE,
    DEFAULT_LIMITER_ON,
    DEFAULT_LIMITER_CEILING_DB,
    DEFAULT_DRY_WET,
    DEFAULT_SAMPLE_RATE,
)


def process_audio(
    audio: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    key: str = DEFAULT_KEY,
    scale: str = DEFAULT_SCALE,
    snap_strength: float = DEFAULT_SNAP_STRENGTH,
    smear: float = DEFAULT_SMEAR,
    bin_smoothing: bool = DEFAULT_BIN_SMOOTHING,
    pre_quant: bool = True,
    post_quant: bool = True,
    distortion_mode: str = DEFAULT_DISTORTION_MODE,
    distortion_params: Union[Dict[str, Any], None] = None,
    limiter_on: bool = DEFAULT_LIMITER_ON,
    limiter_ceiling_db: float = DEFAULT_LIMITER_CEILING_DB,
    dry_wet: float = DEFAULT_DRY_WET,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Main offline processing entry point.

    Milestone 1:
    - API exposes quantizer-related parameters but does not yet apply them.
    - Audio is still passed through unchanged.
    """
    if distortion_params is None:
        distortion_params = {}

    if audio.ndim != 1:
        # For now, enforce mono for MVP core DSP; stereo support can be added later.
        raise ValueError("process_audio currently expects mono audio (1D array)")

    taps = {
        "input": audio.copy(),
        "pre_quant": audio.copy(),
        "post_dist": audio.copy(),
        "output": audio.copy(),
    }
    return audio.copy(), taps
