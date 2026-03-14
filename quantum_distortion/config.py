from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------
ScaleName = Literal["major", "minor", "pentatonic", "dorian", "mixolydian", "harmonic_minor"]
TapSource = Literal["input", "pre_quant", "post_dist", "output"]
QuantizeMode = Literal["autotune_v1", "spectral_bins"]
SubSourceName = Literal["root", "manual", "scale_degree"]


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
DEFAULT_QUANTIZE_MODE: QuantizeMode = "autotune_v1"
DEFAULT_SUB_ENABLED = True
DEFAULT_SUB_SOURCE: SubSourceName = "root"
DEFAULT_SUB_NOTE = "C"
DEFAULT_SUB_SCALE_DEGREE = 0
DEFAULT_SUB_OCTAVE = 2
DEFAULT_SUB_LEVEL = 0.35
DEFAULT_SUB_CUT_HZ = 110.0
DEFAULT_AIR_CUT_HZ = 5000.0
DEFAULT_AIR_MIX = 1.0

# Preview render mode: limits processing to first N seconds for faster iteration
# Set DSP_PREVIEW_MODE=1 environment variable to enable, or pass preview_enabled=True
PREVIEW_ENABLED_DEFAULT = False  # Default to full render for production
PREVIEW_MAX_SECONDS = 10.0  # Process only first 10 seconds in preview mode


# ---------------------------------------------------------------------------
# Pipeline configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Consolidated configuration for the audio processing pipeline.

    Replaces the 20+ keyword arguments of process_audio() with a single
    structured object. All fields have sensible defaults that match the
    previous keyword-argument defaults.
    """
    # Musical key & scale
    key: str = DEFAULT_KEY
    scale: str = DEFAULT_SCALE

    # Quantization
    quantize_mode: QuantizeMode = DEFAULT_QUANTIZE_MODE
    snap_strength: float = DEFAULT_SNAP_STRENGTH
    smear: float = DEFAULT_SMEAR
    bin_smoothing: bool = DEFAULT_BIN_SMOOTHING
    pre_quant: bool = True
    post_quant: bool = True

    # Autotune V1 routing
    sub_enabled: bool = DEFAULT_SUB_ENABLED
    sub_source: SubSourceName = DEFAULT_SUB_SOURCE
    sub_note: str = DEFAULT_SUB_NOTE
    sub_scale_degree: int = DEFAULT_SUB_SCALE_DEGREE
    sub_octave: int = DEFAULT_SUB_OCTAVE
    sub_level: float = DEFAULT_SUB_LEVEL
    sub_cut_hz: float = DEFAULT_SUB_CUT_HZ
    air_cut_hz: float = DEFAULT_AIR_CUT_HZ
    air_mix: float = DEFAULT_AIR_MIX

    # Distortion
    distortion_mode: str = DEFAULT_DISTORTION_MODE
    distortion_params: Dict[str, Any] = field(default_factory=dict)

    # Limiter
    limiter_on: bool = DEFAULT_LIMITER_ON
    limiter_ceiling_db: float = DEFAULT_LIMITER_CEILING_DB

    # Mix
    dry_wet: float = DEFAULT_DRY_WET

    # Preview
    preview_enabled: Optional[bool] = None

    # Multiband
    use_multiband: bool = False
    crossover_hz: float = 300.0
    lowband_drive: float = 1.0

    # Testing
    passthrough_test: bool = False

    # Spectral FX
    spectral_fx_mode: Optional[str] = None
    spectral_fx_strength: float = 0.0
    spectral_fx_params: Dict[str, Any] = field(default_factory=dict)

    # M12 creative features
    spectral_freeze: bool = False
    formant_shift: float = 0.0
    harmonic_lock_hz: float = 0.0

    # Output
    delta_listen: bool = False
    mono_strength: float = 1.0
    output_trim_db: float = 0.0

    @classmethod
    def from_preset(cls, preset_name: str) -> "PipelineConfig":
        """Create a PipelineConfig from a named preset."""
        from quantum_distortion.presets import get_preset
        p = get_preset(preset_name)
        return cls(
            key=str(p["key"]),
            scale=str(p["scale"]),
            quantize_mode=DEFAULT_QUANTIZE_MODE,
            snap_strength=float(p["snap_strength"]),
            smear=float(p["smear"]),
            bin_smoothing=bool(p["bin_smoothing"]),
            pre_quant=bool(p["pre_quant"]),
            post_quant=bool(p["post_quant"]),
            distortion_mode=str(p["distortion_mode"]),
            distortion_params=dict(p["distortion_params"]),
            limiter_on=bool(p["limiter_on"]),
            limiter_ceiling_db=float(p["limiter_ceiling_db"]),
            dry_wet=float(p["dry_wet"]),
        )
