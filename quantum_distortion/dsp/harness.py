"""
Convenience harness for processing audio files through the Quantum Distortion pipeline.

This module provides a simple file-to-file processing interface that wraps
process_audio, making it easy to use in automated tests, regression tests, and scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from quantum_distortion.config import (
    DEFAULT_BIN_SMOOTHING,
    DEFAULT_DRY_WET,
    DEFAULT_KEY,
    DEFAULT_LIMITER_CEILING_DB,
    DEFAULT_LIMITER_ON,
    DEFAULT_SCALE,
    DEFAULT_SMEAR,
    DEFAULT_SNAP_STRENGTH,
    DEFAULT_DISTORTION_MODE,
    DEFAULT_SAMPLE_RATE,
)
from quantum_distortion.io.audio_io import load_audio, save_audio
from quantum_distortion.dsp.pipeline import process_audio


def process_file_to_file(
    infile: Path,
    outfile: Path,
    preset: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience wrapper that loads audio, processes it, and saves the result.
    
    This function:
    - Loads audio from infile
    - Applies process_audio (using either default params or a preset)
    - Writes processed audio to outfile
    - Ignores tap buffers (they are discarded)
    
    Args:
        infile: Path to input audio file
        outfile: Path to output audio file (will be created, parent dirs created if needed)
        preset: Optional preset name to use. If provided, loads preset configuration
                from quantum_distortion.presets. If None, uses default parameters.
        extra_params: Optional dict of additional parameters to override defaults or preset.
                     These are merged with preset/default params, with extra_params taking precedence.
    
    Raises:
        FileNotFoundError: If infile doesn't exist
        KeyError: If preset name is not found
        ValueError: If preset is invalid
    """
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")
    
    # Load audio
    audio, sr = load_audio(infile)
    
    # Convert to mono float32 if needed (process_audio handles this, but we do it here
    # to match the pattern from render_preset.py)
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32)
    
    # Build kwargs for process_audio
    if preset is not None:
        # Load preset configuration
        from quantum_distortion.presets import get_preset
        
        preset_config = get_preset(preset)
        
        # Extract preset parameters
        kwargs = {
            "audio": x,
            "sr": sr,
            "key": str(preset_config["key"]),
            "scale": str(preset_config["scale"]),
            "snap_strength": float(preset_config["snap_strength"]),
            "smear": float(preset_config["smear"]),
            "bin_smoothing": bool(preset_config["bin_smoothing"]),
            "pre_quant": bool(preset_config["pre_quant"]),
            "post_quant": bool(preset_config["post_quant"]),
            "distortion_mode": str(preset_config["distortion_mode"]),
            "distortion_params": dict(preset_config["distortion_params"]),
            "limiter_on": bool(preset_config["limiter_on"]),
            "limiter_ceiling_db": float(preset_config["limiter_ceiling_db"]),
            "dry_wet": float(preset_config["dry_wet"]),
        }
    else:
        # Use default parameters
        kwargs = {
            "audio": x,
            "sr": sr,
            "key": DEFAULT_KEY,
            "scale": DEFAULT_SCALE,
            "snap_strength": DEFAULT_SNAP_STRENGTH,
            "smear": DEFAULT_SMEAR,
            "bin_smoothing": DEFAULT_BIN_SMOOTHING,
            "pre_quant": True,
            "post_quant": True,
            "distortion_mode": DEFAULT_DISTORTION_MODE,
            "distortion_params": {},
            "limiter_on": DEFAULT_LIMITER_ON,
            "limiter_ceiling_db": DEFAULT_LIMITER_CEILING_DB,
            "dry_wet": DEFAULT_DRY_WET,
        }
    
    # Apply extra_params overrides (if provided)
    if extra_params is not None:
        # Remove 'audio' and 'sr' from extra_params if present (these are positional)
        extra_params = dict(extra_params)
        extra_params.pop("audio", None)
        extra_params.pop("sr", None)
        kwargs.update(extra_params)
    
    # Process audio (discard tap buffers)
    processed, _taps = process_audio(**kwargs)
    
    # Save processed audio
    save_audio(outfile, processed, sr)

