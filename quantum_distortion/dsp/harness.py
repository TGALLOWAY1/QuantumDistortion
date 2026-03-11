"""
Convenience harness for processing audio files through the Quantum Distortion pipeline.

This module provides a simple file-to-file processing interface that wraps
process_audio, making it easy to use in automated tests, regression tests, and scripts.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from quantum_distortion.config import (
    PipelineConfig,
    ensure_mono_float32,
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

    Args:
        infile: Path to input audio file
        outfile: Path to output audio file
        preset: Optional preset name to use. If None, uses defaults.
        extra_params: Optional dict of parameter overrides.

    Raises:
        FileNotFoundError: If infile doesn't exist
        KeyError: If preset name is not found
    """
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    audio, sr = load_audio(infile)
    x = ensure_mono_float32(audio)

    # Build PipelineConfig from preset or defaults
    if preset is not None:
        cfg = PipelineConfig.from_preset(preset)
    else:
        cfg = PipelineConfig()

    # Apply extra_params overrides if provided
    if extra_params is not None:
        cfg_dict = asdict(cfg)
        for k, v in extra_params.items():
            if k in cfg_dict:
                setattr(cfg, k, v)

    processed, _taps = process_audio(x, sr=sr, pipeline_config=cfg)
    save_audio(outfile, processed, sr)
