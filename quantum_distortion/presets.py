from __future__ import annotations


from pathlib import Path

from typing import Dict, Any, List, Union


import json


_PRESETS_CACHE: Union[Dict[str, Dict[str, Any]], None] = None


def _load_presets_raw() -> Dict[str, Dict[str, Any]]:
    """
    Load presets from presets/quantum_distortion_presets.json (once per process).
    """
    global _PRESETS_CACHE
    if _PRESETS_CACHE is not None:
        return _PRESETS_CACHE

    # Resolve path relative to this file: quantum_distortion/presets.py
    here = Path(__file__).resolve().parent
    presets_path = here.parent / "presets" / "quantum_distortion_presets.json"

    if not presets_path.exists():
        raise FileNotFoundError(f"Presets file not found: {presets_path}")

    with presets_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Presets JSON must be a dict of name -> preset objects")

    # Basic validation: ensure required keys exist in each preset
    required_top_level = {
        "key",
        "scale",
        "snap_strength",
        "smear",
        "bin_smoothing",
        "pre_quant",
        "post_quant",
        "distortion_mode",
        "distortion_params",
        "limiter_on",
        "limiter_ceiling_db",
        "dry_wet",
    }

    for name, preset in data.items():
        if not isinstance(preset, dict):
            raise ValueError(f"Preset '{name}' must be an object/dict")
        missing = required_top_level - set(preset.keys())
        if missing:
            raise ValueError(f"Preset '{name}' is missing required keys: {sorted(missing)}")

    _PRESETS_CACHE = data
    return data


def list_presets() -> List[str]:
    """
    Return list of available preset names.
    """
    data = _load_presets_raw()
    return sorted(data.keys())


def get_preset(name: str) -> Dict[str, Any]:
    """
    Retrieve a preset configuration by name.

    Raises KeyError if not found.
    """
    data = _load_presets_raw()
    if name not in data:
        raise KeyError(f"Preset not found: {name}")
    return dict(data[name])

