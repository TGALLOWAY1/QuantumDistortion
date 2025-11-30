from __future__ import annotations


from pathlib import Path
import argparse


import numpy as np


from quantum_distortion.io.audio_io import load_audio, save_audio
from quantum_distortion.dsp.pipeline import process_audio
from quantum_distortion.presets import list_presets, get_preset


def main() -> None:
    parser = argparse.ArgumentParser(description="Render audio using a named Quantum Distortion preset.")
    parser.add_argument("--infile", "-i", help="Input audio file (wav/aif/aiff)")
    parser.add_argument("--outfile", "-o", help="Output audio file (wav)")
    parser.add_argument("--preset", "-p", help="Preset name (see --list-presets)")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        names = list_presets()
        print("Available presets:")
        for name in names:
            print(f"  - {name}")
        return

    if not args.infile:
        parser.error("--infile/-i is required when not using --list-presets")
    if not args.outfile:
        parser.error("--outfile/-o is required when not using --list-presets")
    if not args.preset:
        parser.error("--preset/-p is required when not using --list-presets")

    infile = Path(args.infile)
    if not infile.exists():
        raise SystemExit(f"Input file not found: {infile}")

    audio, sr = load_audio(infile)

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32)

    preset = get_preset(args.preset)

    key = str(preset["key"])
    scale = str(preset["scale"])
    snap_strength = float(preset["snap_strength"])
    smear = float(preset["smear"])
    bin_smoothing = bool(preset["bin_smoothing"])
    pre_quant = bool(preset["pre_quant"])
    post_quant = bool(preset["post_quant"])
    distortion_mode = str(preset["distortion_mode"])
    distortion_params = dict(preset["distortion_params"])
    limiter_on = bool(preset["limiter_on"])
    limiter_ceiling_db = float(preset["limiter_ceiling_db"])
    dry_wet = float(preset["dry_wet"])

    print(f"Using preset: {args.preset}")
    print(f"  Key/Scale: {key} {scale}")
    print(f"  Distortion: {distortion_mode} {distortion_params}")
    print(f"  Snap: {snap_strength}, Smear: {smear}, Dry/Wet: {dry_wet}")

    processed, _taps = process_audio(
        audio=x,
        sr=sr,
        key=key,
        scale=scale,
        snap_strength=snap_strength,
        smear=smear,
        bin_smoothing=bin_smoothing,
        pre_quant=pre_quant,
        post_quant=post_quant,
        distortion_mode=distortion_mode,
        distortion_params=distortion_params,
        limiter_on=limiter_on,
        limiter_ceiling_db=limiter_ceiling_db,
        dry_wet=dry_wet,
    )

    save_audio(args.outfile, processed, sr)
    print(f"Saved processed file → {args.outfile}")


if __name__ == "__main__":
    main()

