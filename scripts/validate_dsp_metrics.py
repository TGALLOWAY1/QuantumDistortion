from __future__ import annotations


from pathlib import Path
import argparse


import numpy as np


from quantum_distortion.io.audio_io import load_audio
from quantum_distortion.dsp.pipeline import process_audio
from quantum_distortion.dsp.analyses import avg_cents_offset_from_scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DSP behavior vs. scale alignment metrics.")
    parser.add_argument("--infile", "-i", required=True, help="Input audio file (wav/aif/aiff)")
    parser.add_argument("--key", default="C", help="Musical key (e.g., C, D#, A)")
    parser.add_argument("--scale", default="minor", help="Scale (major, minor, pentatonic, etc.)")
    args = parser.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        raise SystemExit(f"Input file not found: {infile}")

    audio, sr = load_audio(infile)

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32)

    print(f"Loaded {infile} — {x.shape[0] / float(sr):.2f}s @ {sr} Hz, shape={x.shape}")
    print(f"Key: {args.key}, Scale: {args.scale}")

    # Baseline: input alignment
    in_avg_cents, _ = avg_cents_offset_from_scale(
        audio=x,
        sr=sr,
        key=args.key,
        scale=args.scale,  # type: ignore[arg-type]
    )
    print(f"Input avg abs cents offset: {in_avg_cents:.2f} (NaN means insufficient voiced energy)")

    # Process with typical "musical but aggressive" settings
    print("=== Quantum Distortion Validation — Process Params ===")
    print("  key:", args.key)
    print("  scale:", args.scale)
    print("  snap_strength:", 0.8)
    print("  smear:", 0.4)
    print("  bin_smoothing:", True)
    print("  pre_quant:", True)
    print("  post_quant:", True)
    print("  distortion_mode:", "wavefold")
    print("  distortion_params:", {"fold_amount": 3.0, "bias": 0.0, "drive": 1.0, "warmth": 0.5})
    print("  limiter_on:", True)
    print("  limiter_ceiling_db:", -1.0)
    print("  dry_wet:", 1.0)
    print("======================================================")
    processed, _taps = process_audio(
        audio=x,
        sr=sr,
        key=args.key,
        scale=args.scale,
        snap_strength=0.8,
        smear=0.4,
        bin_smoothing=True,
        pre_quant=True,
        post_quant=True,
        distortion_mode="wavefold",
        distortion_params={"fold_amount": 3.0, "bias": 0.0, "drive": 1.0, "warmth": 0.5},
        limiter_on=True,
        limiter_ceiling_db=-1.0,
        dry_wet=1.0,
    )

    out_avg_cents, _ = avg_cents_offset_from_scale(
        audio=processed,
        sr=sr,
        key=args.key,
        scale=args.scale,  # type: ignore[arg-type]
    )
    print(f"Output avg abs cents offset: {out_avg_cents:.2f} (NaN means insufficient voiced energy)")

    if np.isfinite(in_avg_cents) and np.isfinite(out_avg_cents):
        delta = in_avg_cents - out_avg_cents
        print(f"Δ(avg abs cents) input - output: {delta:.2f} (positive = output more in key)")
    else:
        print("One of the cents offsets is NaN; cannot compute Δ meaningfully.")

    # --- Quantizer-only diagnostic: no distortion, no limiter ---
    print("\n=== Quantizer-only Diagnostic (no distortion, no limiter) ===")
    q_only_processed, _ = process_audio(
        audio=x,
        sr=sr,
        key=args.key,
        scale=args.scale,
        snap_strength=1.0,       # maximum snap
        smear=0.0,               # no smear
        bin_smoothing=False,
        pre_quant=True,
        post_quant=True,
        distortion_mode="wavefold",
        distortion_params={"fold_amount": 1.0, "bias": 0.0, "drive": 1.0, "warmth": 0.5},
        limiter_on=False,
        limiter_ceiling_db=-1.0,
        dry_wet=1.0,
    )

    q_avg_cents, _ = avg_cents_offset_from_scale(
        audio=q_only_processed,
        sr=sr,
        key=args.key,
        scale=args.scale,  # type: ignore[arg-type]
    )
    print(f"Quantizer-only output avg abs cents offset: {q_avg_cents:.2f}")


if __name__ == "__main__":
    main()

