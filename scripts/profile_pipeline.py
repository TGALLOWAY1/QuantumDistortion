from __future__ import annotations


from pathlib import Path
import argparse
import time


import numpy as np


from quantum_distortion.io.audio_io import load_audio
from quantum_distortion.dsp.pipeline import process_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Quantum Distortion pipeline runtime.")
    parser.add_argument("--infile", "-i", required=True, help="Input audio file (wav/aif/aiff)")
    args = parser.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        raise SystemExit(f"Input file not found: {infile}")

    audio, sr = load_audio(infile)

    # Downmix to mono if needed; pipeline will also handle this, but we make it explicit.
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32)

    dur_sec = x.shape[0] / float(sr)
    print(f"Loaded {infile} — {dur_sec:.2f} seconds @ {sr} Hz, shape={x.shape}")

    # Profile full process with representative settings
    start = time.perf_counter()
    processed, taps = process_audio(
        audio=x,
        sr=sr,
        key="C",
        scale="minor",
        snap_strength=0.8,
        smear=0.3,
        bin_smoothing=True,
        pre_quant=True,
        post_quant=True,
        distortion_mode="wavefold",
        distortion_params={"fold_amount": 3.0, "bias": 0.0, "drive": 1.0, "warmth": 0.5},
        limiter_on=True,
        limiter_ceiling_db=-1.0,
        dry_wet=1.0,
    )
    end = time.perf_counter()

    elapsed = end - start
    print(f"Processing time: {elapsed:.3f} s for {dur_sec:.2f} s of audio")
    if dur_sec > 0:
        print(f"Speed ratio: {dur_sec / elapsed:.2f} x real-time (if >1, faster than real-time)")

    # Basic sanity on taps
    assert set(taps.keys()) == {"input", "pre_quant", "post_dist", "output"}
    for name, buf in taps.items():
        assert buf.shape[0] == x.shape[0], f"Tap {name} has unexpected length {buf.shape[0]}"

    print("Profile run completed successfully.")


if __name__ == "__main__":
    main()

