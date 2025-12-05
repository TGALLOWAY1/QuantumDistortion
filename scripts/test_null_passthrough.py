#!/usr/bin/env python3
"""
Null test script for OLA STFT/ISTFT reconstruction verification.

This script tests the passthrough_test mode to verify that the OLA-compliant
STFT/ISTFT roundtrip reconstructs the input signal transparently.

Success threshold: RMS of delta < -90 dB indicates correct OLA reconstruction.
"""

from pathlib import Path
import argparse
import sys

import numpy as np

from quantum_distortion.io.audio_io import load_audio, save_audio
from quantum_distortion.dsp.pipeline import process_audio


def compute_rms_db(signal: np.ndarray) -> float:
    """Compute RMS in dB."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms == 0.0:
        return float('-inf')
    return 20.0 * np.log10(rms)


def main() -> None:
    """Run null test: verify passthrough_test mode reconstructs input exactly."""
    parser = argparse.ArgumentParser(
        description="Null test for OLA STFT/ISTFT reconstruction verification"
    )
    
    parser.add_argument(
        "--infile", "-i",
        type=str,
        default="tests/data/wobble_bass.wav",
        help="Input audio file to test (default: tests/data/wobble_bass.wav)"
    )
    
    parser.add_argument(
        "--outfile", "-o",
        type=str,
        default=None,
        help="Output file for delta.wav (default: /tmp/null_test_delta.wav or project root)"
    )
    
    args = parser.parse_args()
    
    # Load test audio
    infile_path = Path(args.infile)
    if not infile_path.exists():
        print(f"Error: Input file not found: {infile_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading test audio: {infile_path}")
    input_audio, sr = load_audio(infile_path)
    print(f"  Sample rate: {sr} Hz")
    print(f"  Shape: {input_audio.shape}")
    print(f"  Duration: {len(input_audio) / sr:.3f} seconds")
    
    # Ensure mono
    if input_audio.ndim == 2:
        input_audio = np.mean(input_audio, axis=1)
        print("  Converted to mono")
    
    # Run pipeline with passthrough_test=True
    # This should perform transparent STFT->ISTFT roundtrip
    print("\nRunning pipeline with passthrough_test=True...")
    output_audio, taps = process_audio(
        input_audio,
        sr=sr,
        passthrough_test=True,  # Bypass all FX, transparent STFT->ISTFT
    )
    
    print(f"  Output shape: {output_audio.shape}")
    
    # Align lengths
    min_length = min(len(input_audio), len(output_audio))
    input_aligned = input_audio[:min_length]
    output_aligned = output_audio[:min_length]
    
    if len(input_audio) != len(output_audio):
        print(f"  Warning: Length mismatch - input={len(input_audio)}, output={len(output_audio)}")
        print(f"  Truncated to {min_length} samples for comparison")
    
    # Compute delta = output - input
    delta = output_aligned - input_aligned
    
    # Compute metrics
    rms_delta = np.sqrt(np.mean(delta ** 2))
    peak_delta = np.max(np.abs(delta))
    rms_delta_db = compute_rms_db(delta)
    peak_delta_db = 20.0 * np.log10(peak_delta) if peak_delta > 0.0 else float('-inf')
    
    # Print results
    print("\n" + "=" * 60)
    print("NULL TEST RESULTS")
    print("=" * 60)
    print(f"RMS of delta:      {rms_delta:.6e} ({rms_delta_db:.2f} dB)")
    print(f"Peak of delta:      {peak_delta:.6e} ({peak_delta_db:.2f} dB)")
    print()
    
    # Success threshold: RMS < -90 dB
    success_threshold_db = -90.0
    if rms_delta_db < success_threshold_db:
        print(f"✓ SUCCESS: RMS < {success_threshold_db} dB")
        print("  OLA reconstruction is correct (transparent roundtrip)")
        success = True
    else:
        print(f"✗ FAILURE: RMS >= {success_threshold_db} dB")
        print("  OLA reconstruction may have issues")
        success = False
    print("=" * 60)
    
    # Save delta.wav
    if args.outfile:
        delta_path = Path(args.outfile)
    else:
        # Try /tmp first, fall back to project root
        delta_path = Path("/tmp/null_test_delta.wav")
        if not delta_path.parent.exists():
            # Fall back to project root
            project_root = Path(__file__).parent.parent
            delta_path = project_root / "null_test_delta.wav"
    
    print(f"\nSaving delta.wav: {delta_path}")
    save_audio(delta_path, delta, sr)
    print(f"  Saved {len(delta)} samples at {sr} Hz")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

