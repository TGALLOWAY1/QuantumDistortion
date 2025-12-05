"""
Quick regression suite for Quantum Distortion audio processing.

This script processes all test fixtures through the pipeline and reports
null-test metrics (residual RMS in dB) to track changes over time.

It compares single-band vs multiband processing modes to evaluate the impact
of the multiband split on transparency and quality.

It also tests spectral FX modes (bitcrush, phase_dispersal, bin_scramble) by
comparing them against a baseline high-band-only render.

Usage:
    python scripts/quick_regression_suite.py [--preset PRESET_NAME]

The script:
1. Locates WAV fixtures in tests/data/
2. Processes each through process_file_to_file:
   - Single-band mode (use_multiband=False)
   - Multiband mode baseline (use_multiband=True, crossover_hz=300.0, no spectral FX)
   - Multiband with each spectral FX mode (bitcrush, phase_dispersal, bin_scramble)
3. Computes residual RMS (in dB) between original and each processed version
4. For spectral FX, computes residual vs baseline high-band render
5. Prints comparison results

More negative dB values indicate more transparent processing (closer to original).
Less negative values indicate more coloration/processing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to path so imports work
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from quantum_distortion.dsp.harness import process_file_to_file
from tests.utils.audio_test_utils import load_audio, null_test


# Expected fixture files (minimum set)
EXPECTED_FIXTURES = [
    "sub_sweep.wav",
    "wobble_bass.wav",
    "kick_sub_combo.wav",
    "midrange_growl_like.wav",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run quick regression suite on audio test fixtures"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name to use for processing (if supported)",
    )
    args = parser.parse_args()

    # Locate fixtures directory (repo_root already set above)
    fixtures_dir = repo_root / "tests" / "data"
    processed_dir = fixtures_dir / "processed"

    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not fixtures_dir.exists():
        print(f"ERROR: Fixtures directory not found: {fixtures_dir}")
        print("Run scripts/generate_test_fixtures.py first to create fixtures.")
        return

    # Find all WAV files in fixtures directory
    fixture_files = sorted(fixtures_dir.glob("*.wav"))
    
    # Filter to only expected fixtures (exclude processed files)
    fixture_files = [
        f for f in fixture_files
        if f.name in EXPECTED_FIXTURES
    ]

    if not fixture_files:
        print(f"ERROR: No fixture files found in {fixtures_dir}")
        print(f"Expected at least one of: {', '.join(EXPECTED_FIXTURES)}")
        print("Run scripts/generate_test_fixtures.py first to create fixtures.")
        return

    print(f"Processing {len(fixture_files)} fixture(s) from {fixtures_dir}")
    if args.preset:
        print(f"Using preset: {args.preset}")
    print()

    results = []
    spectral_fx_results = []

    for fixture_path in fixture_files:
        fixture_name = fixture_path.name
        
        # Build output paths
        output_single_name = fixture_path.stem + "_singleband.wav"
        output_multi_name = fixture_path.stem + "_multiband.wav"
        output_multi_path = processed_dir / output_multi_name

        print(f"Processing: {fixture_name}...", flush=True)

        try:
            # Process single-band version
            print(f"  Single-band -> {output_single_name}...", end=" ", flush=True)
            output_single_path = processed_dir / output_single_name
            process_file_to_file(
                fixture_path,
                output_single_path,
                preset=args.preset,
                extra_params={"use_multiband": False},
            )
            
            # Process multiband baseline (no spectral FX)
            print(f"done")
            print(f"  Multiband baseline -> {output_multi_name}...", end=" ", flush=True)
            process_file_to_file(
                fixture_path,
                output_multi_path,
                preset=args.preset,
                extra_params={
                    "use_multiband": True,
                    "crossover_hz": 300.0,
                    "spectral_fx_mode": None,
                    "spectral_fx_strength": 0.0,
                },
            )
            print(f"done")

            # Load original and processed versions
            original, _ = load_audio(fixture_path)
            processed_single, _ = load_audio(output_single_path)
            processed_multi_baseline, _ = load_audio(output_multi_path)

            # Compute null tests (residual RMS in dB)
            residual_single = null_test(original, processed_single)
            residual_multi = null_test(original, processed_multi_baseline)

            results.append((fixture_name, residual_single, residual_multi))

            # Process spectral FX variants
            spectral_fx_modes = ["bitcrush", "phase_dispersal", "bin_scramble"]
            fx_strength = 0.5
            
            for fx_mode in spectral_fx_modes:
                try:
                    output_fx_name = fixture_path.stem + f"_multiband_{fx_mode}.wav"
                    output_fx_path = processed_dir / output_fx_name
                    
                    print(f"  Multiband + {fx_mode} -> {output_fx_name}...", end=" ", flush=True)
                    process_file_to_file(
                        fixture_path,
                        output_fx_path,
                        preset=args.preset,
                        extra_params={
                            "use_multiband": True,
                            "crossover_hz": 300.0,
                            "spectral_fx_mode": fx_mode,
                            "spectral_fx_strength": fx_strength,
                        },
                    )
                    print(f"done")
                    
                    # Load FX-processed version
                    processed_fx, _ = load_audio(output_fx_path)
                    
                    # Compute residual vs baseline (high-band-only render)
                    residual_vs_baseline = null_test(processed_multi_baseline, processed_fx)
                    
                    spectral_fx_results.append((fixture_name, fx_mode, residual_vs_baseline))
                except Exception as e:
                    print(f"ERROR: {e}")
                    spectral_fx_results.append((fixture_name, fx_mode, None))

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((fixture_name, None, None))

    print()
    print("=" * 80)
    print("Regression Summary: Single-Band vs Multiband")
    print("=" * 80)
    
    for fixture_name, residual_single, residual_multi in results:
        if residual_single is not None and residual_multi is not None:
            print(f"fixture={fixture_name:25s} single={residual_single:7.2f} dB multiband={residual_multi:7.2f} dB")
        else:
            print(f"fixture={fixture_name:25s} ERROR")
    
    print("=" * 80)
    print()
    
    if spectral_fx_results:
        print("=" * 80)
        print("Spectral FX vs Baseline (High-Band Only)")
        print("=" * 80)
        print("(Residual RMS in dB: FX-processed vs baseline multiband render)")
        print()
        
        for fixture_name, fx_mode, residual_vs_baseline in spectral_fx_results:
            if residual_vs_baseline is not None:
                print(f"fixture={fixture_name:25s} fx={fx_mode:15s} residual_vs_baseline={residual_vs_baseline:7.2f} dB")
            else:
                print(f"fixture={fixture_name:25s} fx={fx_mode:15s} ERROR")
        
        print("=" * 80)
        print()
    
    print("Interpretation:")
    print("  More negative dB = more similar / more transparent")
    print("  Less negative dB = more coloration / processing")
    print()
    print("For bass-heavy fixtures (sub_sweep, kick_sub_combo), multiband should")
    print("typically show better transparency in the low end due to time-domain")
    print("processing of the low band.")
    print()
    print("Spectral FX modes (bitcrush, phase_dispersal, bin_scramble) apply only")
    print("to the high band in multiband mode. The residual_vs_baseline metric")
    print("shows how much each FX mode changes the audio compared to baseline.")


if __name__ == "__main__":
    main()

