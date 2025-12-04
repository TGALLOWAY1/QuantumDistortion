"""
Quick regression suite for Quantum Distortion audio processing.

This script processes all test fixtures through the pipeline and reports
null-test metrics (residual RMS in dB) to track changes over time.

It compares single-band vs multiband processing modes to evaluate the impact
of the multiband split on transparency and quality.

Usage:
    python scripts/quick_regression_suite.py [--preset PRESET_NAME]

The script:
1. Locates WAV fixtures in tests/data/
2. Processes each through process_file_to_file twice:
   - Single-band mode (use_multiband=False)
   - Multiband mode (use_multiband=True, crossover_hz=300.0)
3. Computes residual RMS (in dB) between original and each processed version
4. Prints comparison results

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

    for fixture_path in fixture_files:
        fixture_name = fixture_path.name
        
        # Build output paths
        output_single_name = fixture_path.stem + "_singleband.wav"
        output_multi_name = fixture_path.stem + "_multiband.wav"
        output_single_path = processed_dir / output_single_name
        output_multi_path = processed_dir / output_multi_name

        print(f"Processing: {fixture_name}...", flush=True)

        try:
            # Process single-band version
            print(f"  Single-band -> {output_single_name}...", end=" ", flush=True)
            process_file_to_file(
                fixture_path,
                output_single_path,
                preset=args.preset,
                extra_params={"use_multiband": False},
            )
            
            # Process multiband version
            print(f"done")
            print(f"  Multiband -> {output_multi_name}...", end=" ", flush=True)
            process_file_to_file(
                fixture_path,
                output_multi_path,
                preset=args.preset,
                extra_params={
                    "use_multiband": True,
                    "crossover_hz": 300.0,
                },
            )
            print(f"done")

            # Load original and both processed versions
            original, _ = load_audio(fixture_path)
            processed_single, _ = load_audio(output_single_path)
            processed_multi, _ = load_audio(output_multi_path)

            # Compute null tests (residual RMS in dB)
            residual_single = null_test(original, processed_single)
            residual_multi = null_test(original, processed_multi)

            results.append((fixture_name, residual_single, residual_multi))

        except Exception as e:
            print(f"ERROR: {e}")
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
    print("Interpretation:")
    print("  More negative dB = more similar / more transparent")
    print("  Less negative dB = more coloration / processing")
    print()
    print("For bass-heavy fixtures (sub_sweep, kick_sub_combo), multiband should")
    print("typically show better transparency in the low end due to time-domain")
    print("processing of the low band.")


if __name__ == "__main__":
    main()

