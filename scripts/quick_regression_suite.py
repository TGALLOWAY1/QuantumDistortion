"""
Quick regression suite for Quantum Distortion audio processing.

This script processes all test fixtures through the pipeline and reports
null-test metrics (residual RMS in dB) to track changes over time.

Usage:
    python scripts/quick_regression_suite.py [--preset PRESET_NAME]

The script:
1. Locates WAV fixtures in tests/data/
2. Processes each through process_file_to_file
3. Computes residual RMS (in dB) between original and processed
4. Prints summary results

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
        
        # Build output path: <name>_processed.wav
        output_name = fixture_path.stem + "_processed.wav"
        output_path = processed_dir / output_name

        print(f"Processing: {fixture_name} -> {output_path.name}...", end=" ", flush=True)

        try:
            # Process the fixture
            process_file_to_file(
                fixture_path,
                output_path,
                preset=args.preset,
            )

            # Load original and processed
            original, _ = load_audio(fixture_path)
            processed, _ = load_audio(output_path)

            # Compute null test (residual RMS in dB)
            residual_db = null_test(original, processed)

            results.append((fixture_name, residual_db))
            print(f"residual_rms_db={residual_db:.2f}")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append((fixture_name, None))

    print()
    print("=" * 60)
    print("Regression Summary")
    print("=" * 60)
    
    for fixture_name, residual_db in results:
        if residual_db is not None:
            print(f"fixture={fixture_name} residual_rms_db={residual_db:.2f}")
        else:
            print(f"fixture={fixture_name} ERROR")
    
    print("=" * 60)
    print()
    print("Interpretation:")
    print("  More negative dB = more similar / more transparent")
    print("  Less negative dB = more coloration / processing")


if __name__ == "__main__":
    main()

