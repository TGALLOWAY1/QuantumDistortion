"""
Smoke tests for the process_file_to_file harness.
"""

from pathlib import Path

import numpy as np

from quantum_distortion.dsp.harness import process_file_to_file
from quantum_distortion.io.audio_io import load_audio


def test_process_file_to_file_creates_output(tmp_path: Path) -> None:
    """Test that process_file_to_file creates an output file."""
    # Use one of the generated fixtures
    fixture_dir = Path(__file__).parent / "data"
    infile = fixture_dir / "sub_sweep.wav"
    
    if not infile.exists():
        raise FileNotFoundError(
            f"Test fixture not found: {infile}. "
            "Run scripts/generate_test_fixtures.py first."
        )
    
    outfile = tmp_path / "output.wav"
    
    # Process the file
    process_file_to_file(infile, outfile)
    
    # Assert output file was created
    assert outfile.exists(), f"Output file was not created: {outfile}"
    assert outfile.stat().st_size > 0, "Output file is empty"


def test_process_file_to_file_output_is_valid_audio(tmp_path: Path) -> None:
    """Test that the output file can be loaded back as valid audio."""
    fixture_dir = Path(__file__).parent / "data"
    infile = fixture_dir / "wobble_bass.wav"
    
    if not infile.exists():
        raise FileNotFoundError(
            f"Test fixture not found: {infile}. "
            "Run scripts/generate_test_fixtures.py first."
        )
    
    outfile = tmp_path / "output.wav"
    
    # Process the file
    process_file_to_file(infile, outfile)
    
    # Load the output back
    audio, sr = load_audio(outfile)
    
    # Assert it's valid audio
    assert isinstance(audio, np.ndarray), "Output should be a numpy array"
    assert len(audio.shape) <= 2, "Output should be 1D or 2D"
    assert audio.size > 0, "Output should not be empty"
    assert sr > 0, "Sample rate should be positive"
    
    # Assert output has reasonable length (should be similar to input)
    input_audio, input_sr = load_audio(infile)
    assert abs(len(audio) - len(input_audio)) < input_sr * 0.1, (
        "Output length should be similar to input length"
    )


def test_process_file_to_file_with_extra_params(tmp_path: Path) -> None:
    """Test that extra_params can override default parameters."""
    fixture_dir = Path(__file__).parent / "data"
    infile = fixture_dir / "sub_sweep.wav"
    
    if not infile.exists():
        raise FileNotFoundError(
            f"Test fixture not found: {infile}. "
            "Run scripts/generate_test_fixtures.py first."
        )
    
    outfile = tmp_path / "output.wav"
    
    # Process with extra params that disable processing
    process_file_to_file(
        infile,
        outfile,
        extra_params={
            "snap_strength": 0.0,
            "pre_quant": False,
            "post_quant": False,
            "limiter_on": False,
            "dry_wet": 1.0,
        },
    )
    
    # Should still create valid output
    assert outfile.exists(), "Output file should be created"
    audio, sr = load_audio(outfile)
    assert audio.size > 0, "Output should not be empty"

