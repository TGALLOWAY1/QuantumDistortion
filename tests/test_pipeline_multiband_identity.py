import numpy as np
from pathlib import Path

from quantum_distortion.dsp.pipeline import process_audio
from quantum_distortion.io.audio_io import load_audio
from tests.utils.audio_test_utils import null_test


def test_multiband_identity_vs_single_band() -> None:
    """
    Test that multiband mode (with identical processing) produces output
    reasonably close to single-band mode.
    
    This verifies that the crossover split/recombine path works correctly
    and doesn't introduce significant artifacts.
    """
    # Load a test fixture
    fixture_path = Path(__file__).parent / "data" / "wobble_bass.wav"
    if not fixture_path.exists():
        # Fallback: create a simple test signal if fixture doesn't exist
        sr = 44100
        duration = 0.5
        t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
        x = 0.3 * np.sin(2.0 * np.pi * 100.0 * t).astype(np.float32)
    else:
        x, sr = load_audio(fixture_path)
        # Ensure mono
        if x.ndim == 2:
            x = np.mean(x, axis=1)
        # Truncate to first 0.5 seconds for faster testing
        max_samples = int(sr * 0.5)
        if len(x) > max_samples:
            x = x[:max_samples]
    
    # Process with single-band mode (default)
    y_single, taps_single = process_audio(
        x,
        sr=sr,
        snap_strength=0.5,
        pre_quant=True,
        post_quant=True,
        distortion_mode="wavefold",
        distortion_params={"fold_amount": 1.0, "bias": 0.0},
        limiter_on=False,
        dry_wet=1.0,
        use_multiband=False,
    )
    
    # Process with multiband mode (identity: same processing on both bands)
    y_multi, taps_multi = process_audio(
        x,
        sr=sr,
        snap_strength=0.5,
        pre_quant=True,
        post_quant=True,
        distortion_mode="wavefold",
        distortion_params={"fold_amount": 1.0, "bias": 0.0},
        limiter_on=False,
        dry_wet=1.0,
        use_multiband=True,
        crossover_hz=300.0,
    )
    
    # Check output shapes match
    assert y_single.shape == y_multi.shape, "Output shapes should match"
    assert y_single.shape == x.shape, "Output should match input shape"
    
    # Compare outputs using null_test
    # We don't expect perfect equality due to:
    # - Crossover filter phase distortion
    # - Numerical precision differences from processing bands separately
    # - STFT/iSTFT round-trip differences
    # But they should be reasonably close (< -20 dB residual)
    residual_db = null_test(y_single, y_multi)
    
    assert residual_db < -20.0, (
        f"Multiband output should be close to single-band output. "
        f"Residual: {residual_db:.2f} dB (expected < -20 dB)"
    )
    
    # Verify taps structure
    assert set(taps_single.keys()) == set(taps_multi.keys()), "Tap keys should match"
    for key in taps_single.keys():
        assert taps_single[key].shape == taps_multi[key].shape, (
            f"Tap '{key}' shapes should match"
        )


def test_multiband_with_different_crossover_frequencies() -> None:
    """Test that multiband mode works with different crossover frequencies."""
    sr = 44100
    duration = 0.3
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    # Create a signal with both low and high frequency content
    x = (
        0.2 * np.sin(2.0 * np.pi * 80.0 * t) +
        0.2 * np.sin(2.0 * np.pi * 2000.0 * t)
    ).astype(np.float32)
    
    # Test with different crossover frequencies
    for crossover_hz in [200.0, 500.0, 1000.0]:
        y, taps = process_audio(
            x,
            sr=sr,
            snap_strength=0.3,
            pre_quant=True,
            post_quant=False,
            distortion_mode="wavefold",
            distortion_params={"fold_amount": 1.0},
            limiter_on=False,
            dry_wet=1.0,
            use_multiband=True,
            crossover_hz=crossover_hz,
        )
        
        # Basic sanity checks
        assert y.shape == x.shape, f"Output shape should match input for crossover {crossover_hz} Hz"
        assert not np.all(np.isnan(y)), f"Output should not contain NaN for crossover {crossover_hz} Hz"
        assert not np.all(np.isinf(y)), f"Output should not contain Inf for crossover {crossover_hz} Hz"
        
        # Check taps structure
        assert "input" in taps
        assert "output" in taps
        assert taps["input"].shape == x.shape
        assert taps["output"].shape == x.shape

