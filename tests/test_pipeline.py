import numpy as np


from typing import Tuple


from quantum_distortion.dsp.pipeline import process_audio


def _make_sine(freq: float, sr: int = 44100, seconds: float = 0.2) -> Tuple[np.ndarray, int]:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    x = 0.1 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return x, sr


def test_pipeline_runs_and_taps_shapes() -> None:
    x, sr = _make_sine(220.0)

    # Use minimal processing to speed up test
    y, taps = process_audio(
        x,
        sr=sr,
        snap_strength=0.0,
        pre_quant=False,
        post_quant=False,
        limiter_on=False,
    )

    # All taps should exist and have same length as input
    assert set(taps.keys()) == {"input", "pre_quant", "post_dist", "output"}
    for name, buf in taps.items():
        assert buf.shape == x.shape, f"Tap {name} has wrong shape"

    # Output shape must match input
    assert y.shape == x.shape


def test_pipeline_neutral_settings_approx_passthrough() -> None:
    x, sr = _make_sine(220.0)

    # Settings chosen to minimize processing:
    # - No pre/post quantization (pre_quant=False, post_quant=False)
    # - Distortion with low impact (fold_amount=1, bias=0, drive=1)
    # - Limiter off
    # - Dry/wet = 1 (all wet, but nearly unchanged)
    y, _ = process_audio(
        x,
        sr=sr,
        key="C",
        scale="major",
        snap_strength=0.0,
        smear=0.0,
        bin_smoothing=False,
        pre_quant=False,
        post_quant=False,
        distortion_mode="wavefold",
        distortion_params={"fold_amount": 1.0, "bias": 0.0},
        limiter_on=False,
        dry_wet=1.0,
    )

    # Allow small numerical differences due to float ops
    assert np.allclose(y, x, atol=1e-3)


def _make_sweep(sr: int = 44100, seconds: float = 0.2) -> Tuple[np.ndarray, int]:
    """Create a simple sine sweep for testing."""
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    # Sweep from 440 Hz to 880 Hz
    freq = 440.0 + (440.0 * t / seconds)
    x = 0.1 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return x, sr


def test_spectral_fx_integration() -> None:
    """Test that spectral FX are integrated into high-band pipeline."""
    x, sr = _make_sweep(seconds=0.1)  # Short test signal
    
    # Baseline: no spectral FX
    y_baseline, _ = process_audio(
        x,
        sr=sr,
        use_multiband=True,
        crossover_hz=300.0,
        snap_strength=0.5,
        pre_quant=True,
        post_quant=False,
        limiter_on=False,
        spectral_fx_mode=None,
        spectral_fx_strength=0.0,
    )
    
    # Test each spectral FX mode with small strength
    modes = ["bitcrush", "phase_dispersal", "bin_scramble"]
    
    for mode in modes:
        y_fx, _ = process_audio(
            x,
            sr=sr,
            use_multiband=True,
            crossover_hz=300.0,
            snap_strength=0.5,
            pre_quant=True,
            post_quant=False,
            limiter_on=False,
            spectral_fx_mode=mode,
            spectral_fx_strength=0.3,  # Small strength for testing
        )
        
        # Output length should match input
        assert y_fx.shape == x.shape, f"{mode}: output length mismatch"
        
        # RMS difference should be > 0 (effect actually changed audio)
        rms_diff = np.sqrt(np.mean((y_fx - y_baseline) ** 2))
        assert rms_diff > 0.0, f"{mode}: no change detected (RMS diff = {rms_diff})"

