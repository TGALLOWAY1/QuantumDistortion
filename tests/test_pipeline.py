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

