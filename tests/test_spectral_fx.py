import numpy as np

from quantum_distortion.dsp.spectral_fx import (
    bitcrush,
    phase_dispersal,
    bin_scramble,
)


def test_bitcrush_smoke() -> None:
    """Smoke test for bitcrush function."""
    mag = np.linspace(0, 1, 1024)
    phase = np.zeros_like(mag)

    mag_out, phase_out = bitcrush(mag, phase)

    assert mag_out.shape == mag.shape
    assert phase_out.shape == phase.shape
    assert mag_out.dtype == mag.dtype
    assert phase_out.dtype == phase.dtype


def test_bitcrush_uniform_reduces_resolution() -> None:
    """Test that uniform bitcrush reduces resolution."""
    mag = np.linspace(0.0, 1.0, 1024)
    phase = np.zeros_like(mag)

    mag_q, phase_q = bitcrush(mag, phase, method="uniform", step=0.1)

    assert mag_q.shape == mag.shape
    assert phase_q.shape == phase.shape
    assert np.all(mag_q >= 0.0)

    # Check that quantization reduces unique values
    mag_q_unique = np.unique(np.round(mag_q, 3)).size
    assert 0 < mag_q_unique < mag.size


def test_bitcrush_log_preserves_monotonicity() -> None:
    """Test that log bitcrush preserves monotonicity."""
    mag = np.linspace(0.0, 1.0, 1024)
    phase = np.zeros_like(mag)

    mag_q, phase_q = bitcrush(mag, phase, method="log", step_db=3.0)

    # Check no crazy negative jumps (allow small numerical errors)
    assert np.all(np.diff(mag_q) >= -1e-6)

    # Check energy is within reasonable bounds
    assert np.sum(mag_q) <= np.sum(mag) * 1.05


def test_bitcrush_threshold_zeroes_small_bins() -> None:
    """Test that threshold zeroes bins below threshold."""
    # Create mag with some values below 0.1
    mag = np.array([0.05, 0.15, 0.08, 0.25, 0.03, 0.12, 0.20])
    phase = np.zeros_like(mag)

    mag_q, phase_q = bitcrush(mag, phase, threshold=0.1)

    # All bins < 0.1 should be zero
    assert np.all(mag_q[mag < 0.1] == 0.0)
    # Bins >= 0.1 should be preserved (or quantized but not zeroed)
    assert np.all(mag_q[mag >= 0.1] > 0.0)


def test_phase_dispersal_smoke() -> None:
    """Smoke test for phase_dispersal function."""
    mag = np.linspace(0, 1, 1024)
    phase = np.zeros_like(mag)

    mag_out, phase_out = phase_dispersal(mag, phase)

    assert mag_out.shape == mag.shape
    assert phase_out.shape == phase.shape
    assert mag_out.dtype == mag.dtype
    assert phase_out.dtype == phase.dtype


def test_phase_dispersal_preserves_magnitude() -> None:
    """Test that phase_dispersal preserves magnitude but changes phase."""
    np.random.seed(42)
    mag = np.random.rand(1024)
    phase = np.random.rand(1024) * 2 * np.pi - np.pi

    mag_out, phase_out = phase_dispersal(mag, phase, amount=0.5)

    assert np.allclose(mag, mag_out)
    assert not np.allclose(phase, phase_out)


def test_phase_dispersal_wraps_phase_to_valid_range() -> None:
    """Test that phase_dispersal wraps phase to [-pi, pi]."""
    mag = np.ones(1024)
    phase = np.zeros(1024)

    mag_out, phase_out = phase_dispersal(mag, phase, amount=np.pi)

    assert np.all(phase_out >= -np.pi)
    assert np.all(phase_out <= np.pi)


def test_phase_dispersal_respects_threshold() -> None:
    """Test that phase_dispersal respects the threshold parameter."""
    # Create mag such that half < thresh, half > thresh
    thresh = 0.5
    mag = np.concatenate([np.linspace(0.0, 0.4, 512), np.linspace(0.6, 1.0, 512)])
    phase = np.zeros(1024)

    mag_out, phase_out = phase_dispersal(mag, phase, thresh=thresh, amount=1.0)

    below_mask = mag <= thresh
    above_mask = mag > thresh

    # Bins below threshold should change by at most a tiny epsilon
    phase_change_below = np.abs(phase_out[below_mask] - phase[below_mask])
    assert np.all(phase_change_below < 1e-10)

    # Bins above threshold should change more
    phase_change_above = np.abs(phase_out[above_mask] - phase[above_mask])
    assert np.all(phase_change_above > 0.1)


def test_bin_scramble_smoke() -> None:
    """Smoke test for bin_scramble function."""
    mag = np.linspace(0, 1, 1024)
    phase = np.zeros_like(mag)

    mag_out, phase_out = bin_scramble(mag, phase)

    assert mag_out.shape == mag.shape
    assert phase_out.shape == phase.shape
    assert mag_out.dtype == mag.dtype
    assert phase_out.dtype == phase.dtype

