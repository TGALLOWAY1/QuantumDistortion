import numpy as np

from quantum_distortion.dsp.spectral_fx import (
    bitcrush,
    phase_dispersal,
    bin_scramble,
    SPECTRAL_FX_PRESETS,
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


def test_bin_scramble_preserves_shape_and_dtype() -> None:
    """Test that bin_scramble preserves shape and dtype for both modes."""
    np.random.seed(42)
    mag = np.random.rand(1024)
    phase = np.random.rand(1024)

    mag_out1, phase_out1 = bin_scramble(mag, phase, mode="random_pick")
    mag_out2, phase_out2 = bin_scramble(mag, phase, mode="swap")

    assert mag_out1.shape == mag.shape
    assert phase_out1.shape == phase.shape
    assert mag_out1.dtype == mag.dtype
    assert phase_out1.dtype == phase.dtype

    assert mag_out2.shape == mag.shape
    assert phase_out2.shape == phase.shape
    assert mag_out2.dtype == mag.dtype
    assert phase_out2.dtype == phase.dtype


def test_bin_scramble_random_pick_changes_spectrum() -> None:
    """Test that random_pick mode changes the spectrum."""
    np.random.seed(42)
    # Deterministic mag ramp
    mag = np.linspace(0.0, 1.0, 1024)
    phase = np.zeros_like(mag)

    mag_out, phase_out = bin_scramble(mag, phase, mode="random_pick")

    # Spectrum should have changed
    assert np.any(mag_out != mag)

    # Total energy should be within ±5% of original
    energy_ratio = np.sum(mag_out) / np.sum(mag)
    assert 0.95 <= energy_ratio <= 1.05


def test_bin_scramble_swap_mode_is_reasonable() -> None:
    """Test that swap mode preserves values but changes ordering."""
    np.random.seed(42)
    # Known pattern
    mag = np.arange(100, dtype=float)
    phase = np.zeros_like(mag)

    mag_out, phase_out = bin_scramble(mag, phase, mode="swap")

    # The set of values (before scaling) should be unchanged
    # Since we rescale, check that the multiset of values is preserved
    # by comparing sorted arrays (normalized by their sums)
    mag_normalized = np.sort(mag) / np.sum(mag)
    mag_out_normalized = np.sort(mag_out) / np.sum(mag_out)
    # After normalization and sorting, they should be very close
    assert np.allclose(mag_normalized, mag_out_normalized, rtol=1e-6)

    # But ordering should have changed for at least some indices
    # Check that at least some adjacent pairs differ from input
    assert np.any(mag_out != mag)


def test_spectral_fx_presets_are_well_formed() -> None:
    """Test that spectral FX presets are well-formed."""
    # Presets should be non-empty
    assert len(SPECTRAL_FX_PRESETS) > 0
    
    # Each preset should have required keys
    for preset_name, preset in SPECTRAL_FX_PRESETS.items():
        assert "mode" in preset, f"Preset '{preset_name}' missing 'mode' key"
        assert "distortion_strength" in preset, f"Preset '{preset_name}' missing 'distortion_strength' key"
        
        # Mode should be one of the valid modes
        assert preset["mode"] in ["bitcrush", "phase_dispersal", "bin_scramble"], \
            f"Preset '{preset_name}' has invalid mode: {preset['mode']}"
        
        # Distortion strength should be in valid range
        strength = preset["distortion_strength"]
        assert isinstance(strength, (int, float)), \
            f"Preset '{preset_name}' distortion_strength must be numeric"
        assert 0.0 <= strength <= 1.0, \
            f"Preset '{preset_name}' distortion_strength {strength} not in [0.0, 1.0]"
        
        # Params should be a dict if present
        if "params" in preset:
            assert isinstance(preset["params"], dict), \
                f"Preset '{preset_name}' params must be a dict"

