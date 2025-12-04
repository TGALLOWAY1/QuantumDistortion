"""
Tests for audio test utilities (null_test, etc.).
"""

import numpy as np

from tests.utils.audio_test_utils import null_test, print_null_test


def test_null_test_identical_signals() -> None:
    """Test that identical signals produce very low residual (< -80 dB)."""
    # Generate random audio samples
    np.random.seed(42)  # For reproducibility
    x = np.random.randn(1000).astype(np.float32) * 0.1
    
    # Copy to y (identical signal)
    y = x.copy()
    
    residual_db = null_test(x, y)
    
    # Identical signals should have very low residual
    assert residual_db < -80.0, f"Expected residual < -80 dB for identical signals, got {residual_db:.2f} dB"


def test_null_test_perturbed_signals() -> None:
    """Test that slightly perturbed signals produce higher residual than identical ones."""
    # Generate random audio samples
    np.random.seed(42)
    x = np.random.randn(1000).astype(np.float32) * 0.1
    
    # Identical copy
    y = x.copy()
    
    # Slightly perturbed version
    noise = np.random.randn(1000).astype(np.float32) * 1e-5  # Very small noise
    z = x + noise
    
    residual_identical = null_test(x, y)
    residual_perturbed = null_test(x, z)
    
    # Perturbed signal should have higher (less negative) residual
    assert residual_perturbed > residual_identical, (
        f"Expected perturbed residual ({residual_perturbed:.2f} dB) > "
        f"identical residual ({residual_identical:.2f} dB)"
    )
    
    # But still should be quite low since noise is very small
    assert residual_perturbed < -40.0, (
        f"Expected perturbed residual < -40 dB for small noise, got {residual_perturbed:.2f} dB"
    )


def test_null_test_length_mismatch() -> None:
    """Test that length mismatches are handled by truncation."""
    x = np.random.randn(1000).astype(np.float32) * 0.1
    y = np.random.randn(1500).astype(np.float32) * 0.1
    
    # Should not raise an error, should truncate to min length
    residual_db = null_test(x, y)
    
    # Should be a valid float (not NaN or Inf)
    assert np.isfinite(residual_db), f"Expected finite residual, got {residual_db}"


def test_null_test_multichannel() -> None:
    """Test that multi-channel signals are handled correctly."""
    # Generate stereo signals
    np.random.seed(42)
    x = np.random.randn(1000, 2).astype(np.float32) * 0.1
    y = x.copy()
    
    residual_db = null_test(x, y)
    
    # Identical multi-channel signals should have very low residual
    assert residual_db < -80.0, (
        f"Expected residual < -80 dB for identical multi-channel signals, "
        f"got {residual_db:.2f} dB"
    )


def test_null_test_mono_vs_stereo() -> None:
    """Test that mono vs stereo comparison works (stereo is averaged to mono)."""
    np.random.seed(42)
    x = np.random.randn(1000).astype(np.float32) * 0.1
    
    # Create stereo version where both channels are identical to x
    y_stereo = np.stack([x, x], axis=1)
    
    residual_db = null_test(x, y_stereo)
    
    # Should be very low since stereo is just duplicated mono
    assert residual_db < -80.0, (
        f"Expected residual < -80 dB for mono vs duplicated stereo, "
        f"got {residual_db:.2f} dB"
    )


def test_print_null_test() -> None:
    """Test that print_null_test runs without error."""
    x = np.random.randn(100).astype(np.float32) * 0.1
    y = x.copy()
    
    # Should not raise an error
    print_null_test("Test label", x, y)

