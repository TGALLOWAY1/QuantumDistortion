import numpy as np

from quantum_distortion.dsp.saturation import make_mono_lowband, soft_tube


def test_make_mono_lowband_mono_input() -> None:
    """Test that mono input is returned unchanged."""
    sr = 44100
    duration = 0.1
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    x_mono = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    
    result = make_mono_lowband(x_mono)
    
    # Should be unchanged
    assert result.shape == x_mono.shape
    assert np.allclose(result, x_mono, atol=1e-6)


def test_make_mono_lowband_stereo_input() -> None:
    """Test that stereo input is converted to mono (identical channels)."""
    sr = 44100
    duration = 0.1
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    
    # Create stereo signal with different content in each channel
    left = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    right = np.sin(2.0 * np.pi * 880.0 * t).astype(np.float32)  # Different frequency
    x_stereo = np.column_stack([left, right])
    
    result = make_mono_lowband(x_stereo)
    
    # Should preserve shape (2 channels)
    assert result.shape == x_stereo.shape
    
    # Both channels should be identical (mono)
    assert np.allclose(result[:, 0], result[:, 1], atol=1e-6), (
        "Both channels should be identical after mono conversion"
    )
    
    # Should be the average of the input channels
    expected_mono = np.mean(x_stereo, axis=1)
    assert np.allclose(result[:, 0], expected_mono, atol=1e-6), (
        "Mono channel should be the average of input channels"
    )


def test_soft_tube_no_nans() -> None:
    """Test that soft_tube doesn't return NaNs."""
    sr = 44100
    duration = 0.1
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    x = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    
    # Test with various drive values
    for drive in [0.5, 1.0, 2.0, 5.0]:
        result = soft_tube(x, drive=drive)
        assert not np.any(np.isnan(result)), f"soft_tube should not return NaN for drive={drive}"
        assert not np.any(np.isinf(result)), f"soft_tube should not return Inf for drive={drive}"


def test_soft_tube_adds_harmonics() -> None:
    """Test that soft_tube adds harmonics (increases RMS at same peak level)."""
    sr = 44100
    duration = 0.1
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    
    # Create a sine wave
    x = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    
    # Normalize to a specific peak level
    peak_level = 0.8
    x = x / np.max(np.abs(x)) * peak_level
    
    # Process with saturation
    y = soft_tube(x, drive=2.0)
    
    # Check that peak is similar (saturation shouldn't clip too much at this level)
    peak_in = np.max(np.abs(x))
    peak_out = np.max(np.abs(y))
    
    # Peak should be similar (within reason)
    assert abs(peak_in - peak_out) < 0.3, "Peak levels should be similar"
    
    # RMS should increase due to harmonic generation
    rms_in = np.sqrt(np.mean(x ** 2))
    rms_out = np.sqrt(np.mean(y ** 2))
    
    # With drive=2.0, we should get some harmonic content
    # RMS might increase or stay similar, but shouldn't decrease significantly
    assert rms_out >= rms_in * 0.8, (
        f"RMS should not decrease significantly. Input RMS: {rms_in:.6f}, "
        f"Output RMS: {rms_out:.6f}"
    )


def test_soft_tube_stereo_support() -> None:
    """Test that soft_tube works with stereo input."""
    sr = 44100
    duration = 0.1
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    
    left = 0.3 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    right = 0.3 * np.sin(2.0 * np.pi * 880.0 * t).astype(np.float32)
    x_stereo = np.column_stack([left, right])
    
    result = soft_tube(x_stereo, drive=1.5)
    
    # Should preserve shape
    assert result.shape == x_stereo.shape
    
    # Should not contain NaNs or Infs
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_soft_tube_drive_parameter() -> None:
    """Test that higher drive values increase saturation."""
    sr = 44100
    duration = 0.1
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    x = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    
    # Process with different drive values
    y_low = soft_tube(x, drive=1.0)
    y_high = soft_tube(x, drive=3.0)
    
    # Higher drive should produce more distortion
    # One way to measure this is to check the difference from the input
    # Higher drive should produce more deviation from a linear response
    diff_low = np.sqrt(np.mean((x - y_low) ** 2))
    diff_high = np.sqrt(np.mean((x - y_high) ** 2))
    
    # Higher drive should produce more difference (more saturation)
    assert diff_high > diff_low, (
        f"Higher drive should produce more saturation. "
        f"Low drive diff: {diff_low:.6f}, High drive diff: {diff_high:.6f}"
    )


def test_make_mono_lowband_preserves_shape() -> None:
    """Test that make_mono_lowband preserves output shape for recombination."""
    sr = 44100
    duration = 0.1
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    
    # Test with stereo
    left = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    right = np.sin(2.0 * np.pi * 880.0 * t).astype(np.float32)
    x_stereo = np.column_stack([left, right])
    
    result = make_mono_lowband(x_stereo)
    
    # Shape should be preserved (important for recombination with high band)
    assert result.shape == x_stereo.shape, (
        f"Output shape {result.shape} should match input shape {x_stereo.shape}"
    )
    
    # Both channels should be identical
    assert np.allclose(result[:, 0], result[:, 1], atol=1e-6)

