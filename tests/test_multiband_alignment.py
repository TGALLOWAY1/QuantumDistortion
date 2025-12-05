import numpy as np

from quantum_distortion.dsp.pipeline import process_audio


def test_multiband_transient_alignment() -> None:
    """
    Test that transients align properly between multiband and single-band modes.
    
    Creates a click/impulse (single-sample spike) and verifies that the main
    transient occurs at roughly the same sample index in both processing modes.
    """
    sr = 44100
    duration = 0.1  # 100ms
    n_samples = int(sr * duration)
    
    # Create a click/impulse: single-sample spike
    x = np.zeros(n_samples, dtype=np.float32)
    click_position = n_samples // 2  # Place click in the middle
    x[click_position] = 1.0  # Single-sample spike
    
    # Process with single-band mode (minimal processing to avoid reshaping the click)
    y_single, _ = process_audio(
        x,
        sr=sr,
        snap_strength=0.0,  # No quantization
        pre_quant=False,
        post_quant=False,
        distortion_mode="wavefold",
        distortion_params={"fold_amount": 1.0, "bias": 0.0},  # Minimal distortion
        limiter_on=False,
        dry_wet=1.0,
        use_multiband=False,
    )
    
    # Process with multiband mode (minimal processing)
    y_multi, _ = process_audio(
        x,
        sr=sr,
        snap_strength=0.0,  # No quantization
        pre_quant=False,
        post_quant=False,
        distortion_mode="wavefold",
        distortion_params={"fold_amount": 1.0, "bias": 0.0},  # Minimal distortion
        limiter_on=False,
        dry_wet=1.0,
        use_multiband=True,
        crossover_hz=300.0,
        lowband_drive=0.0,  # Minimal saturation to avoid reshaping the click
    )
    
    # Find the peak position in both outputs
    # Use absolute value to find the maximum transient
    idx_single = int(np.argmax(np.abs(y_single)))
    idx_multi = int(np.argmax(np.abs(y_multi)))
    
    # Calculate the offset
    offset = abs(idx_multi - idx_single)
    
    # The offset should be small (within a reasonable tolerance)
    # STFT latency is n_fft/2 = 1024 samples, but alignment should compensate
    # Allow some tolerance for filter ringing and numerical precision
    max_offset = 50  # Allow up to ~1ms offset at 44.1kHz
    
    assert offset < max_offset, (
        f"Transient misalignment too large: {offset} samples "
        f"(single-band peak at {idx_single}, multiband peak at {idx_multi}). "
        f"Expected offset < {max_offset} samples."
    )
    
    # Also verify that both outputs have valid transients (not completely smeared)
    peak_single = np.max(np.abs(y_single))
    peak_multi = np.max(np.abs(y_multi))
    
    # Both should have significant energy at the transient
    assert peak_single > 0.1, "Single-band output should have significant transient"
    assert peak_multi > 0.1, "Multiband output should have significant transient"


def test_multiband_alignment_with_different_crossovers() -> None:
    """Test that alignment works with different crossover frequencies."""
    sr = 44100
    duration = 0.1
    n_samples = int(sr * duration)
    
    # Create a click
    x = np.zeros(n_samples, dtype=np.float32)
    click_position = n_samples // 2
    x[click_position] = 1.0
    
    # Test with different crossover frequencies
    for crossover_hz in [200.0, 500.0, 1000.0]:
        y_multi, _ = process_audio(
            x,
            sr=sr,
            snap_strength=0.0,
            pre_quant=False,
            post_quant=False,
            distortion_mode="wavefold",
            distortion_params={"fold_amount": 1.0},
            limiter_on=False,
            dry_wet=1.0,
            use_multiband=True,
            crossover_hz=crossover_hz,
            lowband_drive=0.0,
        )
        
        # Find peak position
        idx_multi = int(np.argmax(np.abs(y_multi)))
        
        # Peak should be reasonably close to input click position
        # (allowing for filter delay and STFT latency)
        offset_from_input = abs(idx_multi - click_position)
        
        # With alignment, offset should be reasonable
        # Higher crossover = less filter delay, but STFT latency remains
        max_expected_offset = 1200  # ~27ms at 44.1kHz (STFT latency + some margin)
        
        assert offset_from_input < max_expected_offset, (
            f"For crossover {crossover_hz} Hz: transient offset {offset_from_input} samples "
            f"is too large (expected < {max_expected_offset})"
        )
        
        # Output should be valid
        assert not np.any(np.isnan(y_multi)), f"Output should not contain NaN for crossover {crossover_hz} Hz"
        assert not np.any(np.isinf(y_multi)), f"Output should not contain Inf for crossover {crossover_hz} Hz"

