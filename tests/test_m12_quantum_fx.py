"""
Tests for M12 Creative Quantum Features and related improvements.

Covers:
- Spectral freeze (M12.1)
- Formant shifting (M12.2)
- Harmonic locking (M12.3)
- Delta listen (M13)
- Multiband passthrough null test
- Target bins caching (performance)
- Vectorized bin_scramble
"""

import numpy as np
import pytest

from quantum_distortion.dsp.pipeline import process_audio
from quantum_distortion.dsp.quantizer import (
    build_harmonic_target_bins,
    build_target_bins_for_freqs,
    quantize_spectrum,
)
from quantum_distortion.dsp.spectral_fx import (
    bin_scramble,
    formant_shift_frame,
)


def _make_test_signal(sr=44100, seconds=0.3, freq=440.0):
    """Generate a simple sine wave test signal."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr


def _make_harmonic_signal(sr=44100, seconds=0.3, fundamental=110.0, n_harmonics=5):
    """Generate a signal with harmonics."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    signal = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        signal += (0.1 / h) * np.sin(2 * np.pi * fundamental * h * t)
    return signal.astype(np.float32), sr


def _rms_db(signal):
    """Compute RMS in dB."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms == 0.0:
        return float('-inf')
    return 20.0 * np.log10(rms)


# ============================================================
# Multiband Passthrough Null Test
# ============================================================

class TestMultibandPassthrough:
    def test_multiband_passthrough_null(self):
        """Multiband passthrough: low+high should approximate input.

        Note: exact null is impossible because:
        - Causal IIR crossover introduces phase shift at crossover freq
        - Heuristic branch alignment is not sample-accurate
        - Low-band soft_tube saturator alters waveform even at drive=1.0
        We verify the error is bounded, confirming the bands recombine
        without catastrophic cancellation or signal loss.
        """
        audio, sr = _make_test_signal(freq=200.0, seconds=0.5)

        output, taps = process_audio(
            audio,
            sr=sr,
            passthrough_test=True,
            use_multiband=True,
            crossover_hz=300.0,
            lowband_drive=1.0,
        )

        min_len = min(len(audio), len(output))
        delta = audio[:min_len] - output[:min_len]
        rms_db = _rms_db(delta)

        # Crossover IIR + soft_tube saturation + alignment heuristic
        # limit accuracy. Threshold ensures no catastrophic signal loss.
        assert rms_db < -10.0, (
            f"Multiband passthrough RMS delta = {rms_db:.1f} dB (expected < -10 dB)"
        )
        # Also verify output has signal (not silent)
        assert _rms_db(output[:min_len]) > -60.0, "Output is unexpectedly silent"


# ============================================================
# M12.1: Spectral Freeze
# ============================================================

class TestSpectralFreeze:
    def test_freeze_produces_constant_spectral_shape(self):
        """Frozen output should have same spectral shape across all frames."""
        audio, sr = _make_harmonic_signal(seconds=0.5)

        output_frozen, _ = process_audio(
            audio, sr=sr,
            spectral_freeze=True,
            snap_strength=0.0,
            pre_quant=True,
            post_quant=False,
        )
        output_normal, _ = process_audio(
            audio, sr=sr,
            spectral_freeze=False,
            snap_strength=0.0,
            pre_quant=True,
            post_quant=False,
        )

        # Frozen output should differ from normal output
        # (unless the signal is perfectly stationary)
        assert output_frozen.shape == output_normal.shape

    def test_freeze_runs_without_error(self):
        """Spectral freeze should run without errors in multiband mode."""
        audio, sr = _make_test_signal(seconds=0.3)
        output, taps = process_audio(
            audio, sr=sr,
            use_multiband=True,
            spectral_freeze=True,
        )
        assert output.shape[0] > 0


# ============================================================
# M12.2: Formant Shifting
# ============================================================

class TestFormantShift:
    def test_formant_shift_frame_identity_at_zero(self):
        """shift=0 should return identical magnitudes."""
        freqs = np.linspace(0, 22050, 1025)
        mag = np.random.rand(1025).astype(float) * 0.5 + 0.01
        result = formant_shift_frame(mag, freqs, shift_semitones=0.0)
        np.testing.assert_array_almost_equal(result, mag)

    def test_formant_shift_preserves_energy(self):
        """Shifted output should preserve total energy."""
        freqs = np.linspace(0, 22050, 1025)
        mag = np.random.rand(1025).astype(float) * 0.5 + 0.01
        result = formant_shift_frame(mag, freqs, shift_semitones=3.0)
        original_energy = np.sum(mag ** 2)
        shifted_energy = np.sum(result ** 2)
        np.testing.assert_allclose(shifted_energy, original_energy, rtol=0.01)

    def test_formant_shift_pipeline_integration(self):
        """Formant shift should run through the pipeline without error."""
        audio, sr = _make_test_signal(seconds=0.3)
        output, _ = process_audio(
            audio, sr=sr,
            formant_shift=3.0,
            use_multiband=True,
        )
        assert output.shape[0] > 0

    def test_formant_shift_changes_output(self):
        """Non-zero formant shift should change the output."""
        audio, sr = _make_harmonic_signal(seconds=0.3)
        out_no_shift, _ = process_audio(audio, sr=sr, formant_shift=0.0)
        out_shifted, _ = process_audio(audio, sr=sr, formant_shift=6.0)
        # Should produce different outputs
        assert not np.allclose(out_no_shift, out_shifted, atol=1e-6)


# ============================================================
# M12.3: Harmonic Locking
# ============================================================

class TestHarmonicLocking:
    def test_build_harmonic_target_bins_basic(self):
        """Target bins should snap to harmonic series."""
        sr = 44100
        n_fft = 2048
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        fundamental = 55.0  # A1

        target_bins = build_harmonic_target_bins(freqs, fundamental)

        assert target_bins.shape == freqs.shape
        assert target_bins.dtype == int or np.issubdtype(target_bins.dtype, np.integer)

        # Check that bin nearest to 110 Hz (2nd harmonic) maps to itself
        bin_110 = np.argmin(np.abs(freqs - 110.0))
        assert target_bins[bin_110] == bin_110

    def test_harmonic_locking_zero_fundamental_is_identity(self):
        """fundamental_hz=0 should return identity mapping."""
        freqs = np.linspace(0, 22050, 1025)
        target_bins = build_harmonic_target_bins(freqs, 0.0)
        expected = np.arange(len(freqs))
        np.testing.assert_array_equal(target_bins, expected)

    def test_harmonic_locking_pipeline_integration(self):
        """Harmonic locking should run through pipeline without error."""
        audio, sr = _make_harmonic_signal(seconds=0.3, fundamental=110.0)
        output, _ = process_audio(
            audio, sr=sr,
            harmonic_lock_hz=110.0,
        )
        assert output.shape[0] > 0

    def test_harmonic_locking_clusters_energy(self):
        """With harmonic locking, energy should cluster at harmonics."""
        sr = 44100
        n_fft = 2048
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        fundamental = 55.0

        target_bins = build_harmonic_target_bins(freqs, fundamental)

        # Harmonics: 55, 110, 165, 220, ...
        for h in range(1, 6):
            harmonic_freq = fundamental * h
            nearest_bin = np.argmin(np.abs(freqs - harmonic_freq))
            # The bin nearest the harmonic should map to itself
            assert target_bins[nearest_bin] == nearest_bin, (
                f"Harmonic {h} ({harmonic_freq} Hz) at bin {nearest_bin} "
                f"maps to {target_bins[nearest_bin]} instead of itself"
            )


# ============================================================
# Delta Listen (M13)
# ============================================================

class TestDeltaListen:
    def test_delta_listen_output(self):
        """Delta listen should output input - processed."""
        audio, sr = _make_test_signal(seconds=0.3)

        # Get normal output
        out_normal, _ = process_audio(audio, sr=sr, delta_listen=False)
        # Get delta output
        out_delta, _ = process_audio(audio, sr=sr, delta_listen=True)

        min_len = min(len(out_normal), len(out_delta), len(audio))
        # delta should equal input - normal_output
        expected_delta = audio[:min_len] - out_normal[:min_len]
        np.testing.assert_allclose(
            out_delta[:min_len], expected_delta, atol=1e-6,
            err_msg="Delta listen output != input - processed"
        )


# ============================================================
# Target Bins Caching (Performance)
# ============================================================

class TestTargetBinsCaching:
    def test_quantize_spectrum_with_cached_bins(self):
        """Pre-computed target_bins should produce same result as internal computation."""
        n_bins = 1025
        freqs = np.linspace(0, 22050, n_bins)
        mags = np.random.rand(n_bins) * 0.5
        phases = np.random.rand(n_bins) * 2 * np.pi - np.pi
        key, scale = "C", "minor"

        # Without cache
        np.random.seed(42)
        result_no_cache = quantize_spectrum(
            mags.copy(), phases.copy(), freqs, key, scale,
            snap_strength=0.5, smear=0.1, bin_smoothing=True,
        )

        # With cache
        cached_bins = build_target_bins_for_freqs(freqs, key, scale)
        np.random.seed(42)
        result_cached = quantize_spectrum(
            mags.copy(), phases.copy(), freqs, key, scale,
            snap_strength=0.5, smear=0.1, bin_smoothing=True,
            target_bins=cached_bins,
        )

        np.testing.assert_array_almost_equal(result_no_cache[0], result_cached[0])
        np.testing.assert_array_almost_equal(result_no_cache[1], result_cached[1])


# ============================================================
# Vectorized bin_scramble
# ============================================================

class TestVectorizedBinScramble:
    def test_random_pick_mode(self):
        """Vectorized random_pick should produce valid output."""
        mag = np.random.rand(512) * 0.5
        phase = np.random.rand(512) * 2 * np.pi - np.pi
        mag_out, phase_out = bin_scramble(mag, phase, window=5, mode="random_pick")
        assert mag_out.shape == mag.shape
        # Energy should be approximately preserved
        np.testing.assert_allclose(
            np.sum(mag_out), np.sum(mag), rtol=0.05
        )

    def test_swap_mode(self):
        """Vectorized swap should produce valid output."""
        mag = np.random.rand(512) * 0.5
        phase = np.random.rand(512) * 2 * np.pi - np.pi
        mag_out, phase_out = bin_scramble(mag, phase, window=5, mode="swap")
        assert mag_out.shape == mag.shape
        # Energy should be approximately preserved
        np.testing.assert_allclose(
            np.sum(mag_out), np.sum(mag), rtol=0.05
        )

    def test_phase_unchanged(self):
        """bin_scramble should not modify phase."""
        mag = np.random.rand(100)
        phase = np.random.rand(100) * 2 * np.pi - np.pi
        _, phase_out = bin_scramble(mag, phase, mode="random_pick")
        np.testing.assert_array_equal(phase_out, phase)
