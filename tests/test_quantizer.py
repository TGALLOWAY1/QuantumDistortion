import numpy as np


from quantum_distortion.dsp.quantizer import (
    quantize_spectrum,
    build_target_bins_for_freqs,
)


def test_build_target_bins_simple_monotonic() -> None:
    # Frequencies in Hz: DC, off-scale, near A root
    freqs = np.array([0.0, 430.0, 440.0, 450.0], dtype=float)

    # Key A minor: A (440Hz) should be a strong attractor
    target_bins = build_target_bins_for_freqs(freqs, key="A", scale="minor")

    # DC bin maps to itself
    assert target_bins[0] == 0

    # 430 & 450 should both be attracted near 440 bin (index 2)
    assert target_bins[1] == 2
    assert target_bins[2] == 2
    assert target_bins[3] == 2


def test_quantize_moves_energy_to_target_bin() -> None:
    # 4-bin synthetic spectrum
    freqs = np.array([0.0, 430.0, 440.0, 450.0], dtype=float)
    mags = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    phases = np.zeros_like(mags)

    new_mags, new_phases = quantize_spectrum(
        mags=mags,
        phases=phases,
        freqs=freqs,
        key="A",
        scale="minor",
        snap_strength=1.0,
        smear=0.0,
        bin_smoothing=False,
    )

    # All energy should move from bin 1 → bin 2 (target)
    assert np.isclose(new_mags[1], 0.0, atol=1e-6)
    assert np.isclose(new_mags[2], 1.0, atol=1e-6)
    assert np.isclose(np.sum(new_mags), np.sum(mags), atol=1e-6)

    # Phases unchanged
    assert np.allclose(new_phases, phases)


def test_quantize_smear_distributes_energy() -> None:
    freqs = np.array([0.0, 430.0, 440.0, 450.0], dtype=float)
    mags = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    phases = np.zeros_like(mags)

    new_mags, _ = quantize_spectrum(
        mags=mags,
        phases=phases,
        freqs=freqs,
        key="A",
        scale="minor",
        snap_strength=1.0,
        smear=0.8,  # most energy smeared around target
        bin_smoothing=False,
        smear_radius=1,
    )

    # Source bin (1) has energy removed but may receive smeared energy back
    # since it's within smear_radius of target (2)
    # The key is that energy is distributed across multiple bins
    assert new_mags[1] < 0.5  # Less than original, but may have smeared energy

    # Neighbor bins around target (1,2,3) should all have some energy
    assert new_mags[1] > 0.0
    assert new_mags[2] > 0.0
    assert new_mags[3] > 0.0
    
    # Target bin should have the most energy (direct + smeared)
    assert new_mags[2] > new_mags[1]
    assert new_mags[2] > new_mags[3]

    # Total magnitude approximately conserved
    assert np.isclose(np.sum(new_mags), np.sum(mags), atol=1e-4)


def test_bin_smoothing_changes_shape_not_energy() -> None:
    freqs = np.array([100.0, 200.0, 300.0, 400.0], dtype=float)
    mags = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
    phases = np.zeros_like(mags)

    new_mags, _ = quantize_spectrum(
        mags=mags,
        phases=phases,
        freqs=freqs,
        key="C",
        scale="major",
        snap_strength=0.0,    # no movement
        smear=0.0,
        bin_smoothing=True,   # only smoothing
    )

    # Shape must change (central peak spreads)
    assert not np.allclose(new_mags, mags)

    # Energy should be approximately conserved
    assert np.isclose(np.sum(new_mags), np.sum(mags), atol=1e-6)

