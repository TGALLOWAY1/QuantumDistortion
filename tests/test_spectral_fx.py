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


def test_phase_dispersal_smoke() -> None:
    """Smoke test for phase_dispersal function."""
    mag = np.linspace(0, 1, 1024)
    phase = np.zeros_like(mag)

    mag_out, phase_out = phase_dispersal(mag, phase)

    assert mag_out.shape == mag.shape
    assert phase_out.shape == phase.shape
    assert mag_out.dtype == mag.dtype
    assert phase_out.dtype == phase.dtype


def test_bin_scramble_smoke() -> None:
    """Smoke test for bin_scramble function."""
    mag = np.linspace(0, 1, 1024)
    phase = np.zeros_like(mag)

    mag_out, phase_out = bin_scramble(mag, phase)

    assert mag_out.shape == mag.shape
    assert phase_out.shape == phase.shape
    assert mag_out.dtype == mag.dtype
    assert phase_out.dtype == phase.dtype

