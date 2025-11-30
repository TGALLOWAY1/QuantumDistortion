import numpy as np


from quantum_distortion.dsp.distortion import (
    wavefold,
    soft_tube,
    apply_distortion,
)


def _make_sine(freq: float, sr: int = 44100, seconds: float = 0.1) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    return 0.2 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def test_wavefold_generates_harmonics() -> None:
    sr = 44100
    sine = _make_sine(440.0, sr=sr)
    # Use more aggressive parameters to ensure harmonics are generated
    folded = wavefold(sine, fold_amount=8.0, bias=0.0, threshold=0.5)

    # FFT magnitude: expect more than 1 prominent bin (harmonics)
    spec_clean = np.abs(np.fft.rfft(sine))
    spec_folded = np.abs(np.fft.rfft(folded))

    # Find fundamental bin
    freqs = np.fft.rfftfreq(len(sine), 1.0 / sr)
    fundamental_idx = int(np.argmin(np.abs(freqs - 440.0)))

    # Compute harmonic energy (total - fundamental)
    clean_total = np.sum(spec_clean ** 2)
    clean_fund = spec_clean[fundamental_idx] ** 2
    clean_harmonics = clean_total - clean_fund

    folded_total = np.sum(spec_folded ** 2)
    folded_fund = spec_folded[fundamental_idx] ** 2
    folded_harmonics = folded_total - folded_fund

    # Folded signal should have significantly more harmonic energy
    # (normalize by fundamental to account for gain differences)
    clean_ratio = clean_harmonics / max(clean_fund, 1e-12)
    folded_ratio = folded_harmonics / max(folded_fund, 1e-12)

    # The folded signal should have more harmonic content relative to fundamental
    assert folded_ratio > clean_ratio * 1.1  # At least 10% more


def test_soft_tube_drive_increases_thd() -> None:
    sr = 44100
    sine = _make_sine(440.0, sr=sr)

    low_drive = soft_tube(sine, drive=1.0, warmth=0.5)
    high_drive = soft_tube(sine, drive=5.0, warmth=0.5)

    # Compute simple THD proxy: energy excluding fundamental
    spec_low = np.abs(np.fft.rfft(low_drive))
    spec_high = np.abs(np.fft.rfft(high_drive))

    freqs = np.fft.rfftfreq(len(sine), 1.0 / sr)
    fundamental_idx = int(np.argmin(np.abs(freqs - 440.0)))

    def thd(spec: np.ndarray) -> float:
        total = np.sum(spec**2)
        fund = spec[fundamental_idx] ** 2
        return float((total - fund) / max(fund, 1e-12))

    thd_low = thd(spec_low)
    thd_high = thd(spec_high)

    assert thd_high > thd_low


def test_apply_distortion_routes_modes() -> None:
    x = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)

    wf = apply_distortion(x, mode="wavefold", fold_amount=3.0, bias=0.1)
    tube = apply_distortion(x, mode="tube", drive=3.0, warmth=0.7)

    assert wf.shape == x.shape
    assert tube.shape == x.shape

    # Distorted outputs should differ from input and from each other
    assert not np.allclose(wf, x)
    assert not np.allclose(tube, x)
    assert not np.allclose(wf, tube)

