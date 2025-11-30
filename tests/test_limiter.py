import numpy as np


from quantum_distortion.dsp.limiter import peak_limiter, db_to_linear


def test_limiter_reduces_peaks() -> None:
    sr = 44100
    ceiling_db = -3.0
    ceiling_lin = db_to_linear(ceiling_db)

    # Build signal: low-level tone + single large spike
    t = np.linspace(0.0, 0.1, int(sr * 0.1), endpoint=False)
    tone = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    spike = np.zeros_like(tone)
    spike[len(spike) // 2] = 2.0  # large transient

    x = tone + spike
    limited, gain = peak_limiter(x, sr=sr, ceiling_db=ceiling_db, lookahead_ms=5.0, release_ms=50.0)

    # Check that limited signal does not exceed ceiling (with small tolerance)
    peak_out = float(np.max(np.abs(limited)))
    assert peak_out <= ceiling_lin * 1.01  # allow 1% margin


def test_limiter_minimal_effect_below_threshold() -> None:
    sr = 44100
    ceiling_db = -1.0
    ceiling_lin = db_to_linear(ceiling_db)

    t = np.linspace(0.0, 0.1, int(sr * 0.1), endpoint=False)
    x = 0.2 * np.sin(2.0 * np.pi * 440.0 * t)  # well below ceiling

    limited, gain = peak_limiter(x, sr=sr, ceiling_db=ceiling_db, lookahead_ms=5.0, release_ms=20.0)

    # Peaks should remain below ceiling
    assert float(np.max(np.abs(limited))) <= ceiling_lin * 1.01

    # Gain curve should be very close to 1.0 everywhere
    assert np.allclose(gain, np.ones_like(gain), atol=1e-2)


def test_limiter_gain_recovers_after_spike() -> None:
    sr = 44100
    ceiling_db = -6.0

    t = np.linspace(0.0, 0.2, int(sr * 0.2), endpoint=False)
    x = np.zeros_like(t)
    mid = len(x) // 2
    x[mid] = 3.0  # big spike

    limited, gain = peak_limiter(x, sr=sr, ceiling_db=ceiling_db, lookahead_ms=5.0, release_ms=30.0)

    # Gain around the spike should drop significantly
    assert gain[mid] < 0.5

    # Gain near the very end should have recovered towards 1.0
    assert gain[-1] > 0.9

