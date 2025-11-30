from __future__ import annotations


from typing import Tuple


import numpy as np


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def peak_limiter(
    audio: np.ndarray,
    sr: int,
    ceiling_db: float = -1.0,
    lookahead_ms: float = 5.0,
    release_ms: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple lookahead peak limiter.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio buffer.
    sr : int
        Sample rate.
    ceiling_db : float
        Max allowed peak in dBFS (e.g., -1.0).
    lookahead_ms : float
        Lookahead window in milliseconds.
    release_ms : float
        Release time in milliseconds.

    Returns
    -------
    limited_audio : np.ndarray
        Limited audio buffer.
    gain_curve : np.ndarray
        Gain applied at each sample.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim != 1:
        raise ValueError("peak_limiter currently expects mono (1D) audio")

    n_samples = x.shape[0]
    if n_samples == 0:
        return x.astype(np.float32), np.ones_like(x, dtype=np.float32)

    ceiling_lin = db_to_linear(ceiling_db)
    lookahead_samples = int(max(1, round(sr * (lookahead_ms / 1000.0))))
    release_samples = max(1, int(round(sr * (release_ms / 1000.0))))

    gain = np.ones(n_samples, dtype=float)
    g = 1.0

    # Release coefficient for exponential smoothing back towards 1.0
    release_coeff = np.exp(-1.0 / release_samples)

    for n in range(n_samples):
        # Look at upcoming window to anticipate peaks
        window_end = min(n_samples, n + lookahead_samples)
        upcoming = x[n:window_end]
        peak = float(np.max(np.abs(upcoming))) if upcoming.size > 0 else 0.0

        if peak > ceiling_lin and peak > 1e-12:
            desired_g = ceiling_lin / peak
            # Attack: instantly lower gain if needed
            if desired_g < g:
                g = desired_g

        # Release: move g back towards 1.0 over time
        g = 1.0 - (1.0 - g) * release_coeff
        g = float(np.clip(g, 0.0, 1.0))
        gain[n] = g

    y = (x * gain).astype(np.float32)
    return y, gain.astype(np.float32)

