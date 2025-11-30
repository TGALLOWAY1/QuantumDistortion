from __future__ import annotations


from typing import Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


import numpy as np
import matplotlib.pyplot as plt


from quantum_distortion.dsp.quantizer import build_scale_notes


TapSource = Literal["input", "pre_quant", "post_dist", "output"]


def _select_segment(
    audio: np.ndarray,
    sr: int,
    duration: float = 0.1,
    center: bool = True,
) -> np.ndarray:
    """
    Utility: select a short segment from the audio for visualization.

    Parameters
    ----------
    audio : np.ndarray
        1D mono signal.
    sr : int
        Sample rate.
    duration : float
        Duration of the segment in seconds.
    center : bool
        If True, pick a centered segment. Otherwise, from the start.

    Returns
    -------
    segment : np.ndarray
        Selected segment (<= original length).
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim != 1:
        raise ValueError("_select_segment expects mono (1D) audio")

    n_samples = x.shape[0]
    seg_len = int(max(1, round(sr * duration)))
    if seg_len >= n_samples:
        return x.astype(np.float32)

    if center:
        mid = n_samples // 2
        start = max(0, mid - seg_len // 2)
    else:
        start = 0

    end = min(n_samples, start + seg_len)
    return x[start:end].astype(np.float32)


def plot_spectrum(
    audio: np.ndarray,
    sr: int,
    tap_source: TapSource,
    key: Optional[str] = None,
    scale: Optional[str] = None,
    show_scale_lines: bool = False,
    max_freq: Optional[float] = None,
) -> plt.Figure:
    """
    Plot magnitude spectrum (linear frequency axis, dB magnitude).

    Parameters
    ----------
    audio : np.ndarray
        Mono audio buffer.
    sr : int
        Sample rate.
    tap_source : TapSource
        Label for plot title (e.g. "input", "post_dist").
    key, scale : optional
        If provided and show_scale_lines=True, draw vertical lines at in-key frequencies.
    max_freq : float, optional
        Optional upper frequency limit for display.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    segment = _select_segment(audio, sr, duration=0.1, center=True)
    n = segment.shape[0]
    if n == 0:
        raise ValueError("Empty audio segment passed to plot_spectrum")

    window = np.hanning(n)
    seg_win = segment * window

    spec = np.fft.rfft(seg_win)
    mags = np.abs(spec)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # Convert to dB, add small epsilon to avoid log(0)
    eps = 1e-12
    mags_db = 20.0 * np.log10(np.maximum(mags, eps))

    if max_freq is None or max_freq <= 0.0 or max_freq > sr / 2.0:
        max_freq = sr / 2.0

    # Mask by max_freq
    mask = freqs <= max_freq
    freqs_disp = freqs[mask]
    mags_disp = mags_db[mask]

    fig, ax = plt.subplots()
    ax.plot(freqs_disp, mags_disp)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(f"Spectrum — {tap_source}")

    if show_scale_lines and key is not None and scale is not None:
        # Build scale notes in display range
        notes = build_scale_notes(
            key=key,
            scale=scale,  # type: ignore[arg-type]
            min_freq=float(freqs_disp[1]) if freqs_disp.size > 1 else 20.0,
            max_freq=float(max_freq),
        )
        for note in notes:
            if 0.0 < note.freq <= max_freq:
                ax.axvline(note.freq, linestyle="--", linewidth=0.5)

    ax.set_xlim(0.0, max_freq)
    return fig


def plot_oscilloscope(
    audio: np.ndarray,
    sr: int,
    tap_source: TapSource,
    duration: float = 0.02,
) -> plt.Figure:
    """
    Plot time-domain oscilloscope view of a short segment.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio buffer.
    sr : int
        Sample rate.
    tap_source : TapSource
        Label for plot title.
    duration : float
        Duration of displayed segment in seconds.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    segment = _select_segment(audio, sr, duration=duration, center=True)
    n = segment.shape[0]
    if n == 0:
        raise ValueError("Empty audio segment passed to plot_oscilloscope")

    t = np.arange(n) / float(sr)

    fig, ax = plt.subplots()
    ax.plot(t * 1000.0, segment)  # time in ms
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Oscilloscope — {tap_source}")
    return fig


def plot_phase_scope(
    audio: np.ndarray,
    sr: int,
    tap_source: TapSource,
) -> plt.Figure:
    """
    Plot a basic phase scope / Lissajous figure.

    For mono signals, we synthesize a pseudo stereo pair by delaying one copy slightly.
    For a stereo signal (shape (n_samples, 2)), uses L vs R directly.

    Parameters
    ----------
    audio : np.ndarray
        Mono or stereo buffer. Stereo expected as shape (n_samples, 2).
    sr : int
        Sample rate.
    tap_source : TapSource
        Label for plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim == 1:
        # Mono: pseudo stereo
        delay_samples = max(1, int(sr * 0.001))  # ~1ms delay
        if x.shape[0] <= delay_samples:
            left = x
            right = x
        else:
            left = x[:-delay_samples]
            right = x[delay_samples:]
    elif x.ndim == 2 and x.shape[1] == 2:
        left = x[:, 0]
        right = x[:, 1]
    else:
        raise ValueError("plot_phase_scope expects mono (1D) or stereo (2D, n_samples x 2)")

    # Select a segment
    seg_left = _select_segment(left, sr, duration=0.05, center=True)
    seg_right = _select_segment(right, sr, duration=0.05, center=True)

    # Match lengths
    n = min(seg_left.shape[0], seg_right.shape[0])
    seg_left = seg_left[:n]
    seg_right = seg_right[:n]

    fig, ax = plt.subplots()
    ax.plot(seg_left, seg_right, ".", markersize=1)
    ax.set_xlabel("Left")
    ax.set_ylabel("Right")
    ax.set_title(f"Phase Scope — {tap_source}")
    ax.set_aspect("equal", adjustable="box")
    return fig

