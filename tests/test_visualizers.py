import numpy as np
import matplotlib.figure as mpl_fig


from typing import Tuple


from quantum_distortion.ui.visualizers import (
    plot_spectrum,
    plot_oscilloscope,
    plot_phase_scope,
)


def _make_sine(freq: float, sr: int = 44100, seconds: float = 0.1) -> Tuple[np.ndarray, int]:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    x = 0.1 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return x, sr


def test_plot_spectrum_returns_figure() -> None:
    x, sr = _make_sine(440.0)
    fig = plot_spectrum(
        audio=x,
        sr=sr,
        tap_source="input",
        key="A",
        scale="minor",
        show_scale_lines=True,
    )
    assert isinstance(fig, mpl_fig.Figure)


def test_plot_oscilloscope_returns_figure() -> None:
    x, sr = _make_sine(220.0)
    fig = plot_oscilloscope(
        audio=x,
        sr=sr,
        tap_source="post_dist",
        duration=0.02,
    )
    assert isinstance(fig, mpl_fig.Figure)


def test_plot_phase_scope_mono_returns_figure() -> None:
    x, sr = _make_sine(110.0)
    fig = plot_phase_scope(
        audio=x,
        sr=sr,
        tap_source="output",
    )
    assert isinstance(fig, mpl_fig.Figure)


def test_plot_phase_scope_stereo_returns_figure() -> None:
    # Simple stereo: left sine, right cosine
    sr = 44100
    t = np.linspace(0.0, 0.1, int(sr * 0.1), endpoint=False)
    left = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    right = 0.1 * np.cos(2.0 * np.pi * 440.0 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)

    fig = plot_phase_scope(
        audio=stereo,
        sr=sr,
        tap_source="pre_quant",
    )
    assert isinstance(fig, mpl_fig.Figure)

