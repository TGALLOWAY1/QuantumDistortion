import numpy as np

from quantum_distortion.dsp.autotune import (
    AutotuneV1Config,
    build_pitch_track,
    generate_sub_layer,
)
from quantum_distortion.dsp.quantizer import midi_to_freq


def _dominant_freq(
    audio: np.ndarray,
    sr: int,
    min_hz: float,
    max_hz: float,
) -> float:
    audio = np.asarray(audio, dtype=np.float64)
    window = np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    masked_spectrum = spectrum[mask]
    masked_freqs = freqs[mask]
    return float(masked_freqs[int(np.argmax(masked_spectrum))])


def test_generate_sub_layer_lands_on_configured_note() -> None:
    sr = 48_000
    seconds = 1.0
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    reference = 0.25 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)

    cfg = AutotuneV1Config(
        key="C",
        scale="major",
        sub_enabled=True,
        sub_source="root",
        sub_octave=2,
        sub_level=0.5,
    )

    sub = generate_sub_layer(reference, sr, cfg)
    dominant = _dominant_freq(sub, sr, min_hz=40.0, max_hz=120.0)
    expected = midi_to_freq(36.0)  # C2

    assert np.sqrt(np.mean(sub * sub)) > 0.01
    assert abs(dominant - expected) < 2.0


def test_build_pitch_track_rejects_broadband_noise() -> None:
    sr = 48_000
    rng = np.random.default_rng(1234)
    noise = (0.03 * rng.standard_normal(sr)).astype(np.float32)

    diagnostics = build_pitch_track(
        noise,
        sr,
        AutotuneV1Config(
            key="C",
            scale="major",
            sub_enabled=False,
            detector_frame_size=4096,
            detector_hop_size=512,
        ),
    )

    voiced_fraction = float(np.mean(diagnostics.voiced_mask))
    median_ratio_error = float(np.median(np.abs(diagnostics.ratio_track - 1.0)))

    assert voiced_fraction < 0.25
    assert median_ratio_error < 0.02
