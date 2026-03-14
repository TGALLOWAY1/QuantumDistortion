from typing import Tuple

import numpy as np

from quantum_distortion.dsp.pipeline import process_audio
from quantum_distortion.dsp.quantizer import midi_to_freq


def _sine_from_midi(midi: float, sr: int = 48_000, seconds: float = 1.0) -> Tuple[np.ndarray, int]:
    freq = midi_to_freq(midi)
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    x = 0.2 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return x, sr


def _dominant_freq(
    audio: np.ndarray,
    sr: int,
    min_hz: float = 80.0,
    max_hz: float = 1_200.0,
    start_seconds: float = 0.25,
    end_seconds: float = 0.85,
) -> float:
    start = int(start_seconds * sr)
    end = min(len(audio), int(end_seconds * sr))
    segment = np.asarray(audio[start:end], dtype=np.float64)
    if len(segment) == 0:
        raise AssertionError("dominant frequency analysis segment was empty")

    window = np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(segment * window))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(mask):
        raise AssertionError("frequency analysis mask removed all bins")

    masked_spectrum = spectrum[mask]
    masked_freqs = freqs[mask]
    return float(masked_freqs[int(np.argmax(masked_spectrum))])


def test_autotune_v1_process_audio_moves_detuned_tone_toward_scale_note() -> None:
    target_freq = midi_to_freq(69.0)  # A4
    detuned_midi = 69.35  # ~35 cents sharp
    x, sr = _sine_from_midi(detuned_midi)

    y, _ = process_audio(
        audio=x,
        sr=sr,
        key="A",
        scale="minor",
        quantize_mode="autotune_v1",
        snap_strength=1.0,
        smear=0.0,
        bin_smoothing=False,
        pre_quant=True,
        post_quant=False,
        distortion_mode="wavefold",
        distortion_params={"fold_amount": 1.0, "bias": 0.0, "drive": 1.0, "warmth": 0.5},
        limiter_on=False,
        dry_wet=1.0,
        sub_enabled=False,
    )

    in_freq = _dominant_freq(x, sr)
    out_freq = _dominant_freq(y, sr)

    assert abs(out_freq - target_freq) < abs(in_freq - target_freq)
    assert abs(out_freq - target_freq) < 6.0
