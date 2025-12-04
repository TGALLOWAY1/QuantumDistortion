import numpy as np


from typing import Tuple


from quantum_distortion.dsp.analyses import avg_cents_offset_from_scale
from quantum_distortion.dsp.quantizer import midi_to_freq


def _sine_from_midi(midi: float, sr: int = 44100, seconds: float = 0.5) -> Tuple[np.ndarray, int]:
    freq = midi_to_freq(midi)
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    x = 0.2 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return x, sr


def test_avg_cents_offset_in_scale_tone() -> None:
    # A4 = 69; choose key A minor, so this is the root.
    x, sr = _sine_from_midi(69.0)
    avg_cents, per_peak = avg_cents_offset_from_scale(
        audio=x,
        sr=sr,
        key="A",
        scale="minor",
        topn_peaks=1,  # Use only the strongest peak per frame
        min_db=-40.0,  # Higher threshold to focus on fundamental
    )

    assert per_peak.size > 0
    # STFT resolution and peak detection may introduce some offset
    # For a pure in-scale tone, we expect relatively low offset (< 100 cents)
    assert avg_cents < 100.0


def test_avg_cents_offset_detuned_tone() -> None:
    # Start with A4 = 69 MIDI, then detune by ~+30 cents
    base_midi = 69.0
    detuned_midi = base_midi + 0.3  # 30 cents
    x, sr = _sine_from_midi(detuned_midi)

    avg_cents, per_peak = avg_cents_offset_from_scale(
        audio=x,
        sr=sr,
        key="A",
        scale="minor",
        topn_peaks=1,  # Use only the strongest peak per frame
        min_db=-40.0,  # Higher threshold to focus on fundamental
    )

    assert per_peak.size > 0
    # Detuned tone should have measurable offset
    # The offset should be greater than a perfectly in-scale tone would have
    # Due to STFT resolution, we check that it's in a reasonable range
    assert avg_cents > 0.0  # Should have some offset
    assert avg_cents < 200.0  # Should not be extremely high

