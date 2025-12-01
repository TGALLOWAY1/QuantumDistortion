import numpy as np


from typing import Tuple


from quantum_distortion.dsp.pipeline import process_audio
from quantum_distortion.dsp.analyses import avg_cents_offset_from_scale
from quantum_distortion.dsp.quantizer import midi_to_freq


def _sine_from_midi(midi: float, sr: int = 44100, seconds: float = 0.5) -> Tuple[np.ndarray, int]:
    freq = midi_to_freq(midi)
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    x = 0.2 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return x, sr


def test_apply_spectral_quantization_block_reduces_cents_offset() -> None:
    """
    Integration test: _apply_spectral_quantization_block should move a slightly detuned tone
    closer to the nearest scale tone in MIDI/cents space.
    """
    # A4 = 69 MIDI. Detune by +30 cents.
    base_midi = 69.0
    detuned_midi = base_midi + 0.3
    x, sr = _sine_from_midi(detuned_midi)

    key = "A"
    scale = "minor"

    # Baseline: cents offset of raw input
    in_avg_cents, _ = avg_cents_offset_from_scale(
        audio=x,
        sr=sr,
        key=key,
        scale=scale,  # type: ignore[arg-type]
        topn_peaks=1,
        min_db=-40.0,
    )
    assert np.isfinite(in_avg_cents)
    assert in_avg_cents > 5.0  # should be noticeably off

    # Apply spectral quantization with strong settings
    # Use process_audio with only pre-quantization enabled (no distortion, no post-quant)
    y, _ = process_audio(
        audio=x,
        sr=sr,
        key=key,
        scale=scale,
        snap_strength=1.0,
        smear=0.0,
        bin_smoothing=False,
        pre_quant=True,
        post_quant=False,
        distortion_mode="wavefold",
        distortion_params={"fold_amount": 1.0, "bias": 0.0, "drive": 1.0, "warmth": 0.5},
        limiter_on=False,
        dry_wet=1.0,
    )

    out_avg_cents, _ = avg_cents_offset_from_scale(
        audio=y,
        sr=sr,
        key=key,
        scale=scale,  # type: ignore[arg-type]
        topn_peaks=1,
        min_db=-40.0,
    )
    assert np.isfinite(out_avg_cents)

    # Quantization should reduce the average cents offset
    # Note: Due to STFT resolution limits and phase reconstruction challenges,
    # the reduction may be small, but we should see some improvement
    assert out_avg_cents < in_avg_cents, f"Quantization should reduce cents offset, but {out_avg_cents:.2f} >= {in_avg_cents:.2f}"

