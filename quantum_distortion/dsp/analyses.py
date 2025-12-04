from __future__ import annotations


from typing import Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


import numpy as np
import librosa


from quantum_distortion.dsp.quantizer import (
    freq_to_midi,
    build_scale_notes,
)


ScaleName = Literal["major", "minor", "pentatonic", "dorian", "mixolydian", "harmonic_minor"]


def _nearest_scale_midi(
    midi_value: float,
    key: str,
    scale: ScaleName,
) -> float:
    """
    Given a continuous MIDI value, find the nearest scale tone (in MIDI)
    for the specified key/scale.
    """
    if not np.isfinite(midi_value):
        return np.nan

    # Build a wide range of scale tones in MIDI space
    # We'll search by frequency, but it's convenient to use the existing helper.
    # Choose a frequency corresponding to this midi_value, then build scale notes around it.
    from quantum_distortion.dsp.quantizer import midi_to_freq

    freq = midi_to_freq(midi_value)
    min_freq = max(20.0, freq / 4.0)
    max_freq = min(20000.0, freq * 4.0)

    notes = build_scale_notes(
        key=key,
        scale=scale,
        min_freq=float(min_freq),
        max_freq=float(max_freq),
    )
    if not notes:
        return np.nan

    note_midis = np.array([n.midi for n in notes], dtype=float)
    idx = int(np.argmin(np.abs(note_midis - midi_value)))
    return float(note_midis[idx])


def avg_cents_offset_from_scale(
    audio: np.ndarray,
    sr: int,
    key: str,
    scale: ScaleName,
    frame_length: int = 2048,
    hop_length: Union[int, None] = None,
    topn_peaks: int = 3,
    min_db: float = -60.0,
) -> Tuple[float, np.ndarray]:
    """
    Estimate how tightly the audio's prominent spectral peaks align to the given scale.

    Approach:
    - Compute STFT magnitude.
    - For each frame:
      - Find top-N spectral peaks above min_db threshold.
      - Convert their frequencies to MIDI.
      - For each, find nearest scale tone in MIDI.
      - Compute cents offset: 100 * (midi - nearest_scale_midi).
    - Return the average absolute cents offset across all considered peaks,
      plus an array of per-peak absolute offsets (for debugging).

    Returns
    -------
    avg_abs_cents : float
        Mean absolute cents deviation across all peaks.
    per_peak_abs_cents : np.ndarray
        1D array of absolute cents values for each considered peak.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim != 1:
        raise ValueError("avg_cents_offset_from_scale expects mono (1D) audio")

    if hop_length is None:
        hop_length = frame_length // 4

    # STFT
    S = librosa.stft(
        y=x,
        n_fft=frame_length,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    mags = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    # Convert to dB for thresholding
    eps = 1e-12
    mags_db = 20.0 * np.log10(np.maximum(mags, eps))

    per_peak_abs_cents: list[float] = []

    for t in range(mags_db.shape[1]):
        frame_db = mags_db[:, t]

        # Skip silent frames
        if np.max(frame_db) < min_db:
            continue

        # Find candidate peaks: sort by magnitude
        idx_sorted = np.argsort(frame_db)[::-1]  # descending
        count = 0
        for idx in idx_sorted:
            if frame_db[idx] < min_db:
                break
            freq = float(freqs[idx])
            if freq <= 0.0:
                continue

            midi_est = freq_to_midi(freq)
            if not np.isfinite(midi_est):
                continue

            scale_midi = _nearest_scale_midi(midi_est, key=key, scale=scale)
            if not np.isfinite(scale_midi):
                continue

            cents = 100.0 * (midi_est - scale_midi)
            per_peak_abs_cents.append(abs(float(cents)))

            count += 1
            if count >= topn_peaks:
                break

    if not per_peak_abs_cents:
        return float("nan"), np.array([], dtype=float)

    per_peak_arr = np.array(per_peak_abs_cents, dtype=float)
    avg_abs_cents = float(np.mean(per_peak_arr))
    return avg_abs_cents, per_peak_arr

