from __future__ import annotations


from dataclasses import dataclass

from typing import Dict, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


import numpy as np


ScaleName = Literal["major", "minor", "pentatonic", "dorian", "mixolydian", "harmonic_minor"]


NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


SCALE_INTERVALS: Dict[ScaleName, Tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "pentatonic": (0, 2, 4, 7, 9),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
}


HARMONIC_WEIGHTS: Dict[str, float] = {
    "root": 1.0,
    "fifth": 0.8,
    "third": 0.7,
    "seventh": 0.6,
    "other": 0.5,
}


@dataclass(frozen=True)
class ScaleNote:
    midi: int
    freq: float
    role: str  # "root", "fifth", "third", "seventh", "other"
    weight: float


def note_name_to_pitch_class(name: str) -> int:
    """
    Convert note name like "C", "C#", "Db", "A", "Bb" to pitch class 0-11.

    Supports sharps (#) and flats (b); flats are normalized to sharps.
    """
    name = name.strip().upper()
    name = name.replace("DB", "C#").replace("EB", "D#").replace("GB", "F#").replace("AB", "G#").replace("BB", "A#")
    if name not in NOTE_NAMES_SHARP:
        raise ValueError(f"Unsupported key name: {name}")
    return NOTE_NAMES_SHARP.index(name)


def midi_to_freq(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def freq_to_midi(freq: float) -> float:
    if freq <= 0.0:
        return -np.inf
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def _classify_role(pitch_class: int, root_pc: int) -> str:
    """
    Classify a scale degree role relative to the root.
    """
    interval = (pitch_class - root_pc) % 12
    if interval == 0:
        return "root"
    if interval == 7:
        return "fifth"
    if interval in (3, 4):
        return "third"
    if interval in (10, 11):
        return "seventh"
    return "other"


def build_scale_notes(
    key: str,
    scale: ScaleName,
    min_freq: float,
    max_freq: float,
) -> list[ScaleNote]:
    """
    Build a list of scale notes (across octaves) that span [min_freq, max_freq].
    """
    root_pc = note_name_to_pitch_class(key)
    intervals = SCALE_INTERVALS[scale]

    # Choose a broad MIDI range so we cover audio band
    min_midi = int(np.floor(freq_to_midi(max(min_freq, 20.0)))) - 12
    max_midi = int(np.ceil(freq_to_midi(min(max_freq, 22050.0)))) + 12

    notes: list[ScaleNote] = []
    for midi in range(min_midi, max_midi + 1):
        pitch_class = midi % 12
        if (pitch_class - root_pc) % 12 in intervals:
            freq = midi_to_freq(float(midi))
            if freq < min_freq * 0.5 or freq > max_freq * 2.0:
                continue
            role = _classify_role(pitch_class, root_pc)
            weight = HARMONIC_WEIGHTS[role]
            notes.append(ScaleNote(midi=midi, freq=freq, role=role, weight=weight))
    return notes


def build_target_bins_for_freqs(
    freqs: np.ndarray,
    key: str,
    scale: ScaleName,
) -> np.ndarray:
    """
    For each frequency bin, compute the target bin index it should be attracted to,
    based on nearest in-scale note and harmonic weighting.


    Returns
    -------
    target_bins: np.ndarray[int] of shape (len(freqs),)
    """
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1:
        raise ValueError("freqs must be 1D array")

    valid = np.isfinite(freqs) & (freqs > 0.0)
    if not np.any(valid):
        return np.arange(len(freqs), dtype=int)

    min_freq = float(np.min(freqs[valid]))
    max_freq = float(np.max(freqs[valid]))
    scale_notes = build_scale_notes(key, scale, min_freq, max_freq)

    if not scale_notes:
        # Fallback: identity mapping
        return np.arange(len(freqs), dtype=int)

    note_freqs = np.array([n.freq for n in scale_notes], dtype=float)
    note_weights = np.array([n.weight for n in scale_notes], dtype=float)

    # Avoid division by zero
    note_weights = np.clip(note_weights, 1e-3, None)

    target_bins = np.arange(len(freqs), dtype=int)

    for i, f in enumerate(freqs):
        if not valid[i]:
            continue

        # Weighted distance: smaller is better
        df = np.abs(note_freqs - f)
        # cost ~ distance / weight (root/fifths more attractive)
        cost = df / note_weights
        idx = int(np.argmin(cost))
        target_freq = note_freqs[idx]

        # Find nearest FFT bin to target_freq
        bin_idx = int(np.argmin(np.abs(freqs - target_freq)))
        target_bins[i] = bin_idx

    return target_bins


def quantize_spectrum(
    mags: np.ndarray,
    phases: np.ndarray,
    freqs: np.ndarray,
    key: str,
    scale: ScaleName,
    snap_strength: float,
    smear: float,
    bin_smoothing: bool,
    smear_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply spectral quantization to a single FFT frame.


    Parameters
    ----------
    mags : np.ndarray
        Magnitude spectrum (1D).
    phases : np.ndarray
        Phase spectrum (1D), same shape as mags.
    freqs : np.ndarray
        Frequency per bin (1D).
    key : str
        Musical key (e.g. "C", "D#", "A", "Bb").
    scale : ScaleName
        Scale name (e.g. "major", "minor").
    snap_strength : float
        0.0 → no movement, 1.0 → full attraction to target bins.
    smear : float
        0.0 → all moved energy goes to target bin,
        1.0 → all moved energy is smeared around target within smear_radius.
    bin_smoothing : bool
        If True, apply simple moving-average smoothing to magnitudes after quantization.
    smear_radius : int
        Number of bins on each side of target bin to smear into.


    Returns
    -------
    new_mags, new_phases
    """
    mags = np.asarray(mags, dtype=float)
    phases = np.asarray(phases, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    if mags.shape != phases.shape or mags.shape != freqs.shape:
        raise ValueError("mags, phases, and freqs must have the same shape")

    if mags.ndim != 1:
        raise ValueError("quantize_spectrum expects 1D arrays")

    snap_strength = float(np.clip(snap_strength, 0.0, 1.0))
    smear = float(np.clip(smear, 0.0, 1.0))

    # Early out: no snapping, no smoothing
    if snap_strength <= 0.0 and not bin_smoothing:
        return mags.copy(), phases.copy()

    target_bins = build_target_bins_for_freqs(freqs, key, scale)

    new_mags = mags.copy()

    # Move energy toward targets
    for i in range(len(mags)):
        mag = mags[i]
        if mag <= 0.0:
            continue

        target_bin = target_bins[i]
        if target_bin < 0 or target_bin >= len(mags):
            continue

        energy_to_move = mag * snap_strength
        if energy_to_move <= 0.0:
            continue

        # Remove from source
        new_mags[i] -= energy_to_move

        # Split into direct target + smeared neighbors
        base_energy = energy_to_move * (1.0 - smear)
        smear_energy = energy_to_move * smear

        # Direct target
        new_mags[target_bin] += base_energy

        if smear_energy > 0.0 and smear_radius > 0:
            # Simple Gaussian-like kernel over [target-r, target+r]
            indices = np.arange(
                max(0, target_bin - smear_radius),
                min(len(mags), target_bin + smear_radius + 1),
                dtype=int,
            )
            distances = np.abs(indices - target_bin).astype(float)
            # Gaussian-ish weights (center highest)
            sigma = max(1.0, smear_radius / 2.0)
            weights = np.exp(-0.5 * (distances / sigma) ** 2)
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights /= weights_sum
                new_mags[indices] += smear_energy * weights

    # Optional bin smoothing
    if bin_smoothing and len(new_mags) > 2:
        kernel = np.array([0.25, 0.5, 0.25], dtype=float)
        # 'same' convolution
        padded = np.pad(new_mags, (1, 1), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="same")[1:-1]
        new_mags = smoothed

    # Phases remain unchanged in MVP
    return new_mags, phases.copy()

