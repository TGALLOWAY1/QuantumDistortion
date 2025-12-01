from __future__ import annotations


from dataclasses import dataclass

from typing import Dict, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


import numpy as np
from scipy.ndimage import convolve1d

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a dummy decorator if Numba is not available
    class numba:
        @staticmethod
        def njit(*args, **kwargs):
            def decorator(func):
                return func
            return decorator


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
    
    Vectorized implementation: operates on all bins at once using broadcasting.

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

    # Vectorized computation: for each freq bin, find nearest scale note
    # Shape: (n_freqs, n_notes) - distance from each freq to each note
    freqs_2d = freqs[:, np.newaxis]  # Shape: (n_freqs, 1)
    note_freqs_2d = note_freqs[np.newaxis, :]  # Shape: (1, n_notes)
    
    # Weighted distance: smaller is better
    # Broadcasting: (n_freqs, 1) - (1, n_notes) -> (n_freqs, n_notes)
    df = np.abs(freqs_2d - note_freqs_2d)
    # cost ~ distance / weight (root/fifths more attractive)
    # Broadcasting: (n_freqs, n_notes) / (1, n_notes) -> (n_freqs, n_notes)
    cost = df / note_weights[np.newaxis, :]
    
    # Find index of minimum cost for each freq bin
    # Shape: (n_freqs,)
    note_indices = np.argmin(cost, axis=1)
    
    # Get target frequencies for each bin
    target_freqs = note_freqs[note_indices]
    
    # For each target frequency, find nearest FFT bin
    # Vectorized: compute distance from all freqs to each target_freq
    # Shape: (n_freqs, n_freqs) - distance matrix
    target_freqs_2d = target_freqs[:, np.newaxis]  # Shape: (n_freqs, 1)
    freqs_2d = freqs[np.newaxis, :]  # Shape: (1, n_freqs)
    dist_matrix = np.abs(target_freqs_2d - freqs_2d)  # Shape: (n_freqs, n_freqs)
    
    # Find nearest bin for each target frequency
    target_bins = np.argmin(dist_matrix, axis=1)  # Shape: (n_freqs,)
    
    # For invalid bins, keep identity mapping
    target_bins = np.where(valid, target_bins, np.arange(len(freqs), dtype=int))
    
    return target_bins


@numba.njit
def _apply_smear_numba(
    new_mags: np.ndarray,
    target_energy: np.ndarray,
    target_phase_sum: np.ndarray,
    smear_indices: np.ndarray,
    smear_targets: np.ndarray,
    smear_energies: np.ndarray,
    smear_phases: np.ndarray,
    smear_kernel: np.ndarray,
    smear_radius: int,
    n_bins: int,
) -> None:
    """
    Numba-accelerated smear operation: distributes energy around target bins.
    
    This function is JIT-compiled by Numba for performance. It operates in-place
    on the provided arrays. All operations must be Numba-compatible (no Python
    objects, only NumPy arrays and basic numeric types).
    
    Parameters
    ----------
    new_mags : np.ndarray
        Magnitude array to modify in-place, shape (n_bins,)
    target_energy : np.ndarray
        Energy accumulation array, shape (n_bins,)
    target_phase_sum : np.ndarray
        Complex phase accumulation array, shape (n_bins,)
    smear_indices : np.ndarray
        Source bin indices that need smearing, shape (n_smear,)
    smear_targets : np.ndarray
        Target bin indices for each source, shape (n_smear,)
    smear_energies : np.ndarray
        Energy to smear for each source, shape (n_smear,)
    smear_phases : np.ndarray
        Phase values for each source, shape (n_smear,)
    smear_kernel : np.ndarray
        Pre-computed Gaussian-like kernel, shape (kernel_size,)
    smear_radius : int
        Radius of smear operation
    n_bins : int
        Total number of frequency bins
    
    Note
    ----
    The first call to this function will include JIT compilation overhead (~0.1-1s),
    but subsequent calls will be significantly faster than the pure Python loop.
    """
    kernel_size = smear_kernel.shape[0]
    kernel_center = smear_radius
    
    for i in range(len(smear_indices)):
        src_idx = smear_indices[i]
        target_bin = smear_targets[i]
        energy = smear_energies[i]
        phase = smear_phases[i]
        
        # Create local kernel centered at target_bin
        start_idx = max(0, target_bin - smear_radius)
        end_idx = min(n_bins, target_bin + smear_radius + 1)
        local_size = end_idx - start_idx
        
        if local_size > 0:
            # Extract relevant portion of kernel
            kernel_start = max(0, smear_radius - target_bin)
            kernel_end = kernel_start + local_size
            
            # Compute local kernel (Numba-compatible operations)
            local_kernel_sum = 0.0
            for k in range(kernel_start, kernel_end):
                local_kernel_sum += smear_kernel[k]
            
            if local_kernel_sum > 0.0:
                # Distribute energy
                for j in range(local_size):
                    bin_idx = start_idx + j
                    kernel_idx = kernel_start + j
                    weight = smear_kernel[kernel_idx] / local_kernel_sum
                    local_energy = energy * weight
                    
                    new_mags[bin_idx] += local_energy
                    target_energy[bin_idx] += local_energy
                    # Complex phase contribution: e^(i*phase) = cos(phase) + i*sin(phase)
                    # Numba-compatible: construct complex from real and imaginary parts
                    phase_real = np.cos(phase)
                    phase_imag = np.sin(phase)
                    phase_complex = complex(phase_real, phase_imag)
                    target_phase_sum[bin_idx] += local_energy * phase_complex


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
    
    Vectorized implementation: all operations use NumPy array operations
    instead of Python loops.

    Parameters
    ----------
    mags : np.ndarray
        Magnitude spectrum (1D), shape (n_bins,).
    phases : np.ndarray
        Phase spectrum (1D), same shape as mags.
    freqs : np.ndarray
        Frequency per bin (1D), same shape as mags.
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
        Modified magnitude and phase arrays, same shape as input.
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
    n_bins = len(mags)

    # Vectorized energy movement
    # Compute energy to move from each source bin
    energy_to_move = mags * snap_strength
    # Mask for valid moves (positive energy, valid target bins)
    valid_mask = (energy_to_move > 0.0) & (target_bins >= 0) & (target_bins < n_bins)
    
    # Remove energy from source bins (vectorized)
    new_mags = mags.copy() - np.where(valid_mask, energy_to_move, 0.0)
    
    # Split energy into direct target and smear components
    base_energy = energy_to_move * (1.0 - smear)  # Direct to target
    smear_energy = energy_to_move * smear  # To be smeared
    
    # Track energy contributions for phase combination
    target_energy = np.zeros(n_bins, dtype=float)
    target_phase_sum = np.zeros(n_bins, dtype=complex)
    
    # Vectorized direct target accumulation using advanced indexing
    # For each source bin i, add base_energy[i] to target_bins[i]
    np.add.at(new_mags, target_bins[valid_mask], base_energy[valid_mask])
    np.add.at(target_energy, target_bins[valid_mask], base_energy[valid_mask])
    
    # Phase contributions: weighted by energy
    phase_contribs = base_energy[valid_mask] * np.exp(1j * phases[valid_mask])
    np.add.at(target_phase_sum, target_bins[valid_mask], phase_contribs)
    
    # Vectorized smear: distribute energy around target bins
    # NOTE: This uses Numba-accelerated function for the inner loop, as each source
    # bin has a different target center making full vectorization difficult.
    # The Numba JIT compilation happens on first call (~0.1-1s overhead), but
    # subsequent calls are significantly faster than pure Python loops.
    if smear > 0.0 and smear_radius > 0:
        # Build smear kernel: Gaussian-like, centered at 0
        kernel_size = 2 * smear_radius + 1
        kernel_center = smear_radius
        kernel_indices = np.arange(kernel_size, dtype=float) - kernel_center
        sigma = max(1.0, smear_radius / 2.0)
        smear_kernel = np.exp(-0.5 * (kernel_indices / sigma) ** 2)
        smear_kernel /= np.sum(smear_kernel)  # Normalize
        
        # Find all source bins that need smearing
        smear_mask = valid_mask & (smear_energy > 0.0)
        smear_indices = np.where(smear_mask)[0]
        
        if len(smear_indices) > 0:
            # Vectorized preparation: get all target bins and energies at once
            smear_targets = target_bins[smear_indices].astype(np.int32)
            smear_energies = smear_energy[smear_indices].astype(np.float64)
            smear_phases = phases[smear_indices].astype(np.float64)
            
            # Use Numba-accelerated function for smear distribution
            # This function is JIT-compiled and operates in-place on the arrays
            if NUMBA_AVAILABLE:
                _apply_smear_numba(
                    new_mags,
                    target_energy,
                    target_phase_sum,
                    smear_indices.astype(np.int32),
                    smear_targets,
                    smear_energies,
                    smear_phases,
                    smear_kernel.astype(np.float64),
                    np.int32(smear_radius),
                    np.int32(n_bins),
                )
            else:
                # Fallback to pure Python if Numba is not available
                # (This should not happen in production, but provides graceful degradation)
                for idx in range(len(smear_indices)):
                    src_idx = smear_indices[idx]
                    target_bin = smear_targets[idx]
                    energy = smear_energies[idx]
                    phase = smear_phases[idx]
                    
                    start_idx = max(0, target_bin - smear_radius)
                    end_idx = min(n_bins, target_bin + smear_radius + 1)
                    local_size = end_idx - start_idx
                    
                    if local_size > 0:
                        kernel_start = max(0, smear_radius - target_bin)
                        kernel_end = kernel_start + local_size
                        local_kernel = smear_kernel[kernel_start:kernel_end].copy()
                        local_kernel /= np.sum(local_kernel)
                        
                        local_indices = np.arange(start_idx, end_idx, dtype=int)
                        local_energy = energy * local_kernel
                        new_mags[local_indices] += local_energy
                        target_energy[local_indices] += local_energy
                        target_phase_sum[local_indices] += local_energy * np.exp(1j * phase)
    
    # Vectorized phase update: use weighted average of contributing phases
    phase_mask = target_energy > 0.0
    new_phases = phases.copy()
    new_phases[phase_mask] = np.angle(target_phase_sum[phase_mask])

    # Optional bin smoothing: use scipy.ndimage.convolve1d for efficiency
    if bin_smoothing and len(new_mags) > 2:
        # Smoothing kernel: [0.25, 0.5, 0.25] - simple moving average
        # This approximates the previous behavior of edge-padded convolution
        smoothing_kernel = np.array([0.25, 0.5, 0.25], dtype=float)
        new_mags = convolve1d(new_mags, smoothing_kernel, axis=0, mode="nearest")

    return new_mags, new_phases

