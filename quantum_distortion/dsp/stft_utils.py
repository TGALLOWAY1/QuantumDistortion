from __future__ import annotations


from typing import Tuple, Union


import numpy as np
from scipy.signal import windows


def stft_mono(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: Union[int, None] = None,
    window: str = "hann",
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute complex STFT for a mono signal with strict OLA compliance.
    
    OLA Architecture:
    - hop_length is forced to n_fft // 4 (75% overlap)
    - Analysis window: Hann window (sym=False) for OLA compatibility
    - Each frame is windowed before FFT: frame = window * audio_segment
    - This ensures proper overlap-add reconstruction in istft_mono()
    
    Parameters
    ----------
    audio : np.ndarray
        Mono (1D) audio signal
    sr : int
        Sample rate in Hz
    n_fft : int, optional
        FFT size. Default 2048.
    hop_length : int, optional
        Ignored - always set to n_fft // 4 for OLA compliance.
    window : str, optional
        Ignored - always uses Hann window for OLA compliance.
    center : bool, optional
        If True, pad audio symmetrically before processing (librosa-compatible).
        Default True.
    
    Returns
    -------
    S : np.ndarray
        Complex STFT matrix, shape (n_fft//2 + 1, n_frames)
    freqs : np.ndarray
        Frequency array in Hz, shape (n_fft//2 + 1,)
    """
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1:
        raise ValueError("stft_mono expects mono (1D) audio")

    # OLA Constraint: hop_length must equal n_fft // 4
    # This ensures 75% overlap, which is compatible with Hann window for OLA
    hop_length = n_fft // 4
    
    # OLA Constraint: Use Hann window (sym=False) for analysis
    # sym=False ensures the window is periodic (not symmetric), which is required
    # for proper OLA reconstruction when hop_length = n_fft // 4
    analysis_window = windows.hann(n_fft, sym=False)
    
    # Handle center padding (librosa-compatible behavior)
    if center:
        # Pad with n_fft // 2 zeros on both sides
        pad_width = n_fft // 2
        audio = np.pad(audio, pad_width, mode='constant', constant_values=0.0)
    
    # Calculate number of frames
    n_samples = len(audio)
    # Ensure at least one frame even if input is shorter than n_fft
    n_frames = max(1, 1 + (n_samples - n_fft) // hop_length)
    
    # Pre-allocate STFT output: (n_fft//2 + 1) frequency bins, n_frames time frames
    n_bins = n_fft // 2 + 1
    S = np.zeros((n_bins, n_frames), dtype=np.complex128)
    
    # Process each frame with explicit windowing
    # OLA Constraint: Each frame is windowed before FFT
    # frame = analysis_window * audio_segment
    for t in range(n_frames):
        start_idx = t * hop_length
        end_idx = start_idx + n_fft
        
        if end_idx > n_samples:
            # Last frame: pad with zeros if needed
            frame = np.zeros(n_fft, dtype=audio.dtype)
            frame[:n_samples - start_idx] = audio[start_idx:]
        else:
            frame = audio[start_idx:end_idx]
        
        # OLA Constraint: Apply analysis window before FFT
        # This ensures proper overlap-add reconstruction
        windowed_frame = frame * analysis_window
        
        # Compute FFT (rfft for real input, returns n_fft//2 + 1 bins)
        S[:, t] = np.fft.rfft(windowed_frame, n=n_fft)
    
    # Calculate frequency bins
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
    
    return S, freqs


def istft_mono(
    S: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: Union[int, None] = None,
    window: str = "hann",
    length: Union[int, None] = None,
    center: bool = True,
) -> np.ndarray:
    """
    Inverse STFT for a mono signal with strict OLA compliance.
    
    OLA Architecture:
    - hop_length is forced to n_fft // 4 (75% overlap)
    - Synthesis window: Same Hann window as analysis (OLA-compatible)
    - Each IFFT frame is windowed: frame = synthesis_window * ifft_result
    - Overlap-add: Sum overlapping windowed frames to reconstruct signal
    - With hop_length = n_fft // 4 and Hann window, OLA condition is satisfied
    
    Parameters
    ----------
    S : np.ndarray
        Complex STFT matrix, shape (n_fft//2 + 1, n_frames)
    sr : int
        Sample rate in Hz (used for frequency calculation, not reconstruction)
    n_fft : int, optional
        FFT size. Must match the n_fft used in stft_mono(). Default 2048.
    hop_length : int, optional
        Ignored - always set to n_fft // 4 for OLA compliance.
    window : str, optional
        Ignored - always uses Hann window for OLA compliance.
    length : int, optional
        Desired output length in samples. If None, length is inferred from STFT.
    center : bool, optional
        If True, remove symmetric padding added during STFT (librosa-compatible).
        Default True.
    
    Returns
    -------
    y : np.ndarray
        Reconstructed audio signal, float32, shape (length,)
    """
    # OLA Constraint: hop_length must equal n_fft // 4
    hop_length = n_fft // 4
    
    # OLA Constraint: Use same Hann window for synthesis as analysis
    # This ensures proper overlap-add reconstruction
    synthesis_window = windows.hann(n_fft, sym=False)
    
    # Get dimensions
    n_bins, n_frames = S.shape
    if n_bins != n_fft // 2 + 1:
        raise ValueError(f"STFT matrix has {n_bins} bins, expected {n_fft // 2 + 1}")
    
    # Calculate output length for overlap-add buffer
    # The natural ISTFT output length is determined by the number of frames
    natural_output_length = (n_frames - 1) * hop_length + n_fft
    
    # If length is specified, we need to ensure output is at least that long
    # (accounting for center padding removal if needed)
    if length is not None:
        if center:
            # After removing center padding, we need at least 'length' samples
            # So before removal, we need at least length + 2*pad_width
            pad_width = n_fft // 2
            min_output_before_pad_removal = length + 2 * pad_width
            output_length = max(natural_output_length, min_output_before_pad_removal)
        else:
            output_length = max(natural_output_length, length)
    else:
        output_length = natural_output_length
    
    # Pre-allocate output buffer for overlap-add
    # OLA Constraint: Use overlap-add accumulator
    y = np.zeros(output_length, dtype=np.float64)
    
    # OLA Constraint: Compute normalization factor (sum of overlapping windows squared)
    # For perfect reconstruction with same window for analysis/synthesis, we need:
    # sum_t (w[n - t*hop]^2) = constant
    # If not constant, normalize by this sum at each sample
    window_sum_sq = np.zeros(output_length, dtype=np.float64)
    for t in range(n_frames):
        start_idx = t * hop_length
        end_idx = start_idx + n_fft
        
        if end_idx <= output_length:
            window_sum_sq[start_idx:end_idx] += synthesis_window ** 2
        else:
            available = output_length - start_idx
            window_sum_sq[start_idx:] += synthesis_window[:available] ** 2
    
    # Avoid division by zero: use small epsilon where window_sum_sq is zero
    window_sum_sq = np.maximum(window_sum_sq, 1e-10)
    
    # Process each frame with explicit windowing and overlap-add
    # OLA Constraint: Each IFFT frame is windowed, then overlap-added
    for t in range(n_frames):
        # Compute IFFT (irfft for real output)
        # Note: irfft automatically scales by n_fft, so no additional scaling needed
        frame = np.fft.irfft(S[:, t], n=n_fft)
        
        # OLA Constraint: Apply synthesis window to frame
        # This ensures proper overlap-add reconstruction
        windowed_frame = frame * synthesis_window
        
        # OLA Constraint: Overlap-add into output buffer
        # With hop_length = n_fft // 4, frames overlap by 75%
        start_idx = t * hop_length
        end_idx = start_idx + n_fft
        
        if end_idx <= output_length:
            y[start_idx:end_idx] += windowed_frame
        else:
            # Last frame: truncate if needed
            available = output_length - start_idx
            y[start_idx:] += windowed_frame[:available]
    
    # OLA Constraint: Normalize by sum of overlapping windows squared for perfect reconstruction
    # This ensures that samples with more overlapping frames are properly scaled
    # The normalization accounts for the fact that multiple windowed frames contribute to each sample
    y = y / window_sum_sq
    
    # Handle center padding removal (librosa-compatible behavior)
    # This must be done BEFORE length adjustment, since the length parameter
    # typically refers to the original input length (before padding)
    if center:
        pad_width = n_fft // 2
        if len(y) > 2 * pad_width:
            y = y[pad_width:-pad_width]
        # If output is too short, keep as-is (edge case)
    
    # Ensure output matches requested length
    # Note: length parameter typically refers to original input length (after padding removal)
    if length is not None and len(y) != length:
        if len(y) > length:
            y = y[:length]
        else:
            # Pad with zeros if shorter (should be rare with proper OLA)
            y = np.pad(y, (0, length - len(y)), mode='constant', constant_values=0.0)
    
    return y.astype(np.float32)

