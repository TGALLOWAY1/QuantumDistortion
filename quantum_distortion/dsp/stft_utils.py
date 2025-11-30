from __future__ import annotations


from typing import Tuple, Union


import numpy as np
import librosa


def stft_mono(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: Union[int, None] = None,
    window: str = "hann",
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute complex STFT for a mono signal and return (S, freqs).

    S: shape (n_fft//2 + 1, n_frames), complex-valued
    freqs: shape (n_fft//2 + 1,), frequency per bin in Hz
    """
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1:
        raise ValueError("stft_mono expects mono (1D) audio")

    if hop_length is None:
        hop_length = n_fft // 4

    S = librosa.stft(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=center,
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
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
    Inverse STFT for a mono signal.

    S must be complex STFT matrix as produced by librosa.stft.
    """
    if hop_length is None:
        hop_length = n_fft // 4

    y = librosa.istft(
        S,
        hop_length=hop_length,
        window=window,
        length=length,
        center=center,
    )
    return y.astype(np.float32)

