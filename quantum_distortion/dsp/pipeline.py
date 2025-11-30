from __future__ import annotations


from typing import Any, Dict, Tuple, Union


import numpy as np


from quantum_distortion.config import (
    DEFAULT_KEY,
    DEFAULT_SCALE,
    DEFAULT_SNAP_STRENGTH,
    DEFAULT_SMEAR,
    DEFAULT_BIN_SMOOTHING,
    DEFAULT_DISTORTION_MODE,
    DEFAULT_LIMITER_ON,
    DEFAULT_LIMITER_CEILING_DB,
    DEFAULT_DRY_WET,
    DEFAULT_SAMPLE_RATE,
)
from quantum_distortion.dsp.quantizer import quantize_spectrum
from quantum_distortion.dsp.distortion import apply_distortion
from quantum_distortion.dsp.limiter import peak_limiter
from quantum_distortion.dsp.stft_utils import stft_mono, istft_mono


# Default STFT configuration for MVP
N_FFT = 2048
HOP_LENGTH = N_FFT // 4
WINDOW = "hann"
CENTER = True


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Ensure mono float32 audio. If stereo/multi-channel, downmix to mono by averaging.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim == 1:
        return x.astype(np.float32)
    if x.ndim == 2:
        # shape: (n_samples, n_channels)
        return np.mean(x, axis=1).astype(np.float32)
    raise ValueError("Unsupported audio shape; expected 1D or 2D array")


def _apply_spectral_quantization_block(
    audio: np.ndarray,
    sr: int,
    key: str,
    scale: str,
    snap_strength: float,
    smear: float,
    bin_smoothing: bool,
) -> np.ndarray:
    """
    Helper: STFT → per-frame quantize_spectrum → iSTFT.
    """
    # Compute STFT
    S, freqs = stft_mono(
        audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=WINDOW,
        center=CENTER,
    )

    mags = np.abs(S)
    phases = np.angle(S)

    # Quantize each frame independently
    for t in range(S.shape[1]):
        frame_mags = mags[:, t]
        frame_phases = phases[:, t]

        new_mags, new_phases = quantize_spectrum(
            mags=frame_mags,
            phases=frame_phases,
            freqs=freqs,
            key=key,
            scale=scale,  # type: ignore[arg-type]
            snap_strength=snap_strength,
            smear=smear,
            bin_smoothing=bin_smoothing,
        )

        mags[:, t] = new_mags
        phases[:, t] = new_phases

    # Reconstruct complex STFT
    S_q = mags * np.exp(1j * phases)

    # Inverse STFT
    y = istft_mono(
        S_q,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=WINDOW,
        length=len(audio),
        center=CENTER,
    )
    return y.astype(np.float32)


def process_audio(
    audio: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    key: str = DEFAULT_KEY,
    scale: str = DEFAULT_SCALE,
    snap_strength: float = DEFAULT_SNAP_STRENGTH,
    smear: float = DEFAULT_SMEAR,
    bin_smoothing: bool = DEFAULT_BIN_SMOOTHING,
    pre_quant: bool = True,
    post_quant: bool = True,
    distortion_mode: str = DEFAULT_DISTORTION_MODE,
    distortion_params: Union[Dict[str, Any], None] = None,
    limiter_on: bool = DEFAULT_LIMITER_ON,
    limiter_ceiling_db: float = DEFAULT_LIMITER_CEILING_DB,
    dry_wet: float = DEFAULT_DRY_WET,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Main offline processing entry point.

    Pipeline:
        input
          → (optional) spectral pre-quantization
          → time-domain distortion
          → (optional) spectral post-quantization
          → (optional) limiter
          → dry/wet mix with input
    """
    if distortion_params is None:
        distortion_params = {}

    x_in = _ensure_mono(audio)
    n_samples = x_in.shape[0]

    # --- Tap: input ---
    tap_input = x_in.copy()

    # --- Stage A: Pre-Quantization ---
    if pre_quant and snap_strength > 0.0:
        x_pre = _apply_spectral_quantization_block(
            x_in,
            sr=sr,
            key=key,
            scale=scale,
            snap_strength=snap_strength,
            smear=smear,
            bin_smoothing=bin_smoothing,
        )
    else:
        x_pre = x_in.copy()

    tap_pre_quant = x_pre.copy()

    # --- Stage B: Time-Domain Distortion ---
    mode = distortion_mode or "wavefold"

    fold_amount = float(distortion_params.get("fold_amount", 1.0))
    bias = float(distortion_params.get("bias", 0.0))
    drive = float(distortion_params.get("drive", 1.0))
    warmth = float(distortion_params.get("warmth", 0.5))

    x_post_dist = apply_distortion(
        x_pre,
        mode=mode,  # type: ignore[arg-type]
        fold_amount=fold_amount,
        bias=bias,
        drive=drive,
        warmth=warmth,
    )

    tap_post_dist = x_post_dist.copy()

    # --- Stage C: Post-Quantization ---
    if post_quant and snap_strength > 0.0:
        x_post_quant = _apply_spectral_quantization_block(
            x_post_dist,
            sr=sr,
            key=key,
            scale=scale,
            snap_strength=snap_strength,
            smear=smear,
            bin_smoothing=bin_smoothing,
        )
    else:
        x_post_quant = x_post_dist.copy()

    # --- Stage D: Limiter ---
    if limiter_on:
        x_limited, _gain = peak_limiter(
            x_post_quant,
            sr=sr,
            ceiling_db=limiter_ceiling_db,
            lookahead_ms=5.0,
            release_ms=30.0,
        )
    else:
        x_limited = x_post_quant.copy()

    # --- Dry/Wet Mix ---
    dry_wet = float(np.clip(dry_wet, 0.0, 1.0))
    x_out = (dry_wet * x_limited) + ((1.0 - dry_wet) * tap_input)
    x_out = x_out.astype(np.float32)

    # Ensure output length matches input
    if x_out.shape[0] != n_samples:
        if x_out.shape[0] > n_samples:
            x_out = x_out[:n_samples]
        else:
            pad = np.zeros(n_samples - x_out.shape[0], dtype=np.float32)
            x_out = np.concatenate([x_out, pad], axis=0)

    tap_output = x_out.copy()

    taps = {
        "input": tap_input,
        "pre_quant": tap_pre_quant,
        "post_dist": tap_post_dist,
        "output": tap_output,
    }
    return x_out, taps
