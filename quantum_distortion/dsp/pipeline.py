from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union
import os
import time


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
    PREVIEW_ENABLED_DEFAULT,
    PREVIEW_MAX_SECONDS,
)
from quantum_distortion.dsp.quantizer import quantize_spectrum
from quantum_distortion.dsp.distortion import apply_distortion
from quantum_distortion.dsp.limiter import peak_limiter
from quantum_distortion.dsp.stft_utils import stft_mono, istft_mono
from quantum_distortion.dsp.crossover import linkwitz_riley_split, estimate_filter_group_delay_samples
from quantum_distortion.dsp.saturation import soft_tube, make_mono_lowband


# OLA-Compliant STFT Configuration
# The STFT/ISTFT functions in stft_utils.py enforce strict OLA architecture:
# - hop_length is always n_fft // 4 (75% overlap) - enforced by OLA requirements
# - Window is always Hann (sym=False) - enforced for OLA compatibility
# - These parameters are passed for clarity but are ignored by stft_utils functions
# - n_fft=2048 provides good frequency resolution without excessive computation
N_FFT_DEFAULT = 2048
HOP_LENGTH_DEFAULT = N_FFT_DEFAULT // 4  # 512 (enforced by OLA architecture)
WINDOW_DEFAULT = "hann"  # Ignored - OLA architecture enforces Hann window
CENTER_DEFAULT = True  # Controls center padding (librosa-compatible behavior)

# Legacy names for backward compatibility
N_FFT = N_FFT_DEFAULT
HOP_LENGTH = HOP_LENGTH_DEFAULT
WINDOW = WINDOW_DEFAULT
CENTER = CENTER_DEFAULT


@dataclass
class RenderTiming:
    """Timing measurements for audio rendering pipeline."""
    load: float = 0.0
    stft: float = 0.0
    proc: float = 0.0
    istft: float = 0.0
    save: float = 0.0
    total: float = 0.0


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


def _apply_spectral_quantization_to_stft(
    S: np.ndarray,
    freqs: np.ndarray,
    key: str,
    scale: str,
    snap_strength: float,
    smear: float,
    bin_smoothing: bool,
    timing: Union[RenderTiming, None] = None,
) -> np.ndarray:
    """
    Apply spectral quantization directly to an existing STFT matrix S.
    
    This avoids recomputing STFT - all spectral operations are performed
    in-place on the provided S matrix.
    
    Args:
        S: Complex STFT matrix (shape: [freq_bins, frames])
        freqs: Frequency array corresponding to S
        key, scale, snap_strength, smear, bin_smoothing: Quantization parameters
        timing: Optional timing object to accumulate processing time
    
    Returns:
        Modified complex STFT matrix S (same shape as input)
    """
    mags = np.abs(S)
    phases = np.angle(S)

    # Vectorized quantization: process all frames at once
    # The quantize_spectrum function is now vectorized internally, but we still
    # need to process each frame separately because quantization operates per-frame.
    # However, the internal operations within each frame are fully vectorized.
    proc_start = time.perf_counter()
    try:
        # Process each frame (quantization is frame-independent)
        # NOTE: While we loop over frames, all operations within quantize_spectrum
        # are vectorized using NumPy array operations, eliminating nested loops.
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
        proc_end = time.perf_counter()
        if timing is not None:
            timing.proc += (proc_end - proc_start)
    except Exception:
        proc_end = time.perf_counter()
        if timing is not None:
            timing.proc += (proc_end - proc_start)
        raise

    # Reconstruct complex STFT from modified magnitudes and phases
    S_q = mags * np.exp(1j * phases)
    return S_q


def _align_branches(
    low: np.ndarray,
    high: np.ndarray,
    filter_delay_samples: int,
    stft_latency_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align low and high bands to compensate for latency differences.
    
    The high band path goes through STFT processing which has inherent latency
    (window center offset). The low band path is time-domain only (minimal latency).
    
    This function aligns the bands by:
    - Delaying the low band to match the high band's total latency
    - Ensuring both outputs have the same length
    
    Parameters
    ----------
    low : np.ndarray
        Low band audio (time-domain path).
    high : np.ndarray
        High band audio (will go through STFT path).
    filter_delay_samples : int
        Estimated group delay from the crossover filters (affects both bands).
    stft_latency_samples : int
        Additional latency from STFT processing (affects high band only).
    
    Returns
    -------
    low_aligned : np.ndarray
        Aligned low band (delayed to match high band latency).
    high_aligned : np.ndarray
        High band (unchanged, but may be trimmed to match length).
    """
    # Total latency for high band: filter delay + STFT latency
    high_total_latency = filter_delay_samples + stft_latency_samples
    
    # Total latency for low band: filter delay only (time-domain processing is minimal)
    low_total_latency = filter_delay_samples
    
    # Delay difference: how much we need to delay the low band
    delay_diff = high_total_latency - low_total_latency
    
    # Align low band by adding delay (zero-padding at the start)
    if delay_diff > 0:
        # Low band is faster, delay it
        if low.ndim == 1:
            # Mono
            low_aligned = np.concatenate([
                np.zeros(delay_diff, dtype=low.dtype),
                low
            ])
        else:
            # Multi-channel
            n_channels = low.shape[1]
            padding = np.zeros((delay_diff, n_channels), dtype=low.dtype)
            low_aligned = np.concatenate([padding, low], axis=0)
    else:
        # Low band is slower or same, no delay needed
        low_aligned = low.copy()
    
    # Ensure both have the same length (trim to minimum)
    min_length = min(low_aligned.shape[0], high.shape[0])
    
    if low_aligned.ndim == 1:
        low_aligned = low_aligned[:min_length]
        high_aligned = high[:min_length]
    else:
        low_aligned = low_aligned[:min_length, :]
        high_aligned = high[:min_length, :]
    
    return low_aligned, high_aligned


def _process_single_band(
    x_in: np.ndarray,
    sr: int,
    key: str,
    scale: str,
    snap_strength: float,
    smear: float,
    bin_smoothing: bool,
    pre_quant: bool,
    post_quant: bool,
    distortion_mode: str,
    distortion_params: Dict[str, Any],
    limiter_on: bool,
    limiter_ceiling_db: float,
    dry_wet: float,
    tap_input: np.ndarray,
    timing: Union[RenderTiming, None] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Process a single audio band through the full pipeline.
    
    This is an internal helper function that processes one band (either full-band
    in single-band mode, or low/high band in multiband mode).
    
    Returns:
        (processed_audio, taps_dict)
    """
    n_samples = x_in.shape[0]
    
    # =====================================================================
    # SINGLE STFT: Compute STFT once for the entire processing pipeline
    # All spectral operations (pre-quant, post-quant) will operate on this S
    # Uses OLA-compliant STFT (hop=n_fft/4, Hann window, proper overlap-add)
    # =====================================================================
    stft_start = time.perf_counter()
    try:
        # Note: hop_length and window parameters are ignored by stft_mono()
        # OLA architecture enforces: hop_length=n_fft//4, window=Hann
        S, freqs = stft_mono(
            x_in,
            sr=sr,
            n_fft=N_FFT_DEFAULT,
            hop_length=HOP_LENGTH_DEFAULT,  # Ignored - enforced as n_fft//4
            window=WINDOW_DEFAULT,  # Ignored - enforced as Hann
            center=CENTER_DEFAULT,
        )
        stft_end = time.perf_counter()
        if timing is not None:
            timing.stft = (stft_end - stft_start)
    except Exception:
        stft_end = time.perf_counter()
        if timing is not None:
            timing.stft = (stft_end - stft_start)
        raise

    # --- Stage A: Pre-Quantization (operates directly on S) ---
    # Optimization: If pre-quant is enabled, apply quantization to S.
    # If post-quant is also enabled, we defer iSTFT until after post-quant
    # to maintain "single iSTFT" constraint. Otherwise, do iSTFT here.
    if pre_quant and snap_strength > 0.0:
        S = _apply_spectral_quantization_to_stft(
            S,
            freqs,
            key=key,
            scale=scale,
            snap_strength=snap_strength,
            smear=smear,
            bin_smoothing=bin_smoothing,
            timing=timing,
        )
        # Convert to time-domain for tap and distortion
        # NOTE: If post-quant is also enabled, we defer the final iSTFT
        # until after post-quant to maintain "single iSTFT" constraint.
        # This intermediate conversion is necessary for distortion (time-domain only).
        if not (post_quant and snap_strength > 0.0):
            # No post-quant, so this is the only iSTFT needed
            # =================================================================
            # SINGLE iSTFT: Final reconstruction (no post-quant)
            # Uses OLA-compliant ISTFT with proper overlap-add normalization
            # =================================================================
            istft_start = time.perf_counter()
            try:
                # Note: hop_length and window parameters are ignored by istft_mono()
                # OLA architecture enforces: hop_length=n_fft//4, window=Hann
                x_pre = istft_mono(
                    S,
                    sr=sr,
                    n_fft=N_FFT_DEFAULT,
                    hop_length=HOP_LENGTH_DEFAULT,  # Ignored - enforced as n_fft//4
                    window=WINDOW_DEFAULT,  # Ignored - enforced as Hann
                    length=n_samples,
                    center=CENTER_DEFAULT,
                )
                istft_end = time.perf_counter()
                if timing is not None:
                    timing.istft = (istft_end - istft_start)
            except Exception:
                istft_end = time.perf_counter()
                if timing is not None:
                    timing.istft = (istft_end - istft_start)
                raise
            x_pre = x_pre.astype(np.float32)
        else:
            # Post-quant will be enabled, so we defer iSTFT until after post-quant
            # Convert now only for tap and distortion (time-domain required)
            # This is a temporary conversion - final iSTFT happens after post-quant
            # Uses OLA-compliant ISTFT with proper overlap-add normalization
            x_pre_temp = istft_mono(
                S,
                sr=sr,
                n_fft=N_FFT_DEFAULT,
                hop_length=HOP_LENGTH_DEFAULT,  # Ignored - enforced as n_fft//4
                window=WINDOW_DEFAULT,  # Ignored - enforced as Hann
                length=n_samples,
                center=CENTER_DEFAULT,
            )
            x_pre = x_pre_temp.astype(np.float32)
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
    # Optimization strategy:
    # - If only pre-quant OR only post-quant: 1 STFT, 1 iSTFT
    # - If both pre-quant AND post-quant: 2 STFTs (necessary due to time-domain
    #   distortion), but still only 1 iSTFT at the very end
    if post_quant and snap_strength > 0.0:
        if pre_quant and snap_strength > 0.0:
            # Both pre-quant and post-quant enabled:
            # Pre-quant already converted S to time-domain for distortion.
            # We need a second STFT here because distortion is time-domain only
            # and changes the signal. This is the only case where we need 2 STFTs.
            # NOTE: This second STFT is necessary - distortion requires time-domain
            # processing, so we must convert back to frequency domain for post-quant.
            # Uses OLA-compliant STFT (hop=n_fft/4, Hann window)
            stft_start2 = time.perf_counter()
            try:
                # Note: hop_length and window parameters are ignored by stft_mono()
                # OLA architecture enforces: hop_length=n_fft//4, window=Hann
                S_post, freqs_post = stft_mono(
                    x_post_dist,
                    sr=sr,
                    n_fft=N_FFT_DEFAULT,
                    hop_length=HOP_LENGTH_DEFAULT,  # Ignored - enforced as n_fft//4
                    window=WINDOW_DEFAULT,  # Ignored - enforced as Hann
                    center=CENTER_DEFAULT,
                )
                stft_end2 = time.perf_counter()
                if timing is not None:
                    timing.stft += (stft_end2 - stft_start2)
            except Exception:
                stft_end2 = time.perf_counter()
                if timing is not None:
                    timing.stft += (stft_end2 - stft_start2)
                raise
            
            S_post = _apply_spectral_quantization_to_stft(
                S_post,
                freqs_post,
                key=key,
                scale=scale,
                snap_strength=snap_strength,
                smear=smear,
                bin_smoothing=bin_smoothing,
                timing=timing,
            )
            
            # =================================================================
            # SINGLE iSTFT: Final reconstruction from frequency domain
            # This is the ONLY iSTFT when both pre-quant and post-quant are enabled
            # Uses OLA-compliant ISTFT with proper overlap-add normalization
            # =================================================================
            istft_start = time.perf_counter()
            try:
                # Note: hop_length and window parameters are ignored by istft_mono()
                # OLA architecture enforces: hop_length=n_fft//4, window=Hann
                x_post_quant = istft_mono(
                    S_post,
                    sr=sr,
                    n_fft=N_FFT_DEFAULT,
                    hop_length=HOP_LENGTH_DEFAULT,  # Ignored - enforced as n_fft//4
                    window=WINDOW_DEFAULT,  # Ignored - enforced as Hann
                    length=n_samples,
                    center=CENTER_DEFAULT,
                )
                istft_end = time.perf_counter()
                if timing is not None:
                    timing.istft = (istft_end - istft_start)
            except Exception:
                istft_end = time.perf_counter()
                if timing is not None:
                    timing.istft = (istft_end - istft_start)
                raise
            x_post_quant = x_post_quant.astype(np.float32)
        else:
            # Pre-quant was not enabled, so S is still in frequency domain
            # Apply post-quant directly to S (no second STFT needed)
            S = _apply_spectral_quantization_to_stft(
                S,
                freqs,
                key=key,
                scale=scale,
                snap_strength=snap_strength,
                smear=smear,
                bin_smoothing=bin_smoothing,
                timing=timing,
            )
            
            # =================================================================
            # SINGLE iSTFT: Final reconstruction from frequency domain
            # Uses OLA-compliant ISTFT with proper overlap-add normalization
            # =================================================================
            istft_start = time.perf_counter()
            try:
                # Note: hop_length and window parameters are ignored by istft_mono()
                # OLA architecture enforces: hop_length=n_fft//4, window=Hann
                x_post_quant = istft_mono(
                    S,
                    sr=sr,
                    n_fft=N_FFT_DEFAULT,
                    hop_length=HOP_LENGTH_DEFAULT,  # Ignored - enforced as n_fft//4
                    window=WINDOW_DEFAULT,  # Ignored - enforced as Hann
                    length=n_samples,
                    center=CENTER_DEFAULT,
                )
                istft_end = time.perf_counter()
                if timing is not None:
                    timing.istft = (istft_end - istft_start)
            except Exception:
                istft_end = time.perf_counter()
                if timing is not None:
                    timing.istft = (istft_end - istft_start)
                raise
            x_post_quant = x_post_quant.astype(np.float32)
    else:
        # Post-quant not enabled
        if pre_quant and snap_strength > 0.0:
            # Pre-quant was applied and already converted to time-domain
            x_post_quant = x_post_dist.copy()
        else:
            # No quantization was applied, convert S to time-domain
            # =================================================================
            # SINGLE iSTFT: Final reconstruction from frequency domain
            # Uses OLA-compliant ISTFT with proper overlap-add normalization
            # =================================================================
            istft_start = time.perf_counter()
            try:
                # Note: hop_length and window parameters are ignored by istft_mono()
                # OLA architecture enforces: hop_length=n_fft//4, window=Hann
                x_post_quant = istft_mono(
                    S,
                    sr=sr,
                    n_fft=N_FFT_DEFAULT,
                    hop_length=HOP_LENGTH_DEFAULT,  # Ignored - enforced as n_fft//4
                    window=WINDOW_DEFAULT,  # Ignored - enforced as Hann
                    length=n_samples,
                    center=CENTER_DEFAULT,
                )
                istft_end = time.perf_counter()
                if timing is not None:
                    timing.istft = (istft_end - istft_start)
            except Exception:
                istft_end = time.perf_counter()
                if timing is not None:
                    timing.istft = (istft_end - istft_start)
                raise
            x_post_quant = x_post_quant.astype(np.float32)

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
        "pre_quant": tap_pre_quant,
        "post_dist": tap_post_dist,
        "output": tap_output,
    }
    
    return x_out, taps


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
    preview_enabled: Union[bool, None] = None,
    use_multiband: bool = False,
    crossover_hz: float = 300.0,
    lowband_drive: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Main offline processing entry point.

    Pipeline:
        input
          → (optional) preview truncation (if preview_enabled)
          → (optional) multiband split (if use_multiband=True)
          → (optional) spectral pre-quantization
          → time-domain distortion
          → (optional) spectral post-quantization
          → (optional) limiter
          → dry/wet mix with input
          → (optional) multiband recombination (if use_multiband=True)
    
    Parameters
    ----------
    preview_enabled : bool, optional
        If True, process only the first PREVIEW_MAX_SECONDS of audio for faster iteration.
        If None, reads from DSP_PREVIEW_MODE environment variable (1/true = enabled, 0/false/absent = disabled).
        Defaults to PREVIEW_ENABLED_DEFAULT from config.
    use_multiband : bool, optional
        If True, split audio into low/high bands using Linkwitz-Riley crossover,
        process each band separately, then recombine. Defaults to False.
    crossover_hz : float, optional
        Crossover frequency in Hz for multiband processing. Only used when
        use_multiband=True. Defaults to 300.0 Hz.
    lowband_drive : float, optional
        Drive parameter for low-band saturation. Only used when use_multiband=True.
        Higher values increase saturation. Defaults to 1.0.
    
    Timing information is logged to stdout at the end of processing, including preview mode status.
    """
    # Determine preview mode: check parameter, then environment variable, then default
    if preview_enabled is None:
        env_preview = os.getenv("DSP_PREVIEW_MODE", "").strip().lower()
        if env_preview in ("1", "true", "yes", "on"):
            preview_enabled = True
        elif env_preview in ("0", "false", "no", "off", ""):
            preview_enabled = PREVIEW_ENABLED_DEFAULT
        else:
            preview_enabled = PREVIEW_ENABLED_DEFAULT
    preview_enabled = bool(preview_enabled)
    
    total_start = time.perf_counter()
    timing = RenderTiming()
    
    try:
        if distortion_params is None:
            distortion_params = {}

        # Preview mode: truncate audio to first N seconds for faster iteration
        # This happens before any processing, so the rest of the pipeline doesn't
        # need to know whether it's a preview or full render.
        if preview_enabled:
            max_samples = int(sr * PREVIEW_MAX_SECONDS)
            # Truncate along the time dimension (last axis for 1D, first axis for 2D)
            # Audio shape: 1D = (n_samples,), 2D = (n_samples, n_channels)
            if audio.ndim == 1:
                # Mono: shape is (n_samples,)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                    print(f"[PREVIEW] Truncated audio to first {PREVIEW_MAX_SECONDS:.1f}s ({max_samples} samples)")
            elif audio.ndim == 2:
                # Multi-channel: shape is (n_samples, n_channels)
                if audio.shape[0] > max_samples:
                    audio = audio[:max_samples, :]
                    print(f"[PREVIEW] Truncated audio to first {PREVIEW_MAX_SECONDS:.1f}s ({max_samples} samples)")

        x_in = _ensure_mono(audio)
        n_samples = x_in.shape[0]

        # --- Tap: input ---
        tap_input = x_in.copy()

        # =====================================================================
        # MULTIBAND PROCESSING: Split into low/high bands if enabled
        # =====================================================================
        if use_multiband:
            # Split into low and high bands using Linkwitz-Riley crossover
            low_band, high_band = linkwitz_riley_split(x_in, sr, crossover_hz)
            
            # Estimate latency for alignment
            filter_delay_samples = estimate_filter_group_delay_samples(sr, crossover_hz)
            # STFT latency: when center=True, the window is centered, adding n_fft/2 samples of delay
            stft_latency_samples = N_FFT_DEFAULT // 2
            
            # Align branches to compensate for latency differences
            low_aligned, high_aligned = _align_branches(
                low_band,
                high_band,
                filter_delay_samples,
                stft_latency_samples,
            )
            
            # Low band path: time-domain only (mono-maker + saturation)
            # This keeps bass frequencies in the time domain for better transient response
            # NO STFT processing - bypasses OLA STFT pipeline entirely (from M9)
            low_processed = make_mono_lowband(soft_tube(low_aligned, drive=lowband_drive))
            low_processed = low_processed.astype(np.float32)
            
            # High band path: full OLA-compliant STFT pipeline (spectral quantization, etc.)
            # Uses updated OLA STFT/ISTFT from stft_utils.py
            processed_high, taps_high = _process_single_band(
                high_aligned,
                sr=sr,
                key=key,
                scale=scale,
                snap_strength=snap_strength,
                smear=smear,
                bin_smoothing=bin_smoothing,
                pre_quant=pre_quant,
                post_quant=post_quant,
                distortion_mode=distortion_mode,
                distortion_params=distortion_params,
                limiter_on=limiter_on,
                limiter_ceiling_db=limiter_ceiling_db,
                dry_wet=dry_wet,
                tap_input=high_aligned,  # Use band-specific input for dry/wet mix
                timing=timing,
            )
            
            # Recombine bands: mixed output = low_band_processed + high_band_processed
            # Both bands are correctly reconstructed (low: time-domain, high: OLA-compliant ISTFT)
            x_out = low_processed + processed_high
            x_out = x_out.astype(np.float32)
            
            # Ensure output length matches input
            if x_out.shape[0] != n_samples:
                if x_out.shape[0] > n_samples:
                    x_out = x_out[:n_samples]
                else:
                    pad = np.zeros(n_samples - x_out.shape[0], dtype=np.float32)
                    x_out = np.concatenate([x_out, pad], axis=0)
            
            # For taps, combine low and high band contributions
            # Low band: use processed low band for post_dist (since it goes through saturation)
            # High band: use taps from STFT pipeline
            taps = {
                "input": tap_input,
                "pre_quant": taps_high["pre_quant"],  # Only high band has pre_quant
                "post_dist": low_processed + taps_high["post_dist"],  # Combine low saturation + high distortion
                "output": x_out.copy(),
            }
        else:
            # Single-band processing (original behavior)
            x_out, taps_band = _process_single_band(
                x_in,
                sr=sr,
                key=key,
                scale=scale,
                snap_strength=snap_strength,
                smear=smear,
                bin_smoothing=bin_smoothing,
                pre_quant=pre_quant,
                post_quant=post_quant,
                distortion_mode=distortion_mode,
                distortion_params=distortion_params,
                limiter_on=limiter_on,
                limiter_ceiling_db=limiter_ceiling_db,
                dry_wet=dry_wet,
                tap_input=tap_input,
                timing=timing,
            )
            
            # Add input tap to match expected structure
            taps = {
                "input": tap_input,
                **taps_band,
            }
        
        total_end = time.perf_counter()
        timing.total = total_end - total_start
        
        # Print timing log line with preview mode indicator
        mode_str = "preview" if preview_enabled else "full"
        print(
            f"[RENDER_TIMING] mode={mode_str} "
            f"load={timing.load:.3f}s "
            f"stft={timing.stft:.3f}s "
            f"proc={timing.proc:.3f}s "
            f"istft={timing.istft:.3f}s "
            f"save={timing.save:.3f}s "
            f"total={timing.total:.3f}s"
        )
        
        return x_out, taps
    except Exception:
        # Still log timing even if processing fails
        total_end = time.perf_counter()
        timing.total = total_end - total_start
        mode_str = "preview" if preview_enabled else "full"
        print(
            f"[RENDER_TIMING] mode={mode_str} "
            f"load={timing.load:.3f}s "
            f"stft={timing.stft:.3f}s "
            f"proc={timing.proc:.3f}s "
            f"istft={timing.istft:.3f}s "
            f"save={timing.save:.3f}s "
            f"total={timing.total:.3f}s"
        )
        raise
