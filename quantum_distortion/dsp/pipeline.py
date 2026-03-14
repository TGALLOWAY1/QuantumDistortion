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
    DEFAULT_QUANTIZE_MODE,
    DEFAULT_SUB_ENABLED,
    DEFAULT_SUB_SOURCE,
    DEFAULT_SUB_NOTE,
    DEFAULT_SUB_SCALE_DEGREE,
    DEFAULT_SUB_OCTAVE,
    DEFAULT_SUB_LEVEL,
    DEFAULT_SUB_CUT_HZ,
    DEFAULT_AIR_CUT_HZ,
    DEFAULT_AIR_MIX,
    DEFAULT_DISTORTION_MODE,
    DEFAULT_LIMITER_ON,
    DEFAULT_LIMITER_CEILING_DB,
    DEFAULT_DRY_WET,
    DEFAULT_SAMPLE_RATE,
    PREVIEW_ENABLED_DEFAULT,
    PREVIEW_MAX_SECONDS,
    PipelineConfig,
    QuantizeMode,
    SubSourceName,
    ensure_mono_float32,
)
from quantum_distortion.dsp.quantizer import quantize_spectrum, build_target_bins_for_freqs, build_harmonic_target_bins
from quantum_distortion.dsp.autotune import AutotuneV1Config, apply_autotune_v1
from quantum_distortion.dsp.distortion import apply_distortion
from quantum_distortion.dsp.limiter import peak_limiter
from quantum_distortion.dsp.stft_utils import stft_mono, istft_mono
from quantum_distortion.dsp.crossover import linkwitz_riley_split, estimate_filter_group_delay_samples
from quantum_distortion.dsp.saturation import saturate_lowband, make_mono_lowband
from quantum_distortion.dsp import spectral_fx


class _SpectralFXConfig:
    """Simple config object for spectral FX parameters."""
    def __init__(
        self,
        distortion_mode: Union[str, None] = None,
        distortion_strength: float = 0.0,
        distortion_params: Union[Dict[str, Any], None] = None,
    ):
        self.distortion_mode = distortion_mode
        self.distortion_strength = distortion_strength
        self.distortion_params = distortion_params or {}


def apply_spectral_fx(
    mag: np.ndarray,
    phase: np.ndarray,
    cfg: _SpectralFXConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map high-level distortion settings from the config into concrete
    spectral FX parameters and apply them to a single STFT frame.

    This is designed for the HIGH-BAND path only. The low band should
    bypass these FX completely to preserve punch and phase coherence.
    """
    mode = getattr(cfg, "distortion_mode", None)
    s = float(getattr(cfg, "distortion_strength", 0.0))
    params = getattr(cfg, "distortion_params", {}) or {}

    if not mode or s <= 0.0:
        return mag, phase

    if mode == "bitcrush":
        method = params.get("method", "log")
        # Smoothly ramp from subtle to aggressive in dB space.
        step_db = params.get("step_db", 0.5 + 7.5 * (s ** 1.3))
        step = params.get("step", 0.01 + 0.09 * (s ** 1.2))
        threshold = params.get("threshold", None)
        if threshold is None and s >= 0.4:
            threshold = 0.02 * (s ** 1.5) * float(mag.max() if mag.size else 1.0)

        return spectral_fx.bitcrush(
            mag,
            phase,
            method=method,
            step=step,
            step_db=step_db,
            threshold=threshold,
        )

    if mode == "phase_dispersal":
        # Nonlinear mapping so small s is quite gentle.
        amount = params.get("amount", (s ** 1.7) * np.pi)
        # Only rotate louder bins by default.
        thresh_default = 0.01 * float(mag.max() if mag.size else 1.0)
        thresh = params.get("thresh", thresh_default)
        randomized = params.get("randomized", s > 0.35)
        rand_amt = params.get(
            "rand_amt",
            (0.0 if not randomized else 0.2 * (s ** 1.3) * np.pi),
        )
        return spectral_fx.phase_dispersal(
            mag,
            phase,
            thresh=thresh,
            amount=amount,
            randomized=randomized,
            rand_amt=rand_amt,
        )

    if mode == "bin_scramble":
        base_window = params.get("window", None)
        if base_window is None:
            # 3 → 15 as strength increases, using a gentle curve.
            base_window = int(3 + (12 * (s ** 1.2)))
        if base_window < 3:
            base_window = 3
        if base_window % 2 == 0:
            base_window += 1

        mode_name = params.get("mode", "swap" if s < 0.4 else "random_pick")

        return spectral_fx.bin_scramble(
            mag,
            phase,
            window=base_window,
            mode=mode_name,
        )

    # Unknown mode → no-op
    return mag, phase


# OLA-Compliant STFT Configuration
# The STFT/ISTFT functions in stft_utils.py enforce strict OLA architecture:
# - hop_length is always n_fft // 4 (75% overlap)
# - Window is always Hann (sym=False)
# - n_fft=2048 provides good frequency resolution without excessive computation
N_FFT_DEFAULT = 2048
CENTER_DEFAULT = True  # Controls center padding (librosa-compatible behavior)


@dataclass
class RenderTiming:
    """Timing measurements for audio rendering pipeline."""
    load: float = 0.0
    stft: float = 0.0
    proc: float = 0.0
    istft: float = 0.0
    save: float = 0.0
    total: float = 0.0


def _build_quantize_band_mask(
    freqs: np.ndarray,
    min_hz: float,
    max_hz: float,
) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    mask = np.ones_like(freqs, dtype=bool)
    if min_hz > 0.0:
        mask &= freqs >= min_hz
    if max_hz > 0.0:
        mask &= freqs <= max_hz
    if mask.size:
        mask[0] = False
    return mask


def _finalize_single_band_output(
    x_post_quant: np.ndarray,
    sr: int,
    limiter_on: bool,
    limiter_ceiling_db: float,
    dry_wet: float,
    tap_input: np.ndarray,
    output_trim_db: float,
    n_samples: int,
    tap_pre_quant: np.ndarray,
    tap_post_dist: np.ndarray,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
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

    dry_wet = float(np.clip(dry_wet, 0.0, 1.0))
    x_out = (dry_wet * x_limited) + ((1.0 - dry_wet) * tap_input)

    if output_trim_db != 0.0:
        trim_gain = 10.0 ** (output_trim_db / 20.0)
        x_out = x_out * trim_gain

    x_out = x_out.astype(np.float32)
    if x_out.shape[0] != n_samples:
        if x_out.shape[0] > n_samples:
            x_out = x_out[:n_samples]
        else:
            pad = np.zeros(n_samples - x_out.shape[0], dtype=np.float32)
            x_out = np.concatenate([x_out, pad], axis=0)

    taps = {
        "pre_quant": tap_pre_quant,
        "post_dist": tap_post_dist,
        "output": x_out.copy(),
    }
    return x_out, taps




def _apply_spectral_quantization_to_stft(
    S: np.ndarray,
    freqs: np.ndarray,
    key: str,
    scale: str,
    snap_strength: float,
    smear: float,
    bin_smoothing: bool,
    timing: Union[RenderTiming, None] = None,
    is_high_band: bool = False,
    spectral_fx_mode: Union[str, None] = None,
    spectral_fx_strength: float = 0.0,
    spectral_fx_params: Union[Dict[str, Any], None] = None,
    spectral_freeze: bool = False,
    formant_shift: float = 0.0,
    harmonic_lock_hz: float = 0.0,
    quantize_min_hz: float = DEFAULT_SUB_CUT_HZ,
    quantize_max_hz: float = DEFAULT_AIR_CUT_HZ,
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
        is_high_band: If True, this is high-band processing (spectral FX applies)
        spectral_fx_mode: Spectral FX mode ("bitcrush", "phase_dispersal", "bin_scramble", or None)
        spectral_fx_strength: Spectral FX strength (0.0-1.0)
        spectral_fx_params: Optional dict of spectral FX parameter overrides
        spectral_freeze: If True, freeze first-frame magnitudes for all frames (M12.1)
        formant_shift: Formant shift in semitones (M12.2). 0 = no shift.
        harmonic_lock_hz: Fundamental frequency for harmonic locking (M12.3). 0 = disabled.

    Returns:
        Modified complex STFT matrix S (same shape as input)
    """
    mags = np.abs(S)
    phases = np.angle(S)

    proc_start = time.perf_counter()
    try:
        # Cache target bins ONCE before frame loop (performance optimization).
        # build_target_bins_for_freqs() depends only on freqs/key/scale which
        # are constant across all frames. Previously called per-frame (~10s waste).
        if harmonic_lock_hz > 0.0:
            cached_target_bins = build_harmonic_target_bins(freqs, harmonic_lock_hz)
        else:
            cached_target_bins = build_target_bins_for_freqs(freqs, key, scale)

        active_mask = _build_quantize_band_mask(freqs, quantize_min_hz, quantize_max_hz)

        # Spectral freeze (M12.1): capture first-frame magnitudes
        frozen_mags = None
        if spectral_freeze and S.shape[1] > 0:
            frozen_mags = mags[:, 0].copy()

        # Build spectral FX config once (constant across frames)
        sfx_cfg = None
        if is_high_band and spectral_fx_mode is not None:
            sfx_cfg = _SpectralFXConfig(
                distortion_mode=spectral_fx_mode,
                distortion_strength=spectral_fx_strength,
                distortion_params=spectral_fx_params,
            )

        for t in range(S.shape[1]):
            frame_mags = mags[:, t]
            frame_phases = phases[:, t]

            # Spectral freeze: replace magnitudes with first-frame snapshot
            if frozen_mags is not None:
                frame_mags = frozen_mags.copy()

            # Formant shifting (M12.2)
            if formant_shift != 0.0:
                frame_mags = spectral_fx.formant_shift_frame(
                    frame_mags, freqs, formant_shift
                )

            # Apply spectral FX before quantization (only for high-band)
            if sfx_cfg is not None:
                frame_mags, frame_phases = apply_spectral_fx(frame_mags, frame_phases, sfx_cfg)

            new_mags, new_phases = quantize_spectrum(
                mags=frame_mags,
                phases=frame_phases,
                freqs=freqs,
                key=key,
                scale=scale,  # type: ignore[arg-type]
                snap_strength=snap_strength,
                smear=smear,
                bin_smoothing=bin_smoothing,
                target_bins=cached_target_bins,
                active_mask=active_mask,
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
    quantize_mode: QuantizeMode,
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
    passthrough_test: bool = False,
    timing: Union[RenderTiming, None] = None,
    is_high_band: bool = False,
    spectral_fx_mode: Union[str, None] = None,
    spectral_fx_strength: float = 0.0,
    spectral_fx_params: Union[Dict[str, Any], None] = None,
    spectral_freeze: bool = False,
    formant_shift: float = 0.0,
    harmonic_lock_hz: float = 0.0,
    n_fft: int = 2048,
    output_trim_db: float = 0.0,
    sub_enabled: bool = DEFAULT_SUB_ENABLED,
    sub_source: SubSourceName = DEFAULT_SUB_SOURCE,
    sub_note: str = DEFAULT_SUB_NOTE,
    sub_scale_degree: int = DEFAULT_SUB_SCALE_DEGREE,
    sub_octave: int = DEFAULT_SUB_OCTAVE,
    sub_level: float = DEFAULT_SUB_LEVEL,
    sub_cut_hz: float = DEFAULT_SUB_CUT_HZ,
    air_cut_hz: float = DEFAULT_AIR_CUT_HZ,
    air_mix: float = DEFAULT_AIR_MIX,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Process a single audio band through the full pipeline.
    
    This is an internal helper function that processes one band (either full-band
    in single-band mode, or low/high band in multiband mode).
    
    Args:
        passthrough_test: If True, bypass all quantization and spectral FX.
            Performs transparent STFT->ISTFT roundtrip for M10 null test verification.
            Skips magnitude manipulation, distortion, and limiter.
    
    Returns:
        (processed_audio, taps_dict)
    """
    n_samples = x_in.shape[0]
    
    # =====================================================================
    # PASSTHROUGH TEST MODE: Transparent STFT->ISTFT roundtrip
    # Used for M10 null test to verify OLA reconstruction is perfect
    # =====================================================================
    if passthrough_test:
        # Compute STFT
        stft_start = time.perf_counter()
        try:
            S, freqs = stft_mono(
                x_in,
                sr=sr,
                n_fft=N_FFT_DEFAULT,

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
        
        # Immediate ISTFT reconstruction - no magnitude manipulation
        # This verifies that OLA STFT/ISTFT roundtrip is transparent
        istft_start = time.perf_counter()
        try:
            x_out = istft_mono(
                S,
                sr=sr,
                n_fft=N_FFT_DEFAULT,

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
        x_out = x_out.astype(np.float32)
        
        # Ensure output length matches input
        if x_out.shape[0] != n_samples:
            if x_out.shape[0] > n_samples:
                x_out = x_out[:n_samples]
            else:
                pad = np.zeros(n_samples - x_out.shape[0], dtype=np.float32)
                x_out = np.concatenate([x_out, pad], axis=0)
        
        # Passthrough mode: output should reconstruct exactly the input
        # No quantization, no distortion, no limiter, no dry/wet mix
        taps = {
            "pre_quant": x_in.copy(),  # No quantization applied
            "post_dist": x_out.copy(),  # No distortion applied
            "output": x_out.copy(),
        }
        
        return x_out, taps

    if quantize_mode == "autotune_v1":
        if pre_quant and snap_strength > 0.0:
            proc_start = time.perf_counter()
            try:
                autotune_result = apply_autotune_v1(
                    x_in,
                    sr=sr,
                    cfg=AutotuneV1Config(
                        key=key,
                        scale=scale,  # type: ignore[arg-type]
                        strength=float(np.clip(snap_strength, 0.0, 1.0)),
                        sub_enabled=sub_enabled,
                        sub_source=sub_source,
                        sub_note=sub_note,
                        sub_scale_degree=sub_scale_degree,
                        sub_octave=sub_octave,
                        sub_level=sub_level,
                        sub_cut_hz=sub_cut_hz,
                        air_cut_hz=air_cut_hz,
                        air_mix=air_mix,
                    ),
                )
                if timing is not None:
                    timing.proc += time.perf_counter() - proc_start
            except Exception:
                if timing is not None:
                    timing.proc += time.perf_counter() - proc_start
                raise
            x_pre = autotune_result.output.astype(np.float32)
        else:
            x_pre = x_in.copy()

        tap_pre_quant = x_pre.copy()

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

        # V1 autotune path intentionally avoids spectral post-quantization.
        x_post_quant = x_post_dist.copy()

        return _finalize_single_band_output(
            x_post_quant=x_post_quant,
            sr=sr,
            limiter_on=limiter_on,
            limiter_ceiling_db=limiter_ceiling_db,
            dry_wet=dry_wet,
            tap_input=tap_input,
            output_trim_db=output_trim_db,
            n_samples=n_samples,
            tap_pre_quant=tap_pre_quant,
            tap_post_dist=tap_post_dist,
        )
    
    # =====================================================================
    # NORMAL PROCESSING MODE: Full pipeline with quantization, distortion, etc.
    # =====================================================================
    
    # =====================================================================
    # SINGLE STFT: Compute STFT once for the entire processing pipeline
    # All spectral operations (pre-quant, post-quant) will operate on this S
    # Uses OLA-compliant STFT (hop=n_fft/4, Hann window, proper overlap-add)
    # =====================================================================
    stft_start = time.perf_counter()
    try:
        # OLA architecture enforces: hop_length=n_fft//4, window=Hann
        S, freqs = stft_mono(
            x_in,
            sr=sr,
            n_fft=N_FFT_DEFAULT,

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
            is_high_band=is_high_band,
            spectral_fx_mode=spectral_fx_mode,
            spectral_fx_strength=spectral_fx_strength,
            spectral_fx_params=spectral_fx_params,
            spectral_freeze=spectral_freeze,
            formant_shift=formant_shift,
            harmonic_lock_hz=harmonic_lock_hz,
            quantize_min_hz=sub_cut_hz,
            quantize_max_hz=air_cut_hz,
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
                # OLA architecture enforces: hop_length=n_fft//4, window=Hann
                x_pre = istft_mono(
                    S,
                    sr=sr,
                    n_fft=N_FFT_DEFAULT,

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
    
                # OLA architecture enforces: hop_length=n_fft//4, window=Hann
                S_post, freqs_post = stft_mono(
                    x_post_dist,
                    sr=sr,
                    n_fft=N_FFT_DEFAULT,

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
                is_high_band=is_high_band,
                spectral_fx_mode=spectral_fx_mode,
                spectral_fx_strength=spectral_fx_strength,
                spectral_fx_params=spectral_fx_params,
                spectral_freeze=spectral_freeze,
                formant_shift=formant_shift,
                harmonic_lock_hz=harmonic_lock_hz,
                quantize_min_hz=sub_cut_hz,
                quantize_max_hz=air_cut_hz,
            )
            
            # =================================================================
            # SINGLE iSTFT: Final reconstruction from frequency domain
            # This is the ONLY iSTFT when both pre-quant and post-quant are enabled
            # Uses OLA-compliant ISTFT with proper overlap-add normalization
            # =================================================================
            istft_start = time.perf_counter()
            try:
                # OLA architecture enforces: hop_length=n_fft//4, window=Hann
                x_post_quant = istft_mono(
                    S_post,
                    sr=sr,
                    n_fft=N_FFT_DEFAULT,

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
                is_high_band=is_high_band,
                spectral_fx_mode=spectral_fx_mode,
                spectral_fx_strength=spectral_fx_strength,
                spectral_fx_params=spectral_fx_params,
                spectral_freeze=spectral_freeze,
                formant_shift=formant_shift,
                harmonic_lock_hz=harmonic_lock_hz,
                quantize_min_hz=sub_cut_hz,
                quantize_max_hz=air_cut_hz,
            )

            # =================================================================
            # SINGLE iSTFT: Final reconstruction from frequency domain
            # Uses OLA-compliant ISTFT with proper overlap-add normalization
            # =================================================================
            istft_start = time.perf_counter()
            try:
                # OLA architecture enforces: hop_length=n_fft//4, window=Hann
                x_post_quant = istft_mono(
                    S,
                    sr=sr,
                    n_fft=N_FFT_DEFAULT,

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
                # OLA architecture enforces: hop_length=n_fft//4, window=Hann
                x_post_quant = istft_mono(
                    S,
                    sr=sr,
                    n_fft=N_FFT_DEFAULT,

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

    # --- Output Trim ---
    if output_trim_db != 0.0:
        trim_gain = 10.0 ** (output_trim_db / 20.0)
        x_out = x_out * trim_gain

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


def _parse_ui_config(
    config: Dict[str, Any],
    key: str,
    scale: str,
    quantize_mode: QuantizeMode,
    crossover_hz: float,
    lowband_drive: float,
    mono_strength: float,
    spectral_fx_mode: Union[str, None],
    spectral_fx_strength: float,
    spectral_freeze: bool,
    formant_shift: float,
    harmonic_lock_hz: float,
    delta_listen: bool,
    output_trim_db: float,
    sub_enabled: bool,
    sub_source: SubSourceName,
    sub_note: str,
    sub_scale_degree: int,
    sub_octave: int,
    sub_level: float,
    sub_cut_hz: float,
    air_cut_hz: float,
    air_mix: float,
) -> Dict[str, Any]:
    """Parse V2 UI centralized config dict and return overridden parameter values."""
    result: Dict[str, Any] = {
        "key": key, "scale": scale, "quantize_mode": quantize_mode,
        "crossover_hz": crossover_hz,
        "lowband_drive": lowband_drive, "mono_strength": mono_strength,
        "spectral_fx_mode": spectral_fx_mode, "spectral_fx_strength": spectral_fx_strength,
        "spectral_freeze": spectral_freeze, "formant_shift": formant_shift,
        "harmonic_lock_hz": harmonic_lock_hz, "delta_listen": delta_listen,
        "output_trim_db": output_trim_db, "low_trim": 0.0, "use_multiband": True,
        "sub_enabled": sub_enabled, "sub_source": sub_source, "sub_note": sub_note,
        "sub_scale_degree": sub_scale_degree, "sub_octave": sub_octave,
        "sub_level": sub_level, "sub_cut_hz": sub_cut_hz,
        "air_cut_hz": air_cut_hz, "air_mix": air_mix,
    }

    if "quantization" in config:
        quant_cfg = config["quantization"]
        result["key"] = str(quant_cfg.get("key", key))
        result["scale"] = str(quant_cfg.get("scale", scale))
        result["quantize_mode"] = str(quant_cfg.get("mode", quantize_mode))
        result["sub_enabled"] = bool(quant_cfg.get("sub_enabled", sub_enabled))
        result["sub_source"] = str(quant_cfg.get("sub_source", sub_source))
        result["sub_note"] = str(quant_cfg.get("sub_note", sub_note))
        result["sub_scale_degree"] = int(quant_cfg.get("sub_scale_degree", sub_scale_degree))
        result["sub_octave"] = int(quant_cfg.get("sub_octave", sub_octave))
        result["sub_level"] = float(quant_cfg.get("sub_level", sub_level))
        result["sub_cut_hz"] = float(quant_cfg.get("sub_cut_hz", sub_cut_hz))
        result["air_cut_hz"] = float(quant_cfg.get("air_cut_hz", air_cut_hz))
        result["air_mix"] = float(quant_cfg.get("air_mix", air_mix))
    if "crossover_freq" in config:
        result["crossover_hz"] = float(config["crossover_freq"])
    if "low_band" in config:
        low_band_cfg = config["low_band"]
        saturation_amount = low_band_cfg.get("saturation_amount", 0.3)
        result["lowband_drive"] = 1.0 + (saturation_amount * 4.0)
        result["mono_strength"] = float(low_band_cfg.get("mono_strength", mono_strength))
        result["low_trim"] = float(low_band_cfg.get("output_trim_db", 0.0))
    if "high_band" in config:
        high_band_cfg = config["high_band"]
        bin_scrambling = high_band_cfg.get("bin_scrambling", 0.2)
        phase_dispersal_amt = high_band_cfg.get("phase_dispersal", 0.3)
        mag_decimation = high_band_cfg.get("mag_decimation", 0.5)
        if bin_scrambling > 0.0:
            result["spectral_fx_mode"] = "bin_scramble"
            result["spectral_fx_strength"] = bin_scrambling
        elif phase_dispersal_amt > 0.0:
            result["spectral_fx_mode"] = "phase_dispersal"
            result["spectral_fx_strength"] = phase_dispersal_amt
        elif mag_decimation > 0.0:
            result["spectral_fx_mode"] = "bitcrush"
            result["spectral_fx_strength"] = mag_decimation
        result["output_trim_db"] = float(high_band_cfg.get("output_trim_db", output_trim_db))
    if "quantum_fx" in config:
        quantum_fx_cfg = config["quantum_fx"]
        result["spectral_freeze"] = bool(quantum_fx_cfg.get("spectral_freeze", spectral_freeze))
        result["formant_shift"] = float(quantum_fx_cfg.get("formant_shift", formant_shift))
        result["harmonic_lock_hz"] = float(quantum_fx_cfg.get("fundamental_hz", harmonic_lock_hz))
    if "delta_listen" in config:
        result["delta_listen"] = bool(config["delta_listen"])

    return result


def _process_multiband(
    x_in: np.ndarray,
    sr: int,
    key: str,
    scale: str,
    quantize_mode: QuantizeMode,
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
    crossover_hz: float,
    lowband_drive: float,
    mono_strength: float,
    low_trim: float,
    passthrough_test: bool,
    timing: RenderTiming,
    spectral_fx_mode: Union[str, None],
    spectral_fx_strength: float,
    spectral_fx_params: Dict[str, Any],
    spectral_freeze: bool,
    formant_shift: float,
    harmonic_lock_hz: float,
    output_trim_db: float,
    tap_input: np.ndarray,
    sub_enabled: bool,
    sub_source: SubSourceName,
    sub_note: str,
    sub_scale_degree: int,
    sub_octave: int,
    sub_level: float,
    sub_cut_hz: float,
    air_cut_hz: float,
    air_mix: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Multiband processing: split into low/high, process separately, recombine."""
    n_samples = x_in.shape[0]

    low_band, high_band = linkwitz_riley_split(x_in, sr, crossover_hz)
    filter_delay_samples = estimate_filter_group_delay_samples(sr, crossover_hz)
    stft_latency_samples = N_FFT_DEFAULT // 2

    low_aligned, high_aligned = _align_branches(
        low_band, high_band, filter_delay_samples, stft_latency_samples,
    )

    # Low band: time-domain saturation + mono-maker
    low_saturated = saturate_lowband(low_aligned, drive=lowband_drive)
    if mono_strength >= 1.0:
        low_processed = make_mono_lowband(low_saturated)
    elif mono_strength > 0.0:
        low_mono = make_mono_lowband(low_saturated)
        low_processed = (mono_strength * low_mono) + ((1.0 - mono_strength) * low_saturated)
    else:
        low_processed = low_saturated
    if low_trim != 0.0:
        low_processed = low_processed * (10.0 ** (low_trim / 20.0))
    low_processed = low_processed.astype(np.float32)

    # High band: full STFT pipeline
    processed_high, taps_high = _process_single_band(
        high_aligned, sr=sr, key=key, scale=scale, quantize_mode=quantize_mode,
        snap_strength=snap_strength, smear=smear, bin_smoothing=bin_smoothing,
        pre_quant=pre_quant, post_quant=post_quant,
        distortion_mode=distortion_mode, distortion_params=distortion_params,
        limiter_on=limiter_on, limiter_ceiling_db=limiter_ceiling_db,
        dry_wet=dry_wet, tap_input=high_aligned,
        passthrough_test=passthrough_test, timing=timing,
        is_high_band=True,
        spectral_fx_mode=spectral_fx_mode, spectral_fx_strength=spectral_fx_strength,
        spectral_fx_params=spectral_fx_params,
        spectral_freeze=spectral_freeze, formant_shift=formant_shift,
        harmonic_lock_hz=harmonic_lock_hz, output_trim_db=output_trim_db,
        sub_enabled=sub_enabled, sub_source=sub_source, sub_note=sub_note,
        sub_scale_degree=sub_scale_degree, sub_octave=sub_octave,
        sub_level=sub_level, sub_cut_hz=sub_cut_hz, air_cut_hz=air_cut_hz,
        air_mix=air_mix,
    )

    # Recombine
    x_out = (low_processed + processed_high).astype(np.float32)
    if x_out.shape[0] != n_samples:
        if x_out.shape[0] > n_samples:
            x_out = x_out[:n_samples]
        else:
            pad = np.zeros(n_samples - x_out.shape[0], dtype=np.float32)
            x_out = np.concatenate([x_out, pad], axis=0)

    taps = {
        "input": tap_input,
        "pre_quant": taps_high["pre_quant"],
        "post_dist": low_processed + taps_high["post_dist"],
        "output": x_out.copy(),
    }
    return x_out, taps


def process_audio(
    audio: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    key: str = DEFAULT_KEY,
    scale: str = DEFAULT_SCALE,
    quantize_mode: QuantizeMode = DEFAULT_QUANTIZE_MODE,
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
    passthrough_test: bool = False,
    spectral_fx_mode: Union[str, None] = None,
    spectral_fx_strength: float = 0.0,
    spectral_fx_params: Union[Dict[str, Any], None] = None,
    config: Union[Dict[str, Any], None] = None,
    spectral_freeze: bool = False,
    formant_shift: float = 0.0,
    harmonic_lock_hz: float = 0.0,
    delta_listen: bool = False,
    mono_strength: float = 1.0,
    output_trim_db: float = 0.0,
    sub_enabled: bool = DEFAULT_SUB_ENABLED,
    sub_source: SubSourceName = DEFAULT_SUB_SOURCE,
    sub_note: str = DEFAULT_SUB_NOTE,
    sub_scale_degree: int = DEFAULT_SUB_SCALE_DEGREE,
    sub_octave: int = DEFAULT_SUB_OCTAVE,
    sub_level: float = DEFAULT_SUB_LEVEL,
    sub_cut_hz: float = DEFAULT_SUB_CUT_HZ,
    air_cut_hz: float = DEFAULT_AIR_CUT_HZ,
    air_mix: float = DEFAULT_AIR_MIX,
    *,
    pipeline_config: Union[PipelineConfig, None] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Main offline processing entry point.

    Can be called either with individual keyword arguments (backward-compatible)
    or with a PipelineConfig object via pipeline_config=. When pipeline_config
    is provided, its values override the individual keyword arguments.

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
    pipeline_config : PipelineConfig, optional
        If provided, overrides all individual keyword arguments with values
        from the config object. Only audio and sr are still taken as positional.
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
    passthrough_test : bool, optional
        If True, bypass all quantization and spectral FX. Performs transparent
        STFT->ISTFT roundtrip for M10 null test verification. Skips magnitude
        manipulation, distortion, and limiter. Output must reconstruct exactly
        the original high-band input. Defaults to False.
    
    Timing information is logged to stdout at the end of processing, including preview mode status.
    """
    # Apply PipelineConfig if provided — overrides individual kwargs
    if pipeline_config is not None:
        pc = pipeline_config
        key = pc.key
        scale = pc.scale
        quantize_mode = pc.quantize_mode
        snap_strength = pc.snap_strength
        smear = pc.smear
        bin_smoothing = pc.bin_smoothing
        pre_quant = pc.pre_quant
        post_quant = pc.post_quant
        sub_enabled = pc.sub_enabled
        sub_source = pc.sub_source
        sub_note = pc.sub_note
        sub_scale_degree = pc.sub_scale_degree
        sub_octave = pc.sub_octave
        sub_level = pc.sub_level
        sub_cut_hz = pc.sub_cut_hz
        air_cut_hz = pc.air_cut_hz
        air_mix = pc.air_mix
        distortion_mode = pc.distortion_mode
        distortion_params = pc.distortion_params
        limiter_on = pc.limiter_on
        limiter_ceiling_db = pc.limiter_ceiling_db
        dry_wet = pc.dry_wet
        preview_enabled = pc.preview_enabled
        use_multiband = pc.use_multiband
        crossover_hz = pc.crossover_hz
        lowband_drive = pc.lowband_drive
        passthrough_test = pc.passthrough_test
        spectral_fx_mode = pc.spectral_fx_mode
        spectral_fx_strength = pc.spectral_fx_strength
        spectral_fx_params = pc.spectral_fx_params
        spectral_freeze = pc.spectral_freeze
        formant_shift = pc.formant_shift
        harmonic_lock_hz = pc.harmonic_lock_hz
        delta_listen = pc.delta_listen
        mono_strength = pc.mono_strength
        output_trim_db = pc.output_trim_db

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
        if spectral_fx_params is None:
            spectral_fx_params = {}

        # Default low-band trim (overridden by config if provided)
        low_trim = 0.0

        # Parse V2 UI config dict if provided
        if config is not None:
            parsed = _parse_ui_config(
                config, key=key, scale=scale, quantize_mode=quantize_mode, crossover_hz=crossover_hz,
                lowband_drive=lowband_drive, mono_strength=mono_strength,
                spectral_fx_mode=spectral_fx_mode, spectral_fx_strength=spectral_fx_strength,
                spectral_freeze=spectral_freeze, formant_shift=formant_shift,
                harmonic_lock_hz=harmonic_lock_hz, delta_listen=delta_listen,
                output_trim_db=output_trim_db,
                sub_enabled=sub_enabled, sub_source=sub_source, sub_note=sub_note,
                sub_scale_degree=sub_scale_degree, sub_octave=sub_octave,
                sub_level=sub_level, sub_cut_hz=sub_cut_hz, air_cut_hz=air_cut_hz,
                air_mix=air_mix,
            )
            key = parsed["key"]
            scale = parsed["scale"]
            quantize_mode = parsed["quantize_mode"]
            crossover_hz = parsed["crossover_hz"]
            lowband_drive = parsed["lowband_drive"]
            mono_strength = parsed["mono_strength"]
            spectral_fx_mode = parsed["spectral_fx_mode"]
            spectral_fx_strength = parsed["spectral_fx_strength"]
            spectral_freeze = parsed["spectral_freeze"]
            formant_shift = parsed["formant_shift"]
            harmonic_lock_hz = parsed["harmonic_lock_hz"]
            delta_listen = parsed["delta_listen"]
            output_trim_db = parsed["output_trim_db"]
            low_trim = parsed["low_trim"]
            use_multiband = parsed["use_multiband"]
            sub_enabled = parsed["sub_enabled"]
            sub_source = parsed["sub_source"]
            sub_note = parsed["sub_note"]
            sub_scale_degree = parsed["sub_scale_degree"]
            sub_octave = parsed["sub_octave"]
            sub_level = parsed["sub_level"]
            sub_cut_hz = parsed["sub_cut_hz"]
            air_cut_hz = parsed["air_cut_hz"]
            air_mix = parsed["air_mix"]

        # Preview mode: truncate audio
        if preview_enabled:
            max_samples = int(sr * PREVIEW_MAX_SECONDS)
            if audio.ndim == 1 and len(audio) > max_samples:
                audio = audio[:max_samples]
                print(f"[PREVIEW] Truncated audio to first {PREVIEW_MAX_SECONDS:.1f}s ({max_samples} samples)")
            elif audio.ndim == 2 and audio.shape[0] > max_samples:
                audio = audio[:max_samples, :]
                print(f"[PREVIEW] Truncated audio to first {PREVIEW_MAX_SECONDS:.1f}s ({max_samples} samples)")

        x_in = ensure_mono_float32(audio)
        tap_input = x_in.copy()

        if (
            quantize_mode == "autotune_v1"
            and (
                spectral_fx_mode is not None
                or spectral_freeze
                or formant_shift != 0.0
                or harmonic_lock_hz > 0.0
            )
        ):
            quantize_mode = "spectral_bins"

        if quantize_mode == "autotune_v1" and snap_strength > 0.0:
            use_multiband = False

        if use_multiband:
            x_out, taps = _process_multiband(
                x_in, sr=sr, key=key, scale=scale, quantize_mode=quantize_mode,
                snap_strength=snap_strength, smear=smear, bin_smoothing=bin_smoothing,
                pre_quant=pre_quant, post_quant=post_quant,
                distortion_mode=distortion_mode, distortion_params=distortion_params,
                limiter_on=limiter_on, limiter_ceiling_db=limiter_ceiling_db,
                dry_wet=dry_wet, crossover_hz=crossover_hz,
                lowband_drive=lowband_drive, mono_strength=mono_strength,
                low_trim=low_trim, passthrough_test=passthrough_test, timing=timing,
                spectral_fx_mode=spectral_fx_mode, spectral_fx_strength=spectral_fx_strength,
                spectral_fx_params=spectral_fx_params,
                spectral_freeze=spectral_freeze, formant_shift=formant_shift,
                harmonic_lock_hz=harmonic_lock_hz, output_trim_db=output_trim_db,
                tap_input=tap_input,
                sub_enabled=sub_enabled, sub_source=sub_source, sub_note=sub_note,
                sub_scale_degree=sub_scale_degree, sub_octave=sub_octave,
                sub_level=sub_level, sub_cut_hz=sub_cut_hz,
                air_cut_hz=air_cut_hz, air_mix=air_mix,
            )
        else:
            x_out, taps_band = _process_single_band(
                x_in, sr=sr, key=key, scale=scale, quantize_mode=quantize_mode,
                snap_strength=snap_strength, smear=smear, bin_smoothing=bin_smoothing,
                pre_quant=pre_quant, post_quant=post_quant,
                distortion_mode=distortion_mode, distortion_params=distortion_params,
                limiter_on=limiter_on, limiter_ceiling_db=limiter_ceiling_db,
                dry_wet=dry_wet, tap_input=tap_input,
                passthrough_test=passthrough_test, timing=timing,
                is_high_band=False,
                spectral_fx_mode=spectral_fx_mode, spectral_fx_strength=spectral_fx_strength,
                spectral_fx_params=spectral_fx_params,
                spectral_freeze=spectral_freeze, formant_shift=formant_shift,
                harmonic_lock_hz=harmonic_lock_hz, output_trim_db=output_trim_db,
                sub_enabled=sub_enabled, sub_source=sub_source, sub_note=sub_note,
                sub_scale_degree=sub_scale_degree, sub_octave=sub_octave,
                sub_level=sub_level, sub_cut_hz=sub_cut_hz,
                air_cut_hz=air_cut_hz, air_mix=air_mix,
            )
            taps = {"input": tap_input, **taps_band}

        # Delta listen (M13): output = input - processed (hear what was removed)
        if delta_listen:
            min_len = min(len(tap_input), len(x_out))
            x_out = tap_input[:min_len] - x_out[:min_len]
            x_out = x_out.astype(np.float32)
            taps["output"] = x_out.copy()

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
