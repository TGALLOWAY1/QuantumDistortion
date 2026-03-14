from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from quantum_distortion.config import ScaleName, SubSourceName
from quantum_distortion.dsp.quantizer import (
    SCALE_INTERVALS,
    freq_to_midi,
    midi_to_freq,
    note_name_to_pitch_class,
)


@dataclass(frozen=True)
class AutotuneV1Config:
    key: str
    scale: ScaleName
    strength: float = 1.0
    sub_enabled: bool = True
    sub_source: SubSourceName = "root"
    sub_note: str = "C"
    sub_scale_degree: int = 0
    sub_octave: int = 2
    sub_level: float = 0.35
    sub_preserve_mix: float = 0.15
    sub_cut_hz: float = 110.0
    air_cut_hz: float = 5000.0
    air_mix: float = 1.0
    detector_low_hz: float = 110.0
    detector_high_hz: float = 3000.0
    detector_frame_size: int = 4096
    detector_hop_size: int = 512
    detector_min_confidence: float = 0.72
    detector_rms_threshold: float = 0.01
    detector_flatness_threshold: float = 0.55
    note_change_cents: float = 40.0
    note_confirm_frames: int = 3
    note_release_frames: int = 2
    grain_size: int = 1024
    buffer_size: int = 4096
    warm_sub: bool = False


@dataclass(frozen=True)
class PitchTrackDiagnostics:
    ratio_track: np.ndarray
    pitch_track: np.ndarray
    confidence_track: np.ndarray
    voiced_mask: np.ndarray


@dataclass(frozen=True)
class AutotuneV1Result:
    output: np.ndarray
    sub_band: np.ndarray
    body_band: np.ndarray
    air_band: np.ndarray
    corrected_body: np.ndarray
    diagnostics: PitchTrackDiagnostics


def nearest_scale_freq(freq: float, key: str, scale: ScaleName) -> float:
    if freq <= 0.0:
        return freq

    root_pc = note_name_to_pitch_class(key)
    midi_note = freq_to_midi(freq)
    note_in_octave = ((midi_note - root_pc) % 12.0 + 12.0) % 12.0
    octave_base = midi_note - note_in_octave

    best_note = round(midi_note)
    best_dist = np.inf
    for octave_offset in (-1, 0, 1):
        for interval in SCALE_INTERVALS[scale]:
            candidate = octave_base + interval + octave_offset * 12.0
            dist = abs(midi_note - candidate)
            if dist < best_dist:
                best_dist = dist
                best_note = candidate

    return midi_to_freq(float(best_note))


def _design_filter(sr: int, cutoff_hz: float, btype: str, order: int = 4) -> np.ndarray | None:
    nyquist = sr / 2.0
    if cutoff_hz <= 0.0:
        return None
    normalized = float(np.clip(cutoff_hz / nyquist, 1e-5, 0.999))
    return butter(order, normalized, btype=btype, output="sos")


def _zero_phase_filter(audio: np.ndarray, sr: int, cutoff_hz: float, btype: str) -> np.ndarray:
    sos = _design_filter(sr, cutoff_hz, btype=btype)
    if sos is None:
        return audio.astype(np.float32)
    return sosfiltfilt(sos, audio).astype(np.float32)


def split_sub_body_air(
    audio: np.ndarray,
    sr: int,
    sub_cut_hz: float,
    air_cut_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio = np.asarray(audio, dtype=np.float32)
    sub = _zero_phase_filter(audio, sr, sub_cut_hz, "low")
    air = _zero_phase_filter(audio, sr, air_cut_hz, "high")
    body = (audio - sub - air).astype(np.float32)
    return sub.astype(np.float32), body, air.astype(np.float32)


def make_detector_sidechain(
    audio: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    band = np.asarray(audio, dtype=np.float32)
    if low_hz > 0.0:
        band = _zero_phase_filter(band, sr, low_hz, "high")
    if high_hz > 0.0:
        band = _zero_phase_filter(band, sr, high_hz, "low")
    return band.astype(np.float32)


def _spectral_flatness(frame: np.ndarray) -> float:
    window = np.hanning(len(frame))
    spectrum = np.abs(np.fft.rfft(frame * window)) + 1e-8
    geometric = float(np.exp(np.mean(np.log(spectrum))))
    arithmetic = float(np.mean(spectrum))
    if arithmetic <= 1e-8:
        return 1.0
    return geometric / arithmetic


def detect_pitch_yin(
    frame: np.ndarray,
    sr: int,
    min_freq: float = 70.0,
    max_freq: float = 1200.0,
    threshold: float = 0.15,
) -> tuple[float, float]:
    frame = np.asarray(frame, dtype=np.float64)
    if np.max(np.abs(frame)) < 1e-6:
        return 0.0, 0.0

    centered = frame - np.mean(frame)
    max_tau = min(int(sr / max(min_freq, 1e-6)), max(2, len(centered) // 2 - 1))
    min_tau = max(2, int(sr / max(max_freq, 1e-6)))
    if max_tau <= min_tau:
        return 0.0, 0.0

    diff = np.zeros(max_tau + 1, dtype=np.float64)
    for tau in range(1, max_tau + 1):
        delta = centered[:-tau] - centered[tau:]
        diff[tau] = np.dot(delta, delta)

    cmnd = np.ones_like(diff)
    running_sum = 0.0
    for tau in range(1, max_tau + 1):
        running_sum += diff[tau]
        if running_sum > 0.0:
            cmnd[tau] = diff[tau] * tau / running_sum

    tau_estimate = -1
    for tau in range(min_tau, max_tau + 1):
        if cmnd[tau] < threshold:
            while tau + 1 <= max_tau and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_estimate = tau
            break

    if tau_estimate == -1:
        return 0.0, 0.0

    better_tau = float(tau_estimate)
    if min_tau < tau_estimate < max_tau:
        s0 = cmnd[tau_estimate - 1]
        s1 = cmnd[tau_estimate]
        s2 = cmnd[tau_estimate + 1]
        denom = 2.0 * (s0 - 2.0 * s1 + s2)
        if abs(denom) > 1e-12:
            better_tau = tau_estimate + (s0 - s2) / denom

    pitch = float(sr / better_tau) if better_tau > 0 else 0.0
    if pitch < min_freq or pitch > max_freq:
        return 0.0, 0.0

    confidence = float(np.clip(1.0 - cmnd[tau_estimate], 0.0, 1.0))
    return pitch, confidence


def build_pitch_track(
    detector_audio: np.ndarray,
    sr: int,
    cfg: AutotuneV1Config,
) -> PitchTrackDiagnostics:
    detector_audio = np.asarray(detector_audio, dtype=np.float32)
    frame_size = int(max(1024, cfg.detector_frame_size))
    hop_size = int(max(128, cfg.detector_hop_size))
    frame_centers: list[int] = []
    ratio_frames: list[float] = []
    pitch_frames: list[float] = []
    confidence_frames: list[float] = []
    voiced_frames: list[bool] = []

    held_target = 0.0
    candidate_target = 0.0
    candidate_count = 0
    release_count = cfg.note_release_frames + 1
    last_ratio = 1.0

    for start in range(0, len(detector_audio), hop_size):
        frame = detector_audio[start:start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), mode="constant")
        frame_centers.append(min(len(detector_audio) - 1, start + frame_size // 2))

        rms = float(np.sqrt(np.mean(frame * frame)))
        flatness = _spectral_flatness(frame)
        pitch, confidence = detect_pitch_yin(
            frame,
            sr,
            min_freq=max(60.0, cfg.detector_low_hz * 0.65),
            max_freq=max(cfg.detector_high_hz, 400.0),
        )
        is_voiced = (
            pitch > 0.0
            and rms >= cfg.detector_rms_threshold
            and flatness <= cfg.detector_flatness_threshold
            and confidence >= cfg.detector_min_confidence
        )

        if is_voiced:
            target = nearest_scale_freq(pitch, cfg.key, cfg.scale)
            if held_target <= 0.0:
                held_target = target
                candidate_target = 0.0
                candidate_count = 0
            else:
                delta_cents = abs(1200.0 * np.log2(max(target, 1e-6) / max(held_target, 1e-6)))
                if delta_cents >= cfg.note_change_cents:
                    candidate_delta = abs(1200.0 * np.log2(max(target, 1e-6) / max(candidate_target, 1e-6))) if candidate_target > 0.0 else np.inf
                    if candidate_target > 0.0 and candidate_delta < 20.0:
                        candidate_count += 1
                    else:
                        candidate_target = target
                        candidate_count = 1
                    if candidate_count >= cfg.note_confirm_frames:
                        held_target = candidate_target
                        candidate_target = 0.0
                        candidate_count = 0
                else:
                    candidate_target = 0.0
                    candidate_count = 0

            ratio = 1.0 + cfg.strength * ((held_target / pitch) - 1.0)
            ratio = float(np.clip(ratio, 0.5, 2.0))
            release_count = 0
            last_ratio = ratio
        else:
            release_count += 1
            if held_target > 0.0 and release_count <= cfg.note_release_frames:
                ratio = last_ratio
            else:
                held_target = 0.0
                candidate_target = 0.0
                candidate_count = 0
                ratio = 1.0
                last_ratio = 1.0

        ratio_frames.append(ratio)
        pitch_frames.append(pitch)
        confidence_frames.append(confidence)
        voiced_frames.append(is_voiced)

    if not frame_centers:
        zeros = np.ones(len(detector_audio), dtype=np.float32)
        return PitchTrackDiagnostics(
            ratio_track=zeros,
            pitch_track=np.zeros_like(zeros),
            confidence_track=np.zeros_like(zeros),
            voiced_mask=np.zeros(len(detector_audio), dtype=bool),
        )

    sample_positions = np.arange(len(detector_audio), dtype=np.float64)
    centers = np.array(frame_centers, dtype=np.float64)
    ratio_track = np.interp(sample_positions, centers, np.array(ratio_frames, dtype=np.float64)).astype(np.float32)
    pitch_track = np.interp(sample_positions, centers, np.array(pitch_frames, dtype=np.float64)).astype(np.float32)
    confidence_track = np.interp(sample_positions, centers, np.array(confidence_frames, dtype=np.float64)).astype(np.float32)
    voiced_mask = np.interp(sample_positions, centers, np.array(voiced_frames, dtype=np.float64)) >= 0.5

    return PitchTrackDiagnostics(
        ratio_track=ratio_track,
        pitch_track=pitch_track,
        confidence_track=confidence_track,
        voiced_mask=voiced_mask,
    )


def granular_pitch_shift(
    audio: np.ndarray,
    ratio_track: np.ndarray,
    grain_size: int = 1024,
    buffer_size: int = 4096,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    ratio_track = np.asarray(ratio_track, dtype=np.float32)
    if len(audio) == 0 or np.allclose(ratio_track, 1.0, atol=1e-3):
        return audio.copy()

    max_delay = int(max(256, grain_size))
    buffer_size = int(max(max_delay * 2, buffer_size))
    if buffer_size & (buffer_size - 1):
        power = 1
        while power < buffer_size:
            power <<= 1
        buffer_size = power

    buffer_mask = buffer_size - 1
    circ = np.zeros(buffer_size, dtype=np.float32)
    write_pos = 0
    delay_taps = [0.25 * max_delay, 0.75 * max_delay]
    output = np.zeros_like(audio, dtype=np.float32)

    for idx, sample in enumerate(audio):
        circ[write_pos] = sample
        ratio = float(np.clip(ratio_track[idx], 0.5, 2.0))
        delay_slope = 1.0 - ratio

        mixed = 0.0
        weight_sum = 0.0
        for tap_idx in range(2):
            delay_taps[tap_idx] += delay_slope
            while delay_taps[tap_idx] < 0.0:
                delay_taps[tap_idx] += max_delay
            while delay_taps[tap_idx] >= max_delay:
                delay_taps[tap_idx] -= max_delay

            phase = delay_taps[tap_idx] / max_delay
            weight = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
            read_pos = (write_pos - delay_taps[tap_idx] + buffer_size) % buffer_size
            base_idx = int(read_pos) & buffer_mask
            next_idx = (base_idx + 1) & buffer_mask
            frac = read_pos - int(read_pos)
            read_sample = circ[base_idx] * (1.0 - frac) + circ[next_idx] * frac
            mixed += read_sample * weight
            weight_sum += weight

        output[idx] = mixed / weight_sum if weight_sum > 1e-6 else 0.0
        write_pos = (write_pos + 1) & buffer_mask

    latency = max_delay // 2
    if latency > 0 and len(output) > latency:
        output = np.concatenate([output[latency:], np.zeros(latency, dtype=np.float32)])

    return output.astype(np.float32)


def envelope_follow(
    audio: np.ndarray,
    sr: int,
    attack_ms: float = 8.0,
    release_ms: float = 90.0,
) -> np.ndarray:
    audio = np.abs(np.asarray(audio, dtype=np.float32))
    attack_coeff = float(np.exp(-1.0 / max(1.0, attack_ms * 0.001 * sr)))
    release_coeff = float(np.exp(-1.0 / max(1.0, release_ms * 0.001 * sr)))
    env = np.zeros_like(audio, dtype=np.float32)
    current = 0.0
    for idx, sample in enumerate(audio):
        coeff = attack_coeff if sample > current else release_coeff
        current = sample + coeff * (current - sample)
        env[idx] = current
    return env


def _sub_pitch_class(cfg: AutotuneV1Config) -> int:
    if cfg.sub_source == "manual":
        return note_name_to_pitch_class(cfg.sub_note)
    if cfg.sub_source == "scale_degree":
        root_pc = note_name_to_pitch_class(cfg.key)
        intervals = SCALE_INTERVALS[cfg.scale]
        degree = int(np.clip(cfg.sub_scale_degree, 0, len(intervals) - 1))
        return (root_pc + intervals[degree]) % 12
    return note_name_to_pitch_class(cfg.key)


def sub_frequency_from_config(cfg: AutotuneV1Config) -> float:
    pitch_class = _sub_pitch_class(cfg)
    octave = int(np.clip(cfg.sub_octave, 0, 6))
    midi = 12 * (octave + 1) + pitch_class
    return float(midi_to_freq(float(midi)))


def generate_sub_layer(
    reference_audio: np.ndarray,
    sr: int,
    cfg: AutotuneV1Config,
) -> np.ndarray:
    if not cfg.sub_enabled or cfg.sub_level <= 0.0:
        return np.zeros_like(reference_audio, dtype=np.float32)

    freq = sub_frequency_from_config(cfg)
    if freq <= 0.0:
        return np.zeros_like(reference_audio, dtype=np.float32)

    reference_audio = np.asarray(reference_audio, dtype=np.float32)
    env = envelope_follow(reference_audio, sr)
    max_env = float(np.max(env))
    if max_env > 1e-6:
        env = env / max_env

    phase = 2.0 * np.pi * freq * np.arange(len(reference_audio), dtype=np.float32) / float(sr)
    sub = np.sin(phase)
    if cfg.warm_sub:
        sub += 0.15 * np.sin(2.0 * phase)
        sub /= max(float(np.max(np.abs(sub))), 1.0)

    return (cfg.sub_level * env * sub).astype(np.float32)


def apply_autotune_v1(
    audio: np.ndarray,
    sr: int,
    cfg: AutotuneV1Config,
) -> AutotuneV1Result:
    audio = np.asarray(audio, dtype=np.float32)
    sub, body, air = split_sub_body_air(audio, sr, cfg.sub_cut_hz, cfg.air_cut_hz)
    detector = make_detector_sidechain(body, sr, cfg.detector_low_hz, cfg.detector_high_hz)
    diagnostics = build_pitch_track(detector, sr, cfg)
    corrected_body = granular_pitch_shift(body, diagnostics.ratio_track, cfg.grain_size, cfg.buffer_size)
    sub_layer = generate_sub_layer(audio, sr, cfg)
    low_out = (cfg.sub_preserve_mix * sub) + sub_layer if cfg.sub_enabled else sub
    output = low_out + corrected_body + (cfg.air_mix * air)

    return AutotuneV1Result(
        output=output.astype(np.float32),
        sub_band=sub.astype(np.float32),
        body_band=body.astype(np.float32),
        air_band=air.astype(np.float32),
        corrected_body=corrected_body.astype(np.float32),
        diagnostics=diagnostics,
    )
