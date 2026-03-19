import { useRef, useEffect, useCallback, useState } from 'react';
import type { EQBand, RetuneStatus } from '../audio/engine';
import { formatMidiNote, midiToFreq, NOTE_NAMES } from '../audio/retune';

interface SpectrumAnalyzerProps {
  analyser: AnalyserNode | null;
  eqBands: EQBand[];
  retuneStatus: RetuneStatus | null;
  retuneEnabled: boolean;
  retuneKey: number;
  retuneScale: string;
  onBandChange: (index: number, band: Partial<EQBand>) => void;
  width: number;
  height: number;
}

// Frequency label positions
const FREQ_LABELS = [20, 50, 100, 200, 500, '1k', '2k', '5k', '10k', '20k'];
const FREQ_VALUES = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

// Band colors matching the effect module accents
const BAND_COLORS = ['#c45e3e', '#4a6fa5', '#4ec48a', '#8a5ec4', '#3eafc4'];
const RETUNE_STATE_COLORS = {
  mapped: '#4ec48a',
  preserved: '#8b90ad',
  muted: '#d96b6b',
} as const;

function freqToX(freq: number, width: number): number {
  const minLog = Math.log10(20);
  const maxLog = Math.log10(20000);
  return ((Math.log10(freq) - minLog) / (maxLog - minLog)) * width;
}

function xToFreq(x: number, width: number): number {
  const minLog = Math.log10(20);
  const maxLog = Math.log10(20000);
  const log = minLog + (x / width) * (maxLog - minLog);
  return Math.pow(10, log);
}

function gainToY(gain: number, height: number): number {
  // Map -24dB to +24dB to canvas height
  return height / 2 - (gain / 24) * (height / 2);
}

function yToGain(y: number, height: number): number {
  return -((y - height / 2) / (height / 2)) * 24;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function formatKeyLabel(key: number): string {
  return NOTE_NAMES[((Math.round(key) % 12) + 12) % 12];
}

/** Compute approximate EQ response contribution for a single band at a given frequency */
function bandResponse(band: EQBand, freq: number): number {
  const bandType = band.type || 'peak';
  const octaves = Math.log2(freq / band.freq);

  switch (bandType) {
    case 'peak': {
      if (band.gain === 0) return 0;
      return band.gain * Math.exp(-0.5 * Math.pow(octaves * band.q * 1.5, 2));
    }
    case 'lowshelf': {
      if (band.gain === 0) return 0;
      // Sigmoid transition: full gain below, zero above
      const t = 1 / (1 + Math.exp(octaves * band.q * 3));
      return band.gain * t;
    }
    case 'highshelf': {
      if (band.gain === 0) return 0;
      // Sigmoid transition: zero below, full gain above
      const t = 1 / (1 + Math.exp(-octaves * band.q * 3));
      return band.gain * t;
    }
    case 'lowpass': {
      // Show rolloff above cutoff
      if (octaves > 0) return -octaves * band.q * 12;
      return 0;
    }
    case 'highpass': {
      // Show rolloff below cutoff
      if (octaves < 0) return Math.abs(octaves) * band.q * -12;
      return 0;
    }
    default:
      return 0;
  }
}

export function SpectrumAnalyzer({
  analyser,
  eqBands,
  retuneStatus,
  retuneEnabled,
  retuneKey,
  retuneScale,
  onBandChange,
  width,
  height,
}: SpectrumAnalyzerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const draggingBand = useRef<number | null>(null);
  const hoverBand = useRef<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    // --- Grid ---
    ctx.strokeStyle = '#1e1e3a';
    ctx.lineWidth = 1;

    // Frequency grid lines
    for (const freq of FREQ_VALUES) {
      const x = freqToX(freq, width);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Gain grid lines (-24, -12, 0, +12, +24)
    for (const gain of [-24, -12, 0, 12, 24]) {
      const y = gainToY(gain, height);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      if (gain === 0) {
        ctx.strokeStyle = '#3a3a5c';
      } else {
        ctx.strokeStyle = '#1e1e3a';
      }
      ctx.stroke();
    }

    // --- Spectrum data ---
    if (analyser) {
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Float32Array(bufferLength);
      analyser.getFloatFrequencyData(dataArray);
      const nyquist = analyser.context.sampleRate / 2;

      // Draw spectrum fill
      ctx.beginPath();
      ctx.moveTo(0, height);

      for (let i = 1; i < bufferLength; i++) {
        const freq = (i / bufferLength) * nyquist;
        if (freq < 20 || freq > 20000) continue;
        const x = freqToX(freq, width);
        // Map dB range (-100 to 0) to canvas
        const db = Math.max(-100, Math.min(0, dataArray[i]));
        const y = height - ((db + 100) / 100) * height * 0.85;
        if (i === 1 || freq <= 20) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.lineTo(width, height);
      ctx.closePath();

      // Gradient fill
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, 'rgba(78, 196, 138, 0.3)');
      gradient.addColorStop(0.5, 'rgba(62, 175, 196, 0.15)');
      gradient.addColorStop(1, 'rgba(62, 175, 196, 0.02)');
      ctx.fillStyle = gradient;
      ctx.fill();

      // Spectrum line
      ctx.beginPath();
      for (let i = 1; i < bufferLength; i++) {
        const freq = (i / bufferLength) * nyquist;
        if (freq < 20 || freq > 20000) continue;
        const x = freqToX(freq, width);
        const db = Math.max(-100, Math.min(0, dataArray[i]));
        const y = height - ((db + 100) / 100) * height * 0.85;
        if (i === 1 || freq <= 20) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.strokeStyle = 'rgba(78, 196, 138, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // --- EQ curve ---
    // Draw approximate combined EQ curve for all filter types
    ctx.beginPath();
    let started = false;
    for (let px = 0; px < width; px += 2) {
      const freq = xToFreq(px, width);
      let totalGain = 0;
      for (let b = 0; b < eqBands.length; b++) {
        totalGain += bandResponse(eqBands[b], freq);
      }
      const y = gainToY(totalGain, height);
      if (!started) {
        ctx.moveTo(px, y);
        started = true;
      } else {
        ctx.lineTo(px, y);
      }
    }
    ctx.strokeStyle = 'rgba(232, 232, 240, 0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Fill under EQ curve
    ctx.lineTo(width, height / 2);
    ctx.lineTo(0, height / 2);
    ctx.closePath();
    ctx.fillStyle = 'rgba(232, 232, 240, 0.04)';
    ctx.fill();

    // --- EQ Band nodes ---
    for (let i = 0; i < eqBands.length; i++) {
      const band = eqBands[i];
      const x = freqToX(band.freq, width);
      const y = gainToY(band.gain, height);
      const isHovered = hoverBand.current === i;
      const isDragging = draggingBand.current === i;
      const nodeSize = isHovered || isDragging ? 8 : 6;

      // Glow
      if (isHovered || isDragging) {
        ctx.beginPath();
        ctx.arc(x, y, 16, 0, 2 * Math.PI);
        ctx.fillStyle = BAND_COLORS[i] + '20';
        ctx.fill();
      }

      // Node outline
      ctx.beginPath();
      ctx.arc(x, y, nodeSize, 0, 2 * Math.PI);
      ctx.fillStyle = '#0d0d1a';
      ctx.fill();
      ctx.strokeStyle = BAND_COLORS[i];
      ctx.lineWidth = 2;
      ctx.stroke();

      // Node fill
      ctx.beginPath();
      ctx.arc(x, y, nodeSize - 2, 0, 2 * Math.PI);
      ctx.fillStyle = isDragging ? BAND_COLORS[i] : '#14142b';
      ctx.fill();
    }

    // --- Frequency labels ---
    ctx.fillStyle = '#555570';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    for (let i = 0; i < FREQ_LABELS.length; i++) {
      const x = freqToX(FREQ_VALUES[i], width);
      ctx.fillText(String(FREQ_LABELS[i]), x, height - 4);
    }

    if (retuneEnabled) {
      const overlayNotes = [...(retuneStatus?.notes ?? [])]
        .sort((a, b) => b.energy - a.energy)
        .slice(0, 4);
      const overlayHeight = 42 + overlayNotes.length * 18;
      const overlayWidth = width - 24;

      ctx.save();
      drawRoundedRect(ctx, 12, 12, overlayWidth, overlayHeight, 12);
      const overlayGradient = ctx.createLinearGradient(12, 12, 12, 12 + overlayHeight);
      overlayGradient.addColorStop(0, 'rgba(13, 13, 26, 0.9)');
      overlayGradient.addColorStop(1, 'rgba(20, 20, 43, 0.76)');
      ctx.fillStyle = overlayGradient;
      ctx.fill();
      ctx.strokeStyle = 'rgba(78, 196, 138, 0.22)';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.textAlign = 'left';
      ctx.font = '11px Inter, sans-serif';
      ctx.fillStyle = '#e8e8f0';
      const stateLabel = !retuneStatus
        ? 'Waiting'
        : retuneStatus.overloaded
          ? 'Overloaded'
          : retuneStatus.noteCount > 0
            ? 'Tracking'
            : 'Listening';
      ctx.fillText(`RETUNE ${formatKeyLabel(retuneKey)} ${retuneScale.replace('_', ' ')} · ${stateLabel}`, 24, 31);

      ctx.font = '10px Inter, sans-serif';
      ctx.fillStyle = '#8888a8';
      const meta = [
        `${retuneStatus?.noteCount ?? 0} groups`,
        retuneStatus ? `${Math.round(retuneStatus.confidence * 100)}% conf` : 'no lock',
        retuneStatus?.lowEndLocked ? 'low locked' : 'low floating',
      ];
      if (retuneStatus?.transientBypassActive) {
        meta.push('attack hold');
      }
      ctx.fillText(meta.join(' · '), 24, 47);

      overlayNotes.forEach((note, index) => {
        const sourceX = clamp(freqToX(midiToFreq(note.sourceMidi), width), 90, width - 24);
        const targetX = clamp(freqToX(midiToFreq(note.targetMidi), width), 90, width - 24);
        const y = 67 + index * 18;
        const color = RETUNE_STATE_COLORS[note.state];
        const label = note.state === 'mapped'
          ? `${formatMidiNote(note.sourceMidi)} → ${formatMidiNote(note.targetMidi)}`
          : `${formatMidiNote(note.sourceMidi)} ${note.state}`;

        ctx.strokeStyle = `${color}cc`;
        ctx.lineWidth = note.isLowEnd ? 2.2 : 1.4;
        ctx.beginPath();
        ctx.moveTo(sourceX, y);
        ctx.lineTo(targetX, y);
        ctx.stroke();

        ctx.fillStyle = '#0d0d1a';
        ctx.beginPath();
        ctx.arc(sourceX, y, 3.5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(targetX, y, 4.5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        const chipText = note.isLowEnd ? `${label} · LOW` : label;
        ctx.font = '10px Inter, sans-serif';
        const chipWidth = ctx.measureText(chipText).width + 14;
        const chipX = clamp((sourceX + targetX) / 2 - chipWidth / 2, 96, width - chipWidth - 18);
        drawRoundedRect(ctx, chipX, y - 10, chipWidth, 16, 8);
        ctx.fillStyle = 'rgba(20, 20, 43, 0.94)';
        ctx.fill();
        ctx.strokeStyle = `${color}55`;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = note.state === 'muted' ? '#f3b2b2' : '#d7d9e5';
        ctx.fillText(chipText, chipX + 7, y + 4);
      });
      ctx.restore();
    }
  }, [analyser, eqBands, retuneEnabled, retuneKey, retuneScale, retuneStatus, width, height]);

  useEffect(() => {
    const render = () => {
      drawFrame();
      animRef.current = requestAnimationFrame(render);
    };

    animRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animRef.current);
  }, [drawFrame]);

  // Hit testing for band nodes
  const findBandAt = useCallback((clientX: number, clientY: number): number | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    for (let i = 0; i < eqBands.length; i++) {
      const bx = freqToX(eqBands[i].freq, width);
      const by = gainToY(eqBands[i].gain, height);
      const dist = Math.sqrt((x - bx) ** 2 + (y - by) ** 2);
      if (dist < 16) return i;
    }
    return null;
  }, [eqBands, width, height]);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    const idx = findBandAt(e.clientX, e.clientY);
    if (idx !== null) {
      draggingBand.current = idx;
      setIsDragging(true);
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    }
  }, [findBandAt]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (draggingBand.current !== null) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const freq = Math.max(20, Math.min(20000, xToFreq(x, width)));
      const gain = Math.max(-24, Math.min(24, yToGain(y, height)));
      onBandChange(draggingBand.current, { freq, gain });
    } else {
      hoverBand.current = findBandAt(e.clientX, e.clientY);
    }
  }, [findBandAt, onBandChange, width, height]);

  const onPointerUp = useCallback(() => {
    draggingBand.current = null;
    setIsDragging(false);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, cursor: isDragging ? 'grabbing' : 'crosshair' }}
      className="block rounded-lg"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={() => {
        hoverBand.current = null;
        draggingBand.current = null;
        setIsDragging(false);
      }}
    />
  );
}
