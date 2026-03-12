import { useRef, useEffect, useCallback } from 'react';
import type { EQBand } from '../audio/engine';

interface SpectrumAnalyzerProps {
  analyser: AnalyserNode | null;
  eqBands: EQBand[];
  onBandChange: (index: number, band: Partial<EQBand>) => void;
  width: number;
  height: number;
}

// Frequency label positions
const FREQ_LABELS = [20, 50, 100, 200, 500, '1k', '2k', '5k', '10k', '20k'];
const FREQ_VALUES = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

// Band colors matching the effect module accents
const BAND_COLORS = ['#c45e3e', '#4a6fa5', '#4ec48a', '#8a5ec4', '#3eafc4'];

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

export function SpectrumAnalyzer({ analyser, eqBands, onBandChange, width, height }: SpectrumAnalyzerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const draggingBand = useRef<number | null>(null);
  const hoverBand = useRef<number | null>(null);

  const draw = useCallback(() => {
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
    // Draw approximate combined EQ curve
    ctx.beginPath();
    let started = false;
    for (let px = 0; px < width; px += 2) {
      const freq = xToFreq(px, width);
      let totalGain = 0;
      for (let b = 0; b < eqBands.length; b++) {
        const band = eqBands[b];
        if (band.gain === 0) continue;
        // Approximate bell curve response
        const octaves = Math.log2(freq / band.freq);
        const response = band.gain * Math.exp(-0.5 * Math.pow(octaves * band.q * 1.5, 2));
        totalGain += response;
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

    animRef.current = requestAnimationFrame(draw);
  }, [analyser, eqBands, width, height]);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

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
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, cursor: draggingBand.current !== null ? 'grabbing' : 'crosshair' }}
      className="block rounded-lg"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={() => { hoverBand.current = null; }}
    />
  );
}
