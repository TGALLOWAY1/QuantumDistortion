import { useRef, useEffect, useCallback } from 'react';
import type { EQBand } from '../audio/engine';

interface SpectrumAnalyzerProps {
  analyser: AnalyserNode | null;
  eqBands: EQBand[];
  onBandChange: (index: number, band: Partial<EQBand>) => void;
  width: number;
  height: number;
  retuneBandStartHz: number;
  retuneBandEndHz: number;
  retuneBandColor: string;
}

const FREQ_VALUES = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
const FREQ_LABELS = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'];
const GAIN_VALUES = [24, 12, 0, -12, -24];
const BAND_COLORS = ['#2d5f9f', '#3b73bb', '#4891ff', '#58a5f8', '#72bcff'];

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
  return height / 2 - (gain / 24) * (height / 2);
}

function yToGain(y: number, height: number): number {
  return -((y - height / 2) / (height / 2)) * 24;
}

export function SpectrumAnalyzer({
  analyser,
  eqBands,
  onBandChange,
  width,
  height,
  retuneBandStartHz,
  retuneBandEndHz,
  retuneBandColor,
}: SpectrumAnalyzerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const dragIndex = useRef<number | null>(null);

  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = '#101126';
    ctx.fillRect(0, 0, width, height);

    for (const freq of FREQ_VALUES) {
      const x = freqToX(freq, width);
      ctx.beginPath();
      ctx.strokeStyle = '#1f2440';
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    for (const gain of GAIN_VALUES) {
      const y = gainToY(gain, height);
      ctx.beginPath();
      ctx.strokeStyle = gain === 0 ? '#394369' : '#1f2440';
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const bandStartX = freqToX(retuneBandStartHz, width);
    const bandEndX = freqToX(retuneBandEndHz, width);
    ctx.fillStyle = 'rgba(72, 145, 255, 0.16)';
    ctx.fillRect(bandStartX, 0, bandEndX - bandStartX, height);
    ctx.strokeStyle = retuneBandColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(bandStartX, 0, bandEndX - bandStartX, height);

    for (const x of [bandStartX, bandEndX]) {
      ctx.fillStyle = '#4891ff';
      ctx.fillRect(x - 2, 8, 4, height - 16);
      ctx.fillStyle = '#b8d1ff';
      ctx.fillRect(x - 4, height / 2 - 14, 8, 28);
    }

    ctx.fillStyle = '#c8d6ff';
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText('RETUNE BAND', bandStartX + 10, 20);
    ctx.fillText('200 Hz – 5.00 kHz', bandStartX + 10, 36);

    if (analyser) {
      const dataArray = new Float32Array(analyser.frequencyBinCount);
      analyser.getFloatFrequencyData(dataArray);
      const nyquist = analyser.context.sampleRate / 2;

      ctx.beginPath();
      for (let i = 1; i < dataArray.length; i++) {
        const freq = (i / dataArray.length) * nyquist;
        if (freq < 20 || freq > 20000) continue;
        const x = freqToX(freq, width);
        const db = Math.max(-100, Math.min(0, dataArray[i]));
        const y = height - ((db + 100) / 100) * height * 0.82;
        if (i === 1) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#56c5ce';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    for (let i = 0; i < eqBands.length; i++) {
      const x = freqToX(eqBands[i].freq, width);
      const y = gainToY(eqBands[i].gain, height);
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#0d0d1a';
      ctx.fill();
      ctx.strokeStyle = BAND_COLORS[i % BAND_COLORS.length];
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    ctx.fillStyle = '#66709a';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    FREQ_VALUES.forEach((f, i) => {
      ctx.fillText(FREQ_LABELS[i], freqToX(f, width), height - 6);
    });

    ctx.textAlign = 'right';
    GAIN_VALUES.forEach((gain) => {
      ctx.fillText(String(gain), 28, gainToY(gain, height) - 4);
    });

    const meterX = width - 22;
    const meterY = 16;
    const meterW = 8;
    const meterH = height - 32;
    ctx.fillStyle = '#141a2e';
    ctx.fillRect(meterX, meterY, meterW, meterH);
    ctx.fillRect(meterX + 10, meterY, meterW, meterH);
    ctx.strokeStyle = '#344062';
    ctx.strokeRect(meterX, meterY, meterW, meterH);
    ctx.strokeRect(meterX + 10, meterY, meterW, meterH);

    let level = 0.35;
    if (analyser) {
      const td = new Float32Array(analyser.fftSize);
      analyser.getFloatTimeDomainData(td);
      let peak = 0;
      for (let i = 0; i < td.length; i++) peak = Math.max(peak, Math.abs(td[i]));
      level = Math.min(1, peak * 2.2);
    }
    const fillH = meterH * level;
    const grad = ctx.createLinearGradient(0, meterY + meterH, 0, meterY);
    grad.addColorStop(0, '#2bc6cf');
    grad.addColorStop(0.7, '#8bde8f');
    grad.addColorStop(1, '#f4d36f');
    ctx.fillStyle = grad;
    ctx.fillRect(meterX, meterY + meterH - fillH, meterW, fillH);
    ctx.fillRect(meterX + 10, meterY + meterH - fillH * 0.95, meterW, fillH * 0.95);
  }, [analyser, eqBands, height, retuneBandColor, retuneBandEndHz, retuneBandStartHz, width]);

  useEffect(() => {
    const render = () => {
      drawFrame();
      animRef.current = requestAnimationFrame(render);
    };
    animRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animRef.current);
  }, [drawFrame]);

  const findBand = useCallback((x: number, y: number) => {
    for (let i = 0; i < eqBands.length; i++) {
      const bx = freqToX(eqBands[i].freq, width);
      const by = gainToY(eqBands[i].gain, height);
      if (Math.hypot(bx - x, by - y) < 15) return i;
    }
    return null;
  }, [eqBands, height, width]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, cursor: 'crosshair' }}
      className="block rounded-lg border border-border"
      onPointerDown={(e) => {
        const rect = (e.target as HTMLElement).getBoundingClientRect();
        dragIndex.current = findBand(e.clientX - rect.left, e.clientY - rect.top);
      }}
      onPointerMove={(e) => {
        if (dragIndex.current === null) return;
        const rect = (e.target as HTMLElement).getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        onBandChange(dragIndex.current, {
          freq: Math.max(20, Math.min(20000, xToFreq(x, width))),
          gain: Math.max(-24, Math.min(24, yToGain(y, height))),
        });
      }}
      onPointerUp={() => { dragIndex.current = null; }}
      onPointerLeave={() => { dragIndex.current = null; }}
    />
  );
}
