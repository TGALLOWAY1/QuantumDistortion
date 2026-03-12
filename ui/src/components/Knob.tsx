import { useRef, useCallback, useEffect, useState } from 'react';

interface KnobProps {
  value: number;       // 0 to 1
  onChange: (v: number) => void;
  size?: number;
  label: string;
  color?: string;
  min?: number;
  max?: number;
  displayValue?: string;
}

export function Knob({
  value,
  onChange,
  size = 56,
  label,
  color = '#888',
  min = 0,
  max = 1,
  displayValue,
}: KnobProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragging = useRef(false);
  const dragStartY = useRef(0);
  const dragStartVal = useRef(0);
  const [hovering, setHovering] = useState(false);

  const normalized = (value - min) / (max - min);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    const cx = size / 2;
    const cy = size / 2;
    const radius = size / 2 - 4;

    // Background ring
    ctx.clearRect(0, 0, size, size);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0.75 * Math.PI, 2.25 * Math.PI);
    ctx.strokeStyle = '#2a2a4a';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Value arc
    const startAngle = 0.75 * Math.PI;
    const endAngle = startAngle + normalized * 1.5 * Math.PI;
    if (normalized > 0.001) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius, startAngle, endAngle);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    // Knob body
    const gradient = ctx.createRadialGradient(cx - 3, cy - 3, 0, cx, cy, radius - 4);
    gradient.addColorStop(0, '#4a4a6a');
    gradient.addColorStop(1, '#2a2a4a');
    ctx.beginPath();
    ctx.arc(cx, cy, radius - 5, 0, 2 * Math.PI);
    ctx.fillStyle = gradient;
    ctx.fill();

    // Indicator line
    const indicatorAngle = startAngle + normalized * 1.5 * Math.PI;
    const innerR = 6;
    const outerR = radius - 8;
    ctx.beginPath();
    ctx.moveTo(
      cx + Math.cos(indicatorAngle) * innerR,
      cy + Math.sin(indicatorAngle) * innerR,
    );
    ctx.lineTo(
      cx + Math.cos(indicatorAngle) * outerR,
      cy + Math.sin(indicatorAngle) * outerR,
    );
    ctx.strokeStyle = '#e8e8f0';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.stroke();
  }, [size, normalized, color]);

  useEffect(() => { draw(); }, [draw]);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    dragging.current = true;
    dragStartY.current = e.clientY;
    dragStartVal.current = normalized;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, [normalized]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    const delta = (dragStartY.current - e.clientY) / 150;
    const newNorm = Math.max(0, Math.min(1, dragStartVal.current + delta));
    const newVal = min + newNorm * (max - min);
    onChange(newVal);
  }, [min, max, onChange]);

  const onPointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  const onDoubleClick = useCallback(() => {
    // Reset to center
    onChange(min + (max - min) * 0.5);
  }, [min, max, onChange]);

  return (
    <div
      className="flex flex-col items-center gap-1"
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
    >
      <canvas
        ref={canvasRef}
        style={{ width: size, height: size, cursor: 'ns-resize' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onDoubleClick={onDoubleClick}
      />
      {hovering && displayValue && (
        <span className="text-[10px] text-text-primary tabular-nums">
          {displayValue}
        </span>
      )}
      <span className="text-[10px] text-text-secondary tracking-wider uppercase">
        {label}
      </span>
    </div>
  );
}
