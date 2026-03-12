import type { EQBand } from '../audio/engine';

interface InfoBarProps {
  selectedBand: number | null;
  eqBands: EQBand[];
}

function formatFreq(f: number): string {
  if (f >= 1000) return (f / 1000).toFixed(1) + ' kHz';
  return Math.round(f) + ' Hz';
}

export function InfoBar({ selectedBand, eqBands }: InfoBarProps) {
  const band = selectedBand !== null ? eqBands[selectedBand] : null;

  return (
    <div className="flex items-center gap-6 px-4 py-1.5 bg-surface-1 border-y border-border text-xs">
      <span className="text-text-dim">
        {selectedBand !== null ? `Band ${selectedBand + 1}` : '\u00A0'}
      </span>
      <span className="text-text-secondary">
        Freq <span className="text-text-primary tabular-nums">{band ? formatFreq(band.freq) : '—'}</span>
      </span>
      <span className="text-text-secondary">
        Gain <span className="text-text-primary tabular-nums">{band ? band.gain.toFixed(1) + ' dB' : '—'}</span>
      </span>
      <span className="text-text-secondary">
        Q <span className="text-text-primary tabular-nums">{band ? band.q.toFixed(1) : '—'}</span>
      </span>

      <div className="flex-1" />

      {/* Filter shapes (visual only for now) */}
      <div className="flex items-center gap-1">
        <span className="text-text-dim">Shape</span>
        {['peak', 'lowshelf', 'highshelf', 'lowpass', 'highpass'].map((shape) => (
          <button
            key={shape}
            className="w-6 h-5 rounded bg-surface-2 border border-border text-[8px] text-text-dim hover:text-text-primary hover:border-text-dim transition-colors"
            title={shape}
          >
            {shape === 'peak' ? '∧' : shape === 'lowshelf' ? '⌐' : shape === 'highshelf' ? '¬' : shape === 'lowpass' ? '╲' : '╱'}
          </button>
        ))}
      </div>
    </div>
  );
}
