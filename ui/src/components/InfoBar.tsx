import type { EQBand, FilterType } from '../audio/engine';

interface InfoBarProps {
  selectedBand: number | null;
  eqBands: EQBand[];
  onShapeChange: (type: FilterType) => void;
}

function formatFreq(f: number): string {
  if (f >= 1000) return (f / 1000).toFixed(1) + ' kHz';
  return Math.round(f) + ' Hz';
}

function ShapeIcon({ type }: { type: FilterType }) {
  return (
    <svg width="18" height="12" viewBox="0 0 18 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      {type === 'peak' && (
        <path d="M1 9 Q5 9 7 3 Q9 -1 11 3 Q13 9 17 9" />
      )}
      {type === 'lowshelf' && (
        <path d="M1 3 L5 3 Q7 3 9 9 L17 9" />
      )}
      {type === 'highshelf' && (
        <path d="M1 9 L9 9 Q11 9 13 3 L17 3" />
      )}
      {type === 'lowpass' && (
        <path d="M1 4 L8 4 Q11 4 13 7 Q15 11 17 11" />
      )}
      {type === 'highpass' && (
        <path d="M1 11 Q3 11 5 7 Q7 4 10 4 L17 4" />
      )}
    </svg>
  );
}

const FILTER_TYPES: FilterType[] = ['peak', 'lowshelf', 'highshelf', 'lowpass', 'highpass'];
const FILTER_LABELS: Record<FilterType, string> = {
  peak: 'Peak',
  lowshelf: 'Low Shelf',
  highshelf: 'High Shelf',
  lowpass: 'Low Pass',
  highpass: 'High Pass',
};

export function InfoBar({ selectedBand, eqBands, onShapeChange }: InfoBarProps) {
  const band = selectedBand !== null ? eqBands[selectedBand] : null;
  const activeType = band?.type ?? null;

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

      {/* Filter shape selector */}
      <div className="flex items-center gap-1.5">
        <span className="text-text-dim mr-1">Shape</span>
        {FILTER_TYPES.map((shape) => {
          const isActive = activeType === shape;
          return (
            <button
              key={shape}
              onClick={() => onShapeChange(shape)}
              disabled={selectedBand === null}
              className={`w-9 h-7 rounded flex items-center justify-center border transition-colors ${
                isActive
                  ? 'bg-accent-retune/20 border-accent-retune/50 text-accent-retune'
                  : 'bg-surface-2 border-border text-text-dim hover:text-text-primary hover:border-text-dim'
              } ${selectedBand === null ? 'opacity-30 cursor-not-allowed' : ''}`}
              title={FILTER_LABELS[shape]}
            >
              <ShapeIcon type={shape} />
            </button>
          );
        })}
      </div>
    </div>
  );
}
