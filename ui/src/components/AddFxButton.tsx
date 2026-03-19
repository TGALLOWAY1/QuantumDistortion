import { useState, useRef, useEffect } from 'react';
import { FX_CATALOG } from '../audio/engine';
import type { FxType } from '../audio/engine';

interface AddFxButtonProps {
  onAdd: (type: FxType) => void;
  availableTypes: FxType[];
}

const FX_TYPES = Object.keys(FX_CATALOG) as FxType[];

export function AddFxButton({ onAdd, availableTypes }: AddFxButtonProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const visibleTypes = availableTypes.length > 0 ? availableTypes : FX_TYPES;

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('pointerdown', handleClick);
    return () => document.removeEventListener('pointerdown', handleClick);
  }, [open]);

  return (
    <div ref={ref} className="relative flex-shrink-0">
      <button
        onClick={() => {
          if (availableTypes.length > 0) setOpen(!open);
        }}
        disabled={availableTypes.length === 0}
        className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border
          hover:border-text-dim hover:bg-surface-2/50 transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
        style={{ minWidth: 80, minHeight: 140 }}
        title="Add effect module"
      >
        <span className="text-2xl text-text-dim">+</span>
        <span className="text-[10px] text-text-dim mt-1">
          {availableTypes.length > 0 ? 'Show FX' : 'All Shown'}
        </span>
      </button>

      {open && (
        <div className="absolute bottom-full left-0 mb-2 bg-surface-2 border border-border rounded-lg shadow-xl z-50 py-1 min-w-[140px]">
          {visibleTypes.map((type) => {
            const meta = FX_CATALOG[type];
            return (
              <button
                key={type}
                onClick={() => { onAdd(type); setOpen(false); }}
                className="flex items-center gap-2 w-full px-3 py-1.5 text-xs text-text-secondary hover:text-text-primary hover:bg-surface-3 transition-colors"
              >
                <div
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ background: meta.color }}
                />
                {meta.label}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
