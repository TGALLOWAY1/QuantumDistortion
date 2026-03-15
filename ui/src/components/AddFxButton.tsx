import { useState, useRef, useEffect } from 'react';
import { FX_CATALOG } from '../audio/engine';
import type { FxType } from '../audio/engine';

interface AddFxButtonProps {
  onAdd: (type: FxType) => void;
}

const FX_TYPES = Object.keys(FX_CATALOG) as FxType[];

export function AddFxButton({ onAdd }: AddFxButtonProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

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
        onClick={() => setOpen(!open)}
        className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border
          hover:border-text-dim hover:bg-surface-2/50 transition-colors cursor-pointer"
        style={{ minWidth: 80, minHeight: 140 }}
        title="Add effect module"
      >
        <span className="text-2xl text-text-dim">+</span>
        <span className="text-[10px] text-text-dim mt-1">Add FX</span>
      </button>

      {open && (
        <div
          className="fixed bg-surface-2 border border-border rounded-lg shadow-xl py-1 min-w-[140px]"
          style={{
            zIndex: 9999,
            bottom: ref.current
              ? window.innerHeight - ref.current.getBoundingClientRect().top + 8
              : 'auto',
            right: ref.current
              ? window.innerWidth - ref.current.getBoundingClientRect().right
              : 0,
          }}
        >
          {FX_TYPES.map((type) => {
            const meta = FX_CATALOG[type];
            return (
              <button
                key={type}
                onClick={() => { onAdd(type); setOpen(false); }}
                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-text-secondary hover:text-text-primary hover:bg-surface-3 transition-colors"
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
