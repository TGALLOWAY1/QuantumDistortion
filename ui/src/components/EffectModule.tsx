import type { ReactNode } from 'react';

interface EffectModuleProps {
  title: string;
  color: string;
  enabled: boolean;
  onToggle: () => void;
  onRemove?: () => void;
  subtitle?: string;
  subtitleOptions?: string[];
  onSubtitleChange?: (val: string) => void;
  icon?: ReactNode;
  children: ReactNode;
}

export function EffectModule({
  title,
  color,
  enabled,
  onToggle,
  onRemove,
  subtitle,
  subtitleOptions,
  onSubtitleChange,
  icon,
  children,
}: EffectModuleProps) {
  return (
    <div
      className="flex flex-col rounded-xl overflow-hidden transition-opacity"
      style={{
        background: enabled
          ? `linear-gradient(180deg, ${color}18 0%, #14142b 100%)`
          : '#14142b',
        opacity: enabled ? 1 : 0.5,
        border: `1px solid ${enabled ? color + '40' : '#2a2a4a'}`,
        minWidth: 180,
      }}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2">
        <button
          onClick={onToggle}
          className="w-4 h-4 rounded-full border-2 flex items-center justify-center transition-colors"
          style={{
            borderColor: color,
            background: enabled ? color : 'transparent',
          }}
        >
          {enabled && (
            <div className="w-1.5 h-1.5 rounded-full bg-white" />
          )}
        </button>
        <span className="text-xs font-semibold tracking-wider uppercase text-text-primary">
          {title}
        </span>
        <div className="flex-1" />
        {icon && <span className="text-text-secondary text-sm">{icon}</span>}
        {onRemove && (
          <button
            onClick={(e) => { e.stopPropagation(); onRemove(); }}
            className="w-5 h-5 rounded-full flex items-center justify-center text-text-dim hover:text-red-400 hover:bg-red-400/10 transition-colors text-xs leading-none"
            title="Remove module"
          >
            ✕
          </button>
        )}
      </div>

      {/* Subtitle / type selector */}
      {subtitle && (
        <div className="flex items-center justify-center gap-2 px-3 pb-1">
          {subtitleOptions && onSubtitleChange ? (
            <>
              <button
                className="text-text-dim hover:text-text-primary text-xs"
                onClick={() => {
                  const idx = subtitleOptions.indexOf(subtitle);
                  const prev = (idx - 1 + subtitleOptions.length) % subtitleOptions.length;
                  onSubtitleChange(subtitleOptions[prev]);
                }}
              >
                &#9664;
              </button>
              <span className="text-xs text-text-secondary min-w-[70px] text-center capitalize">
                {subtitle}
              </span>
              <button
                className="text-text-dim hover:text-text-primary text-xs"
                onClick={() => {
                  const idx = subtitleOptions.indexOf(subtitle);
                  const next = (idx + 1) % subtitleOptions.length;
                  onSubtitleChange(subtitleOptions[next]);
                }}
              >
                &#9654;
              </button>
            </>
          ) : (
            <span className="text-xs text-text-secondary capitalize">{subtitle}</span>
          )}
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-end justify-center gap-4 px-3 py-3">
        {children}
      </div>
    </div>
  );
}
