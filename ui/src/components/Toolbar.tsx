interface ToolbarProps {
  bypass: boolean;
  onBypassToggle: () => void;
  dryWet: number;
  onDryWetChange: (v: number) => void;
  onOpenFile: () => void;
  fileName: string | null;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onStop: () => void;
  currentTime: number;
  duration: number;
  onSeek: (t: number) => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function Toolbar({
  bypass,
  onBypassToggle,
  dryWet,
  onDryWetChange,
  onOpenFile,
  fileName,
  isPlaying,
  onTogglePlay,
  onStop,
  currentTime,
  duration,
  onSeek,
}: ToolbarProps) {
  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-surface-1 border-b border-border">
      {/* Logo */}
      <div className="flex items-center gap-2 mr-2">
        <div className="w-6 h-6 rounded bg-accent-quantize/20 flex items-center justify-center">
          <span className="text-accent-quantize text-xs font-bold">Q</span>
        </div>
        <span className="text-sm font-semibold tracking-wide text-text-primary">
          QUANTUM DISTORTION
        </span>
      </div>

      {/* Separator */}
      <div className="w-px h-6 bg-border" />

      {/* Bypass */}
      <button
        onClick={onBypassToggle}
        className={`px-3 py-1 text-xs rounded font-medium transition-colors ${
          bypass
            ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/40'
            : 'bg-surface-2 text-text-secondary border border-border hover:text-text-primary'
        }`}
      >
        Bypass
      </button>

      {/* Separator */}
      <div className="w-px h-6 bg-border" />

      {/* File / Transport */}
      <button
        onClick={onOpenFile}
        className="px-3 py-1 text-xs rounded bg-surface-2 text-text-secondary border border-border hover:text-text-primary hover:border-text-dim transition-colors"
      >
        {fileName ?? 'Open File'}
      </button>

      <button
        onClick={onTogglePlay}
        disabled={!fileName}
        className="w-7 h-7 rounded-full bg-surface-2 border border-border flex items-center justify-center hover:border-text-dim disabled:opacity-30 transition-colors"
      >
        {isPlaying ? (
          <span className="text-text-primary text-[10px]">&#9646;&#9646;</span>
        ) : (
          <span className="text-text-primary text-xs ml-0.5">&#9654;</span>
        )}
      </button>

      <button
        onClick={onStop}
        disabled={!fileName}
        className="w-7 h-7 rounded-full bg-surface-2 border border-border flex items-center justify-center hover:border-text-dim disabled:opacity-30 transition-colors"
      >
        <span className="text-text-primary text-[10px]">&#9632;</span>
      </button>

      {/* Time display */}
      {duration > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-xs tabular-nums text-text-secondary">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
          <input
            type="range"
            min={0}
            max={duration}
            step={0.01}
            value={currentTime}
            onChange={(e) => onSeek(parseFloat(e.target.value))}
            className="w-32"
          />
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Amount / Dry-Wet */}
      <span className="text-xs text-text-secondary">Amount</span>
      <div className="flex items-center gap-2 w-36">
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={dryWet}
          onChange={(e) => onDryWetChange(parseFloat(e.target.value))}
          className="flex-1"
          style={{
            background: `linear-gradient(to right, #3eafc4 ${dryWet * 100}%, #2a2a4a ${dryWet * 100}%)`,
          }}
        />
        <span className="text-xs tabular-nums text-text-dim w-8 text-right">
          {Math.round(dryWet * 100)}%
        </span>
      </div>
    </div>
  );
}
