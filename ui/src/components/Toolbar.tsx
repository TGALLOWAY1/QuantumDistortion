interface ToolbarProps {
  bypass: boolean;
  onBypassToggle: () => void;
  onOpenFile: () => void;
  fileName: string | null;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onStop: () => void;
}

export function Toolbar({
  bypass,
  onBypassToggle,
  onOpenFile,
  fileName,
  isPlaying,
  onTogglePlay,
  onStop,
}: ToolbarProps) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-surface-1 border-b border-border">
      <div className="flex items-center gap-2 mr-2">
        <div className="w-5 h-5 rounded bg-accent-retune/20 flex items-center justify-center">
          <span className="text-accent-retune text-[10px] font-bold">Q</span>
        </div>
        <span className="text-xs font-semibold tracking-[0.16em] text-text-primary">
          QUANTUM DISTORTION
        </span>
      </div>

      <div className="w-px h-5 bg-border" />

      <button className="px-2 py-1 text-[10px] rounded bg-surface-2 border border-border text-text-secondary hover:text-text-primary">◀</button>
      <select className="px-2 py-1 text-[10px] rounded bg-surface-2 border border-border text-text-primary min-w-36">
        <option>Default Preset</option>
        <option>Modern Bass Tight</option>
        <option>Wide Harmonics</option>
      </select>
      <button className="px-2 py-1 text-[10px] rounded bg-surface-2 border border-border text-text-secondary hover:text-text-primary">▶</button>

      <button
        onClick={onBypassToggle}
        className={`px-2.5 py-1 text-[10px] rounded font-medium border ${
          bypass
            ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40'
            : 'bg-surface-2 text-text-secondary border-border hover:text-text-primary'
        }`}
      >
        BYPASS
      </button>
      <button className="px-2 py-1 text-[10px] rounded bg-surface-2 border border-border text-text-secondary">A/B</button>
      <button className="px-2 py-1 text-[10px] rounded bg-surface-2 border border-border text-text-secondary">UNDO</button>
      <button className="px-2 py-1 text-[10px] rounded bg-surface-2 border border-border text-text-secondary">REDO</button>
      <button className="px-2 py-1 text-[10px] rounded bg-surface-2 border border-border text-text-secondary">SET</button>

      <div className="flex-1" />

      <button
        onClick={onOpenFile}
        className="px-2.5 py-1 text-[10px] rounded bg-surface-2 border border-border text-text-secondary hover:text-text-primary"
      >
        {fileName ?? 'OPEN'}
      </button>
      <button
        onClick={onTogglePlay}
        disabled={!fileName}
        className="w-7 h-7 rounded-full bg-surface-2 border border-border flex items-center justify-center disabled:opacity-30"
      >
        <span className="text-text-primary text-[10px]">{isPlaying ? '❚❚' : '▶'}</span>
      </button>
      <button
        onClick={onStop}
        disabled={!fileName}
        className="w-7 h-7 rounded-full bg-surface-2 border border-border flex items-center justify-center disabled:opacity-30"
      >
        <span className="text-text-primary text-[10px]">■</span>
      </button>
    </div>
  );
}
