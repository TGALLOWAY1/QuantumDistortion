import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { Toolbar } from './components/Toolbar';
import { SpectrumAnalyzer } from './components/SpectrumAnalyzer';
import { RetuneModule } from './components/RetuneModule';
import { Knob } from './components/Knob';
import { useAudioEngine } from './hooks/useAudioEngine';

declare global {
  interface Window {
    electronAPI?: {
      openAudioFile: () => Promise<{ name: string; buffer: ArrayBuffer } | null>;
    };
  }
}

const SATURATION_TYPE_OPTIONS = ['Analog Tape', 'Tube', 'Diode Clip', 'Foldback', 'Soft Clip'];
const SOFTNESS = ['Soft', 'Medium', 'Hard'];
const NOTE_OPTIONS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function getSaturationTypeLabel(type: 'tape' | 'tube' | 'wavefold'): string {
  if (type === 'wavefold') return 'Foldback';
  if (type === 'tube') return 'Tube';
  return 'Analog Tape';
}

function getSaturationTypeValue(label: string): 'tape' | 'tube' | 'wavefold' {
  if (label === 'Tube') return 'tube';
  if (label === 'Foldback') return 'wavefold';
  return 'tape';
}

function ModuleShell({
  number,
  title,
  color,
  enabled,
  onToggle,
  minWidth,
  children,
}: {
  number?: string;
  title: string;
  color: string;
  enabled: boolean;
  onToggle: () => void;
  minWidth: number;
  children: ReactNode;
}) {
  return (
    <div
      className="flex flex-col rounded-xl overflow-hidden flex-shrink-0 backdrop-blur-[1px]"
      style={{
        background: `linear-gradient(180deg, #11182d 0%, #0f1324 100%)`,
        border: `1px solid ${color}33`,
        minWidth,
        opacity: enabled ? 1 : 0.6,
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.02)',
      }}
    >
      <div className="flex items-center gap-2 px-2.5 py-1.5 border-b" style={{ borderColor: '#23314d' }}>
        {number && (
          <span
            className="w-[16px] h-[16px] rounded-full text-[10px] font-semibold flex items-center justify-center"
            style={{ background: `${color}2a`, color }}
          >
            {number}
          </span>
        )}
        <span className="text-xs font-semibold tracking-wider uppercase text-text-primary">{title}</span>
        <div className="flex-1" />
        <button
          onClick={onToggle}
          className="w-4 h-4 rounded-full border"
          style={{ borderColor: color, background: enabled ? `${color}22` : 'transparent' }}
        />
      </div>
      <div className="px-2 py-2 flex flex-col gap-1.5">{children}</div>
    </div>
  );
}

export default function App() {
  const {
    engine,
    isPlaying,
    fileName,
    params,
    init,
    loadFile,
    togglePlay,
    stop,
    updateParams,
  } = useAudioEngine();

  const containerRef = useRef<HTMLDivElement>(null);
  const [specWidth, setSpecWidth] = useState(1100);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setSpecWidth(entry.contentRect.width);
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const handleOpenFile = useCallback(async () => {
    await init();

    if (window.electronAPI) {
      const result = await window.electronAPI.openAudioFile();
      if (result) {
        await loadFile(result.buffer, result.name);
      }
      return;
    }

    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'audio/*';
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      await loadFile(await file.arrayBuffer(), file.name);
    };
    input.click();
  }, [init, loadFile]);

  return (
    <div className="h-screen flex flex-col bg-surface-0">
      <Toolbar
        bypass={params.bypass}
        onBypassToggle={() => updateParams({ bypass: !params.bypass })}
        onOpenFile={handleOpenFile}
        fileName={fileName}
        isPlaying={isPlaying}
        onTogglePlay={togglePlay}
        onStop={() => stop()}
      />

      <div ref={containerRef} className="px-2 pt-2">
        <SpectrumAnalyzer
          analyser={engine?.getAnalyser() ?? null}
          eqBands={params.eqBands}
          onBandChange={(index, band) => {
            const newBands = [...params.eqBands];
            newBands[index] = { ...newBands[index], ...band };
            updateParams({ eqBands: newBands });
          }}
          width={specWidth}
          height={350}
          retuneBandStartHz={200}
          retuneBandEndHz={5000}
          retuneBandColor="#4891ff"
        />
      </div>

      <div className="flex gap-2 px-2 py-3 overflow-x-auto items-stretch">
        <ModuleShell
          number="1"
          title="Saturator"
          color="#d88b52"
          enabled={params.saturateEnabled}
          onToggle={() => updateParams({ saturateEnabled: !params.saturateEnabled })}
          minWidth={220}
        >
          <div className="grid grid-cols-2 gap-1.5">
            <select
              className="text-xs rounded bg-surface-2 border border-border px-2 py-1 text-text-primary"
              value={getSaturationTypeLabel(params.saturateType)}
              onChange={(e) => updateParams({ saturateType: getSaturationTypeValue(e.target.value) })}
            >
              {SATURATION_TYPE_OPTIONS.map((type) => <option key={type}>{type}</option>)}
            </select>
            <select value={params.saturateSoftness} onChange={(e) => updateParams({ saturateSoftness: e.target.value as 'soft' | 'medium' | 'hard' })} className="text-xs rounded bg-surface-2 border border-border px-2 py-1 text-text-primary">
              {SOFTNESS.map((softness) => <option key={softness} value={softness.toLowerCase()}>{softness}</option>)}
            </select>
          </div>
          <div className="flex justify-around items-end gap-1">
            <Knob label="Drive" size={48} value={params.saturateDrive} onChange={(v) => updateParams({ saturateDrive: v })} color="#d88b52" displayValue={`${Math.round(params.saturateDrive * 100)}%`} />
            <Knob label="Tone" size={48} value={params.saturateTilt} onChange={(v) => updateParams({ saturateTilt: v })} color="#d88b52" displayValue={`${Math.round((params.saturateTilt - 0.5) * 200)}`} />
            <Knob label="Mix" size={48} value={params.saturateMix} onChange={(v) => updateParams({ saturateMix: v })} color="#d88b52" displayValue={`${Math.round(params.saturateMix * 100)}%`} />
          </div>
        </ModuleShell>

        <RetuneModule color="#4891ff" params={params} updateParams={updateParams} />

        <ModuleShell
          number="3"
          title="Saturator"
          color="#9d6ee9"
          enabled={params.saturate2Enabled}
          onToggle={() => updateParams({ saturate2Enabled: !params.saturate2Enabled })}
          minWidth={220}
        >
          <div className="grid grid-cols-2 gap-1.5">
            <select
              className="text-xs rounded bg-surface-2 border border-border px-2 py-1 text-text-primary"
              value={getSaturationTypeLabel(params.saturate2Type)}
              onChange={(e) => updateParams({ saturate2Type: getSaturationTypeValue(e.target.value) })}
            >
              {SATURATION_TYPE_OPTIONS.map((type) => <option key={type}>{type}</option>)}
            </select>
            <select value={params.saturate2Softness} onChange={(e) => updateParams({ saturate2Softness: e.target.value as 'soft' | 'medium' | 'hard' })} className="text-xs rounded bg-surface-2 border border-border px-2 py-1 text-text-primary">
              {SOFTNESS.map((softness) => <option key={softness} value={softness.toLowerCase()}>{softness}</option>)}
            </select>
          </div>
          <div className="flex justify-around items-end gap-1">
            <Knob label="Drive" size={48} value={params.saturate2Drive} onChange={(v) => updateParams({ saturate2Drive: v })} color="#9d6ee9" displayValue={`${Math.round(params.saturate2Drive * 100)}%`} />
            <Knob label="Tone" size={48} value={params.saturate2Tilt} onChange={(v) => updateParams({ saturate2Tilt: v })} color="#9d6ee9" displayValue={`${Math.round((params.saturate2Tilt - 0.5) * 200)}`} />
            <Knob label="Mix" size={48} value={params.saturate2Mix} onChange={(v) => updateParams({ saturate2Mix: v })} color="#9d6ee9" displayValue={`${Math.round(params.saturate2Mix * 100)}%`} />
          </div>
        </ModuleShell>

        <ModuleShell
          title="Sub"
          color="#34c7cf"
          enabled={params.subEnabled}
          onToggle={() => updateParams({ subEnabled: !params.subEnabled })}
          minWidth={220}
        >
          <div className="grid grid-cols-3 gap-1.5">
            <select value={params.subRootNote} onChange={(e) => updateParams({ subRootNote: Number(e.target.value) })} className="text-xs rounded bg-surface-2 border border-border px-1.5 py-1 text-text-primary">
              {NOTE_OPTIONS.map((note, i) => <option key={note} value={i}>{note}</option>)}
            </select>
            <select value={params.subOctave} onChange={(e) => updateParams({ subOctave: Number(e.target.value) as -2 | -1 | 0 })} className="text-xs rounded bg-surface-2 border border-border px-1.5 py-1 text-text-primary">
              <option value={-2}>-2</option><option value={-1}>-1</option><option value={0}>0</option>
            </select>
            <select value={params.subWaveform} onChange={(e) => updateParams({ subWaveform: e.target.value as 'sine' | 'triangle' | 'rounded_square' })} className="text-xs rounded bg-surface-2 border border-border px-1.5 py-1 text-text-primary">
              <option value="sine">Sine</option><option value="triangle">Tri</option><option value="rounded_square">Sqr</option>
            </select>
          </div>
          <div className="flex justify-center">
            <Knob label="Level" size={48} value={params.subLevel} onChange={(v) => updateParams({ subLevel: v })} color="#34c7cf" displayValue={`${Math.round(params.subLevel * 100)}%`} />
          </div>
        </ModuleShell>

        <ModuleShell title="Output" color="#3eafc4" enabled={true} onToggle={() => undefined} minWidth={240}>
          <div className="grid grid-cols-4 gap-1 justify-items-center">
            <Knob label="Low" size={48} value={params.lowGain} onChange={(v) => updateParams({ lowGain: v })} color="#d88b52" min={0} max={2} displayValue={`${(params.lowGain * 100).toFixed(0)}%`} />
            <Knob label="High" size={48} value={params.highGain} onChange={(v) => updateParams({ highGain: v })} color="#4891ff" min={0} max={2} displayValue={`${(params.highGain * 100).toFixed(0)}%`} />
            <Knob label="Dry/Wet" size={48} value={params.dryWet} onChange={(v) => updateParams({ dryWet: v })} color="#3eafc4" displayValue={`${Math.round(params.dryWet * 100)}%`} />
            <Knob label="Gain" size={48} value={params.masterGain} onChange={(v) => updateParams({ masterGain: v })} color="#3eafc4" min={0} max={2} displayValue={`${(params.masterGain * 100).toFixed(0)}%`} />
          </div>
          <select value={params.outputLimiterType} onChange={(e) => updateParams({ outputLimiterType: e.target.value as 'transparent' | 'soft' | 'hard' | 'off' })} className="text-xs rounded bg-surface-2 border border-border px-2 py-1 text-text-primary">
            <option value="transparent">Transparent</option><option value="soft">Soft</option><option value="hard">Hard</option><option value="off">Off</option>
          </select>
        </ModuleShell>

        <button
          type="button"
          className="flex flex-col items-center justify-center rounded-xl border border-dashed flex-shrink-0 text-text-dim hover:text-text-primary hover:border-text-secondary transition-colors"
          style={{ borderColor: '#23314d', minWidth: 140 }}
        >
          <span className="text-lg leading-none mb-1">+</span>
          <span className="text-[10px] tracking-wider uppercase">Add Effect</span>
        </button>
      </div>
    </div>
  );
}
