import { useCallback, useRef, useState, useEffect } from 'react';
import { Toolbar } from './components/Toolbar';
import { InfoBar } from './components/InfoBar';
import { SpectrumAnalyzer } from './components/SpectrumAnalyzer';
import { EffectModule } from './components/EffectModule';
import { Knob } from './components/Knob';
import { useAudioEngine } from './hooks/useAudioEngine';
import type { EQBand } from './audio/engine';

// Electron API type
declare global {
  interface Window {
    electronAPI?: {
      openAudioFile: () => Promise<{ name: string; buffer: ArrayBuffer } | null>;
      saveAudioFile: (buffer: ArrayBuffer, name: string) => Promise<boolean>;
    };
  }
}

export default function App() {
  const {
    engine,
    isPlaying,
    fileName,
    duration,
    currentTime,
    params,
    init,
    loadFile,
    togglePlay,
    stop,
    seek,
    updateParams,
  } = useAudioEngine();

  const containerRef = useRef<HTMLDivElement>(null);
  const [specWidth, setSpecWidth] = useState(1100);
  const specHeight = 320;
  const [selectedBand, setSelectedBand] = useState<number | null>(null);

  // Resize observer for spectrum width
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
    // Ensure audio context is initialized (requires user gesture)
    await init();

    if (window.electronAPI) {
      const result = await window.electronAPI.openAudioFile();
      if (result) {
        await loadFile(result.buffer, result.name);
      }
    } else {
      // Fallback: browser file input
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'audio/*';
      input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) return;
        const buffer = await file.arrayBuffer();
        await loadFile(buffer, file.name);
      };
      input.click();
    }
  }, [init, loadFile]);

  const handleBandChange = useCallback((index: number, changes: Partial<EQBand>) => {
    setSelectedBand(index);
    const newBands = [...params.eqBands];
    newBands[index] = { ...newBands[index], ...changes };
    updateParams({ eqBands: newBands });
  }, [params.eqBands, updateParams]);

  return (
    <div className="h-screen flex flex-col bg-surface-0">
      {/* Top Toolbar */}
      <Toolbar
        bypass={params.bypass}
        onBypassToggle={() => updateParams({ bypass: !params.bypass })}
        dryWet={params.dryWet}
        onDryWetChange={(v) => updateParams({ dryWet: v })}
        onOpenFile={handleOpenFile}
        fileName={fileName}
        isPlaying={isPlaying}
        onTogglePlay={togglePlay}
        onStop={() => { stop(); seek(0); }}
        currentTime={currentTime}
        duration={duration}
        onSeek={seek}
      />

      {/* Spectrum Display */}
      <div ref={containerRef} className="flex-1 min-h-0 px-2 py-1">
        <SpectrumAnalyzer
          analyser={engine?.getAnalyser() ?? null}
          eqBands={params.eqBands}
          onBandChange={handleBandChange}
          width={specWidth}
          height={specHeight}
        />
      </div>

      {/* Info Bar */}
      <InfoBar selectedBand={selectedBand} eqBands={params.eqBands} />

      {/* Effect Modules Row */}
      <div className="flex gap-2 px-2 py-3 overflow-x-auto">
        {/* SATURATE */}
        <EffectModule
          title="Saturate"
          color="#c45e3e"
          enabled={params.saturateEnabled}
          onToggle={() => updateParams({ saturateEnabled: !params.saturateEnabled })}
          subtitle={params.saturateType}
          subtitleOptions={['tape', 'tube', 'wavefold']}
          onSubtitleChange={(v) => updateParams({ saturateType: v as 'tape' | 'tube' | 'wavefold' })}
        >
          <Knob
            label="Drive"
            value={params.saturateDrive}
            onChange={(v) => updateParams({ saturateDrive: v })}
            color="#c45e3e"
            displayValue={`${Math.round(params.saturateDrive * 100)}%`}
          />
          <Knob
            label="Tilt"
            value={params.saturateTilt}
            onChange={(v) => updateParams({ saturateTilt: v })}
            color="#c45e3e"
            displayValue={`${((params.saturateTilt - 0.5) * 100).toFixed(0)}`}
          />
        </EffectModule>

        {/* QUANTIZE (unique to this plugin) */}
        <EffectModule
          title="Quantize"
          color="#4ec48a"
          enabled={params.quantizeEnabled}
          onToggle={() => updateParams({ quantizeEnabled: !params.quantizeEnabled })}
          subtitle={`${['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][params.quantizeKey]} ${params.quantizeScale}`}
          subtitleOptions={['major', 'minor', 'pentatonic', 'dorian', 'mixolydian', 'harmonic_minor']}
          onSubtitleChange={(v) => updateParams({ quantizeScale: v })}
        >
          <Knob
            label="Strength"
            value={params.quantizeStrength}
            onChange={(v) => updateParams({ quantizeStrength: v })}
            color="#4ec48a"
            displayValue={`${Math.round(params.quantizeStrength * 100)}%`}
          />
          <Knob
            label="Key"
            value={params.quantizeKey}
            onChange={(v) => updateParams({ quantizeKey: Math.round(v) })}
            color="#4ec48a"
            min={0}
            max={11}
            displayValue={['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][Math.round(params.quantizeKey)]}
          />
        </EffectModule>

        {/* DELAY */}
        <EffectModule
          title="Delay"
          color="#3eafc4"
          enabled={params.delayEnabled}
          onToggle={() => updateParams({ delayEnabled: !params.delayEnabled })}
          subtitle="Classic"
        >
          <Knob
            label="Time"
            value={params.delayTime}
            onChange={(v) => updateParams({ delayTime: v })}
            color="#3eafc4"
            min={0.01}
            max={1.0}
            displayValue={`${Math.round(params.delayTime * 1000)}ms`}
          />
          <Knob
            label="Feedback"
            value={params.delayFeedback}
            onChange={(v) => updateParams({ delayFeedback: v })}
            color="#3eafc4"
            displayValue={`${Math.round(params.delayFeedback * 100)}%`}
          />
        </EffectModule>

        {/* MODULATE */}
        <EffectModule
          title="Modulate"
          color="#8a5ec4"
          enabled={params.modEnabled}
          onToggle={() => updateParams({ modEnabled: !params.modEnabled })}
          subtitle="Chorus"
        >
          <Knob
            label="Depth"
            value={params.modDepth}
            onChange={(v) => updateParams({ modDepth: v })}
            color="#8a5ec4"
            displayValue={`${Math.round(params.modDepth * 100)}%`}
          />
          <Knob
            label="Rate"
            value={params.modRate}
            onChange={(v) => updateParams({ modRate: v })}
            color="#8a5ec4"
            min={0.1}
            max={10}
            displayValue={`${params.modRate.toFixed(1)} Hz`}
          />
        </EffectModule>

        {/* LO-FI */}
        <EffectModule
          title="Lo-Fi"
          color="#8a7e5e"
          enabled={params.lofiEnabled}
          onToggle={() => updateParams({ lofiEnabled: !params.lofiEnabled })}
          subtitle="Cassette"
        >
          <Knob
            label="Wear"
            value={params.lofiWear}
            onChange={(v) => updateParams({ lofiWear: v })}
            color="#8a7e5e"
            displayValue={`${Math.round(params.lofiWear * 100)}%`}
          />
          <Knob
            label="Wobble"
            value={params.lofiWobble}
            onChange={(v) => updateParams({ lofiWobble: v })}
            color="#8a7e5e"
            displayValue={`${Math.round(params.lofiWobble * 100)}%`}
          />
        </EffectModule>
      </div>
    </div>
  );
}
