import { useCallback, useRef, useState, useEffect } from 'react';
import { Toolbar } from './components/Toolbar';
import { InfoBar } from './components/InfoBar';
import { SpectrumAnalyzer } from './components/SpectrumAnalyzer';
import { EffectModule } from './components/EffectModule';
import { Knob } from './components/Knob';
import { AddFxButton } from './components/AddFxButton';
import { useAudioEngine } from './hooks/useAudioEngine';
import { FX_CATALOG } from './audio/engine';
import type { EQBand, FilterType, FxSlot, FxType, EngineParams } from './audio/engine';

// Electron API type
declare global {
  interface Window {
    electronAPI?: {
      openAudioFile: () => Promise<{ name: string; buffer: ArrayBuffer } | null>;
      saveAudioFile: (buffer: ArrayBuffer, name: string) => Promise<boolean>;
    };
  }
}

// --- Output Module (always last, not removable) ---
function OutputModule({ params, updateParams }: {
  params: EngineParams;
  updateParams: (p: Partial<EngineParams>) => void;
}) {
  return (
    <div
      className="flex flex-col rounded-xl overflow-hidden flex-shrink-0"
      style={{
        background: 'linear-gradient(180deg, #3eafc418 0%, #14142b 100%)',
        border: '1px solid #3eafc440',
        minWidth: 180,
      }}
    >
      <div className="flex items-center gap-2 px-3 py-2">
        <span className="text-xs font-semibold tracking-wider uppercase text-text-primary">
          Output
        </span>
      </div>
      <div className="flex items-end justify-center gap-4 px-3 py-3 mt-auto">
        <Knob
          label="Dry/Wet"
          value={params.dryWet}
          onChange={(v) => updateParams({ dryWet: v })}
          color="#3eafc4"
          displayValue={`${Math.round(params.dryWet * 100)}%`}
        />
        <Knob
          label="Output"
          value={params.masterGain}
          onChange={(v) => updateParams({ masterGain: v })}
          color="#3eafc4"
          min={0}
          max={2}
          displayValue={`${(params.masterGain * 100).toFixed(0)}%`}
        />
      </div>
    </div>
  );
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
    restart,
    seek,
    updateParams,
  } = useAudioEngine();

  const containerRef = useRef<HTMLDivElement>(null);
  const [specWidth, setSpecWidth] = useState(1100);
  const specHeight = 320;
  const [selectedBand, setSelectedBand] = useState<number | null>(null);

  // --- Dynamic FX slots (default: Saturate → Quantize → Saturate 2) ---
  const [fxSlots, setFxSlots] = useState<FxSlot[]>([
    { id: crypto.randomUUID(), type: 'saturate' },
    { id: crypto.randomUUID(), type: 'quantize' },
    { id: crypto.randomUUID(), type: 'saturate2' },
  ]);

  const addFxSlot = useCallback((type: FxType) => {
    setFxSlots(prev => [...prev, { id: crypto.randomUUID(), type }]);
  }, []);

  const removeFxSlot = useCallback((id: string) => {
    setFxSlots(prev => prev.filter(s => s.id !== id));
  }, []);

  // --- Keyboard shortcuts ---
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (!fileName) return;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.code === 'KeyR' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        restart();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePlay, restart, fileName]);

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

  const handleShapeChange = useCallback((type: FilterType) => {
    if (selectedBand === null) return;
    const newBands = [...params.eqBands];
    newBands[selectedBand] = { ...newBands[selectedBand], type };
    updateParams({ eqBands: newBands });
  }, [selectedBand, params.eqBands, updateParams]);

  // --- Render an FX slot as the correct EffectModule ---
  function renderFxSlot(slot: FxSlot) {
    const meta = FX_CATALOG[slot.type];

    switch (slot.type) {
      case 'saturate':
        return (
          <EffectModule
            key={slot.id}
            title={meta.label}
            color={meta.color}
            enabled={params.saturateEnabled}
            onToggle={() => updateParams({ saturateEnabled: !params.saturateEnabled })}
            onRemove={() => removeFxSlot(slot.id)}
            subtitle={params.saturateType}
            subtitleOptions={['tape', 'tube', 'wavefold']}
            onSubtitleChange={(v) => updateParams({ saturateType: v as 'tape' | 'tube' | 'wavefold' })}
          >
            <Knob
              label="Drive"
              value={params.saturateDrive}
              onChange={(v) => updateParams({ saturateDrive: v })}
              color={meta.color}
              displayValue={`${Math.round(params.saturateDrive * 100)}%`}
            />
            <Knob
              label="Tilt"
              value={params.saturateTilt}
              onChange={(v) => updateParams({ saturateTilt: v })}
              color={meta.color}
              displayValue={`${((params.saturateTilt - 0.5) * 100).toFixed(0)}`}
            />
          </EffectModule>
        );

      case 'saturate2':
        return (
          <EffectModule
            key={slot.id}
            title={meta.label}
            color={meta.color}
            enabled={params.saturate2Enabled}
            onToggle={() => updateParams({ saturate2Enabled: !params.saturate2Enabled })}
            onRemove={() => removeFxSlot(slot.id)}
            subtitle={params.saturate2Type}
            subtitleOptions={['tape', 'tube', 'wavefold']}
            onSubtitleChange={(v) => updateParams({ saturate2Type: v as 'tape' | 'tube' | 'wavefold' })}
          >
            <Knob
              label="Drive"
              value={params.saturate2Drive}
              onChange={(v) => updateParams({ saturate2Drive: v })}
              color={meta.color}
              displayValue={`${Math.round(params.saturate2Drive * 100)}%`}
            />
            <Knob
              label="Tilt"
              value={params.saturate2Tilt}
              onChange={(v) => updateParams({ saturate2Tilt: v })}
              color={meta.color}
              displayValue={`${((params.saturate2Tilt - 0.5) * 100).toFixed(0)}`}
            />
          </EffectModule>
        );

      case 'quantize':
        {
        const noteNames = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
        const scaleDegreeCounts: Record<string, number> = {
          major: 7,
          minor: 7,
          pentatonic: 5,
          dorian: 7,
          mixolydian: 7,
          harmonic_minor: 7,
        };
        const subSources = ['root', 'manual', 'scale_degree'] as const;
        const subSourceIndex = subSources.indexOf(params.quantizeSubSource);
        const scaleDegreeMax = Math.max(0, (scaleDegreeCounts[params.quantizeScale] ?? 7) - 1);
        const subDegreeDisplay = `Deg ${Math.round(params.quantizeSubDegree) + 1}`;
        return (
          <EffectModule
            key={slot.id}
            title={meta.label}
            color={meta.color}
            enabled={params.quantizeEnabled}
            onToggle={() => updateParams({ quantizeEnabled: !params.quantizeEnabled })}
            onRemove={() => removeFxSlot(slot.id)}
            subtitle={params.quantizeScale}
            subtitleOptions={['major', 'minor', 'pentatonic', 'dorian', 'mixolydian', 'harmonic_minor']}
            onSubtitleChange={(v) => updateParams({ quantizeScale: v })}
          >
            <Knob
              label="Strength"
              value={params.quantizeStrength}
              onChange={(v) => updateParams({ quantizeStrength: v })}
              color={meta.color}
              displayValue={`${Math.round(params.quantizeStrength * 100)}%`}
            />
            <Knob
              label="Sub"
              value={params.quantizeSubEnabled ? 1 : 0}
              onChange={(v) => updateParams({ quantizeSubEnabled: Math.round(v) >= 1 })}
              color={meta.color}
              min={0}
              max={1}
              displayValue={params.quantizeSubEnabled ? 'On' : 'Off'}
            />
            <Knob
              label="Key"
              value={params.quantizeKey}
              onChange={(v) => updateParams({ quantizeKey: Math.round(v) })}
              color={meta.color}
              min={0}
              max={11}
              displayValue={noteNames[Math.round(params.quantizeKey)]}
            />
            <Knob
              label="Sub Src"
              value={subSourceIndex < 0 ? 0 : subSourceIndex}
              onChange={(v) => updateParams({ quantizeSubSource: subSources[Math.max(0, Math.min(subSources.length - 1, Math.round(v)))] })}
              color={meta.color}
              min={0}
              max={2}
              displayValue={params.quantizeSubSource === 'scale_degree' ? 'Degree' : params.quantizeSubSource}
            />
            <Knob
              label={params.quantizeSubSource === 'scale_degree' ? 'Degree' : 'Sub Note'}
              value={params.quantizeSubSource === 'scale_degree' ? params.quantizeSubDegree : params.quantizeSubNote}
              onChange={(v) => (
                params.quantizeSubSource === 'scale_degree'
                  ? updateParams({ quantizeSubDegree: Math.round(v) })
                  : updateParams({ quantizeSubNote: Math.round(v) })
              )}
              color={meta.color}
              min={0}
              max={params.quantizeSubSource === 'scale_degree' ? scaleDegreeMax : 11}
              displayValue={params.quantizeSubSource === 'scale_degree' ? subDegreeDisplay : noteNames[Math.round(params.quantizeSubNote)]}
            />
            <Knob
              label="Sub Oct"
              value={params.quantizeSubOctave}
              onChange={(v) => updateParams({ quantizeSubOctave: Math.round(v) })}
              color={meta.color}
              min={0}
              max={4}
              displayValue={`Oct ${Math.round(params.quantizeSubOctave)}`}
            />
            <Knob
              label="Sub Lvl"
              value={params.quantizeSubLevel}
              onChange={(v) => updateParams({ quantizeSubLevel: v })}
              color={meta.color}
              displayValue={`${Math.round(params.quantizeSubLevel * 100)}%`}
            />
            <Knob
              label="Air"
              value={params.quantizeAirMix}
              onChange={(v) => updateParams({ quantizeAirMix: v })}
              color={meta.color}
              displayValue={`${Math.round(params.quantizeAirMix * 100)}%`}
            />
          </EffectModule>
        );
        }

      case 'delay':
        return (
          <EffectModule
            key={slot.id}
            title={meta.label}
            color={meta.color}
            enabled={params.delayEnabled}
            onToggle={() => updateParams({ delayEnabled: !params.delayEnabled })}
            onRemove={() => removeFxSlot(slot.id)}
            subtitle="Classic"
          >
            <Knob
              label="Time"
              value={params.delayTime}
              onChange={(v) => updateParams({ delayTime: v })}
              color={meta.color}
              min={0.01}
              max={1.0}
              displayValue={`${Math.round(params.delayTime * 1000)}ms`}
            />
            <Knob
              label="Feedback"
              value={params.delayFeedback}
              onChange={(v) => updateParams({ delayFeedback: v })}
              color={meta.color}
              displayValue={`${Math.round(params.delayFeedback * 100)}%`}
            />
          </EffectModule>
        );

      case 'modulate':
        return (
          <EffectModule
            key={slot.id}
            title={meta.label}
            color={meta.color}
            enabled={params.modEnabled}
            onToggle={() => updateParams({ modEnabled: !params.modEnabled })}
            onRemove={() => removeFxSlot(slot.id)}
            subtitle="Chorus"
          >
            <Knob
              label="Depth"
              value={params.modDepth}
              onChange={(v) => updateParams({ modDepth: v })}
              color={meta.color}
              displayValue={`${Math.round(params.modDepth * 100)}%`}
            />
            <Knob
              label="Rate"
              value={params.modRate}
              onChange={(v) => updateParams({ modRate: v })}
              color={meta.color}
              min={0.1}
              max={10}
              displayValue={`${params.modRate.toFixed(1)} Hz`}
            />
          </EffectModule>
        );

      case 'lofi':
        return (
          <EffectModule
            key={slot.id}
            title={meta.label}
            color={meta.color}
            enabled={params.lofiEnabled}
            onToggle={() => updateParams({ lofiEnabled: !params.lofiEnabled })}
            onRemove={() => removeFxSlot(slot.id)}
            subtitle="Cassette"
          >
            <Knob
              label="Wear"
              value={params.lofiWear}
              onChange={(v) => updateParams({ lofiWear: v })}
              color={meta.color}
              displayValue={`${Math.round(params.lofiWear * 100)}%`}
            />
            <Knob
              label="Wobble"
              value={params.lofiWobble}
              onChange={(v) => updateParams({ lofiWobble: v })}
              color={meta.color}
              displayValue={`${Math.round(params.lofiWobble * 100)}%`}
            />
          </EffectModule>
        );
    }
  }

  return (
    <div className="h-screen flex flex-col bg-surface-0">
      {/* Top Toolbar */}
      <Toolbar
        bypass={params.bypass}
        onBypassToggle={() => updateParams({ bypass: !params.bypass })}
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
      <InfoBar
        selectedBand={selectedBand}
        eqBands={params.eqBands}
        onShapeChange={handleShapeChange}
      />

      {/* Effect Modules Row */}
      <div className="flex gap-2 px-2 py-3 overflow-x-auto items-start">
        {fxSlots.map(slot => renderFxSlot(slot))}
        <AddFxButton onAdd={addFxSlot} />
        <OutputModule params={params} updateParams={updateParams} />
      </div>
    </div>
  );
}
