import { useCallback, useRef, useState, useEffect } from 'react';
import { Toolbar } from './components/Toolbar';
import { InfoBar } from './components/InfoBar';
import { SpectrumAnalyzer } from './components/SpectrumAnalyzer';
import { EffectModule } from './components/EffectModule';
import { Knob } from './components/Knob';
import { AddFxButton } from './components/AddFxButton';
import { RetuneModule } from './components/RetuneModule';
import { useAudioEngine } from './hooks/useAudioEngine';
import { FX_CATALOG } from './audio/engine';
import type { EQBand, FilterType, FxSlot, FxType, EngineParams, PeqInstance } from './audio/engine';

const FX_ORDER: FxType[] = ['peq', 'saturate', 'retune', 'delay', 'modulate', 'lofi', 'saturate2', 'sub'];
const SINGLETON_FX: FxType[] = ['saturate', 'retune', 'delay', 'modulate', 'lofi', 'saturate2', 'sub'];

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
          label="Low"
          value={params.lowGain}
          onChange={(v) => updateParams({ lowGain: v })}
          color="#c45e3e"
          min={0}
          max={2}
          displayValue={`${(params.lowGain * 100).toFixed(0)}%`}
        />
        <Knob
          label="High"
          value={params.highGain}
          onChange={(v) => updateParams({ highGain: v })}
          color="#8ab4c4"
          min={0}
          max={2}
          displayValue={`${(params.highGain * 100).toFixed(0)}%`}
        />
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
    retuneStatus,
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
  const [showDevPanel, setShowDevPanel] = useState(false);

  // --- Dynamic FX slots (default: Saturate → Retune → Saturate 2) ---
  const [fxSlots, setFxSlots] = useState<FxSlot[]>([
    { id: crypto.randomUUID(), type: 'saturate' },
    { id: crypto.randomUUID(), type: 'retune' },
    { id: crypto.randomUUID(), type: 'saturate2' },
    { id: crypto.randomUUID(), type: 'sub' },
  ]);

  const addFxSlot = useCallback((type: FxType) => {
    if (SINGLETON_FX.includes(type) && fxSlots.some(slot => slot.type === type)) {
      return;
    }
    const newId = crypto.randomUUID();
    setFxSlots(prev => [...prev, { id: newId, type }]);
    if (type === 'peq') {
      updateParams({
        peqInstances: [...params.peqInstances, {
          id: newId,
          enabled: true,
          mode: 'cut',
          key: params.retuneKey,
          scale: params.retuneScale,
          amount: 0.5,
          q: 5.0,
        }],
      });
    }
  }, [fxSlots, params.peqInstances, params.retuneKey, params.retuneScale, updateParams]);

  const removeFxSlot = useCallback((id: string) => {
    const slot = fxSlots.find(s => s.id === id);
    if (!slot) return;

    setFxSlots(prev => prev.filter(s => s.id !== id));

    if (slot.type === 'peq') {
      updateParams({
        peqInstances: params.peqInstances.filter(inst => inst.id !== id),
      });
      return;
    }

    switch (slot.type) {
      case 'saturate':
        updateParams({ saturateEnabled: false });
        break;
      case 'retune':
        updateParams({ retuneEnabled: false });
        break;
      case 'delay':
        updateParams({ delayEnabled: false });
        break;
      case 'modulate':
        updateParams({ modEnabled: false });
        break;
      case 'lofi':
        updateParams({ lofiEnabled: false });
        break;
      case 'saturate2':
        updateParams({ saturate2Enabled: false });
        break;
      case 'sub':
        updateParams({ subEnabled: false });
        break;
    }
  }, [fxSlots, params.peqInstances, updateParams]);

  const orderedFxSlots = [...fxSlots].sort(
    (a, b) => FX_ORDER.indexOf(a.type) - FX_ORDER.indexOf(b.type)
  );

  const availableFxTypes = (Object.keys(FX_CATALOG) as FxType[]).filter((type) => (
    type === 'peq' || !orderedFxSlots.some((slot) => slot.type === type)
  ));

  // --- Keyboard shortcuts ---
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      // Dev panel toggle works without a file loaded
      if (e.code === 'KeyD' && e.shiftKey && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        setShowDevPanel(prev => !prev);
        return;
      }

      if (!fileName) return;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.code === 'KeyR' && !e.metaKey && !e.ctrlKey && !e.shiftKey) {
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

      case 'sub':
        return (
          <EffectModule
            key={slot.id}
            title={meta.label}
            color={meta.color}
            enabled={params.subEnabled}
            onToggle={() => updateParams({ subEnabled: !params.subEnabled })}
            onRemove={() => removeFxSlot(slot.id)}
            subtitle={retuneStatus?.lowEndLocked ? 'Locked' : 'Waiting'}
          >
            <Knob
              label="Level"
              value={params.subLevel}
              onChange={(v) => updateParams({ subLevel: v })}
              color={meta.color}
              displayValue={`${Math.round(params.subLevel * 100)}%`}
            />
          </EffectModule>
        );

      case 'retune':
        return (
          <RetuneModule
            key={slot.id}
            color={meta.color}
            params={params}
            status={retuneStatus}
            updateParams={updateParams}
            onRemove={() => removeFxSlot(slot.id)}
          />
        );

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

      case 'peq':
        {
        const peqNoteNames = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
        const inst = params.peqInstances.find(p => p.id === slot.id);
        if (!inst) return null;
        const instIdx = params.peqInstances.indexOf(inst);
        const updatePeq = (changes: Partial<PeqInstance>) => {
          const updated = [...params.peqInstances];
          updated[instIdx] = { ...updated[instIdx], ...changes };
          updateParams({ peqInstances: updated });
        };
        return (
          <EffectModule
            key={slot.id}
            title={meta.label}
            color={meta.color}
            enabled={inst.enabled}
            onToggle={() => updatePeq({ enabled: !inst.enabled })}
            onRemove={() => removeFxSlot(slot.id)}
            subtitle={inst.scale}
            subtitleOptions={['major', 'minor', 'pentatonic', 'dorian', 'mixolydian', 'harmonic_minor']}
            onSubtitleChange={(v) => updatePeq({ scale: v })}
          >
            <button
              onClick={() => updatePeq({ mode: inst.mode === 'boost' ? 'cut' : 'boost' })}
              className="flex flex-col items-center justify-center rounded-lg px-2 py-1 transition-colors"
              style={{
                background: inst.mode === 'boost' ? '#5ec4b830' : '#c45e3e30',
                border: `1px solid ${inst.mode === 'boost' ? '#5ec4b860' : '#c45e3e60'}`,
                minWidth: 42,
              }}
              title={inst.mode === 'boost' ? 'Boosting in-key frequencies' : 'Cutting out-of-key frequencies'}
            >
              <span className="text-lg font-bold" style={{ color: inst.mode === 'boost' ? '#5ec4b8' : '#c45e3e' }}>
                {inst.mode === 'boost' ? '(+)' : '(−)'}
              </span>
              <span className="text-[9px] uppercase tracking-wider" style={{ color: inst.mode === 'boost' ? '#5ec4b8' : '#c45e3e' }}>
                {inst.mode === 'boost' ? 'Boost' : 'Cut'}
              </span>
            </button>
            <Knob
              label="Key"
              value={inst.key}
              onChange={(v) => updatePeq({ key: Math.round(v) })}
              color={meta.color}
              min={0}
              max={11}
              displayValue={peqNoteNames[Math.round(inst.key)]}
            />
            <Knob
              label="Amount"
              value={inst.amount}
              onChange={(v) => updatePeq({ amount: v })}
              color={meta.color}
              displayValue={`${Math.round(inst.amount * 100)}%`}
            />
            <Knob
              label="Q"
              value={inst.q}
              onChange={(v) => updatePeq({ q: v })}
              color={meta.color}
              min={0.5}
              max={12}
              displayValue={inst.q.toFixed(1)}
            />
          </EffectModule>
        );
        }
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
          retuneStatus={retuneStatus}
          retuneEnabled={params.retuneEnabled}
          retuneKey={params.retuneKey}
          retuneScale={params.retuneScale}
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

      {/* Dev Panel (Shift+D to toggle) */}
      {showDevPanel && (
        <div
          className="flex items-center gap-4 px-4 py-2 mx-2 rounded-lg"
          style={{
            background: '#1a1a2e',
            border: '1px solid #ff6b3540',
          }}
        >
          <span className="text-xs font-mono text-yellow-400 uppercase tracking-wider">Dev</span>
          <Knob
            label="Drive Range"
            value={params._devDriveRange}
            onChange={(v) => updateParams({ _devDriveRange: v })}
            color="#ff6b35"
            displayValue={`${(params._devDriveRange * 100).toFixed(0)}%`}
          />
          <span className="text-xs text-gray-500 font-mono">
            Tape max: {(1 + 20 * params._devDriveRange).toFixed(0)}x |
            Tube max: {(1 + 25 * params._devDriveRange).toFixed(0)}x
          </span>
        </div>
      )}

      {/* Effect Modules Row */}
      <div className="flex gap-2 px-2 py-3 overflow-x-auto items-start">
        {orderedFxSlots.map(slot => renderFxSlot(slot))}
        <AddFxButton onAdd={addFxSlot} availableTypes={availableFxTypes} />
        <OutputModule params={params} updateParams={updateParams} />
      </div>
    </div>
  );
}
