import { EffectModule } from './EffectModule';
import { Knob } from './Knob';
import type { EngineParams, RetuneStatus } from '../audio/engine';
import { NOTE_NAMES, RETUNE_SCALES, buildScaleMask, formatMidiNote } from '../audio/retune';

interface RetuneModuleProps {
  color: string;
  params: EngineParams;
  status: RetuneStatus | null;
  updateParams: (patch: Partial<EngineParams>) => void;
  onRemove: () => void;
}

const SCALE_OPTIONS = Object.keys(RETUNE_SCALES) as Array<keyof typeof RETUNE_SCALES>;
const OUT_OF_MASK_MODES: EngineParams['retuneOutOfMaskMode'][] = ['nearest', 'preserve', 'mute'];

function formatScaleLabel(scale: string) {
  return scale.replace('_', ' ');
}

function formatLatency(ms: number | undefined) {
  return Number.isFinite(ms) ? `${Math.round(ms ?? 0)} ms` : '—';
}

export function RetuneModule({
  color,
  params,
  status,
  updateParams,
  onRemove,
}: RetuneModuleProps) {
  const targetMask = params.retuneTargetMask.length === 12
    ? params.retuneTargetMask
    : buildScaleMask(params.retuneKey, params.retuneScale);
  const sortedNotes = [...(status?.notes ?? [])].sort((a, b) => b.energy - a.energy).slice(0, 6);

  const stateLabel = !params.retuneEnabled
    ? 'Off'
    : status?.overloaded
      ? 'Overloaded'
      : status?.noteCount
        ? 'Tracking'
        : 'Listening';

  const handleKeyChange = (key: number) => {
    updateParams({
      retuneKey: key,
      retuneTargetMask: buildScaleMask(key, params.retuneScale),
    });
  };

  const handleScaleChange = (scale: EngineParams['retuneScale']) => {
    updateParams({
      retuneScale: scale,
      retuneTargetMask: buildScaleMask(params.retuneKey, scale),
    });
  };

  const handleMaskToggle = (pitchClass: number) => {
    const nextMask = [...targetMask];
    const enabledCount = nextMask.filter(Boolean).length;
    if (nextMask[pitchClass] && enabledCount === 1) {
      return;
    }
    nextMask[pitchClass] = !nextMask[pitchClass];
    updateParams({ retuneTargetMask: nextMask });
  };

  return (
    <EffectModule
      title="Retune"
      color={color}
      enabled={params.retuneEnabled}
      onToggle={() => updateParams({ retuneEnabled: !params.retuneEnabled })}
      onRemove={onRemove}
      subtitle={formatScaleLabel(params.retuneScale)}
      subtitleOptions={SCALE_OPTIONS}
      onSubtitleChange={(value) => handleScaleChange(value as EngineParams['retuneScale'])}
      minWidth={560}
      controlsClassName="flex flex-col gap-3 px-3 py-3"
    >
      <div className="grid grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)] gap-3">
        <div className="rounded-lg border border-border bg-surface-2/60 p-3">
          <div className="flex items-center justify-between gap-2">
            <div>
              <div className="text-[10px] uppercase tracking-[0.24em] text-text-dim">Status</div>
              <div className="mt-1 text-sm font-semibold text-text-primary">{stateLabel}</div>
            </div>
            <div className="text-right text-[11px] text-text-secondary">
              <div>{status?.backendName ?? 'Waiting for engine'}</div>
              <div>{formatLatency(status?.analysisLatencyMs)}</div>
            </div>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-text-secondary">
            <div className="rounded-md bg-surface-1/80 px-2 py-1.5">
              <div className="text-text-dim">Detected</div>
              <div className="mt-0.5 text-text-primary">{formatMidiNote(status?.notes[0]?.sourceMidi ?? Number.NaN)}</div>
            </div>
            <div className="rounded-md bg-surface-1/80 px-2 py-1.5">
              <div className="text-text-dim">Target</div>
              <div className="mt-0.5 text-text-primary">{formatMidiNote(status?.notes[0]?.targetMidi ?? Number.NaN)}</div>
            </div>
            <div className="rounded-md bg-surface-1/80 px-2 py-1.5">
              <div className="text-text-dim">Confidence</div>
              <div className="mt-0.5 text-text-primary">{status ? `${Math.round(status.confidence * 100)}%` : '—'}</div>
            </div>
            <div className="rounded-md bg-surface-1/80 px-2 py-1.5">
              <div className="text-text-dim">Low End</div>
              <div className="mt-0.5 text-text-primary">
                {status?.lowEndLocked ? 'Locked' : 'Loose'}
                {status?.transientBypassActive ? ' · Attack Hold' : ''}
              </div>
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {sortedNotes.length > 0 ? (
              sortedNotes.map((note, index) => (
                <div
                  key={`${note.sourceMidi}-${note.targetMidi}-${index}`}
                  className="rounded-md border border-border bg-surface-1/90 px-2 py-1.5 text-[11px] text-text-secondary"
                >
                  <div className="flex items-center gap-1.5 text-text-primary">
                    <span>{formatMidiNote(note.sourceMidi)}</span>
                    <span className="text-text-dim">→</span>
                    <span>{formatMidiNote(note.targetMidi)}</span>
                  </div>
                  <div className="mt-0.5 flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-text-dim">
                    <span>{note.state}</span>
                    {note.isLowEnd && <span className="text-[#4ec48a]">Low</span>}
                    <span>{Math.round(note.confidence * 100)}%</span>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-[11px] text-text-dim">No tracked note groups yet.</div>
            )}
          </div>
        </div>

        <div className="rounded-lg border border-border bg-surface-2/60 p-3">
          <div className="flex items-center justify-between gap-2">
            <div className="text-[10px] uppercase tracking-[0.24em] text-text-dim">Target Mask</div>
            <button
              onClick={() => updateParams({ retuneTargetMask: buildScaleMask(params.retuneKey, params.retuneScale) })}
              className="rounded border border-border px-2 py-1 text-[10px] uppercase tracking-wide text-text-secondary transition-colors hover:border-text-dim hover:text-text-primary"
            >
              Reset
            </button>
          </div>
          <div className="mt-3 grid grid-cols-4 gap-2">
            {NOTE_NAMES.map((noteName, pitchClass) => {
              const active = Boolean(targetMask[pitchClass]);
              const isKeyRoot = pitchClass === params.retuneKey;
              return (
                <button
                  key={noteName}
                  onClick={() => handleMaskToggle(pitchClass)}
                  className="rounded-md border px-2 py-2 text-xs font-medium transition-colors"
                  style={{
                    background: active ? `${color}22` : '#14142b',
                    borderColor: active ? `${color}66` : '#3a3a5c',
                    color: active ? '#e8e8f0' : '#8888a8',
                    boxShadow: isKeyRoot ? `inset 0 0 0 1px ${color}` : 'none',
                  }}
                >
                  {noteName}
                </button>
              );
            })}
          </div>
          <div className="mt-3">
            <div className="text-[10px] uppercase tracking-[0.24em] text-text-dim">Out of Mask</div>
            <div className="mt-2 flex gap-2">
              {OUT_OF_MASK_MODES.map((mode) => {
                const active = params.retuneOutOfMaskMode === mode;
                return (
                  <button
                    key={mode}
                    onClick={() => updateParams({ retuneOutOfMaskMode: mode })}
                    className="rounded-md border px-2.5 py-1.5 text-[11px] uppercase tracking-wide transition-colors"
                    style={{
                      background: active ? `${color}22` : '#14142b',
                      borderColor: active ? `${color}66` : '#3a3a5c',
                      color: active ? '#e8e8f0' : '#8888a8',
                    }}
                  >
                    {mode}
                  </button>
                );
              })}
            </div>
          </div>
          <div className="mt-3 flex gap-2">
            <button
              onClick={() => updateParams({ retunePreserveTransients: !params.retunePreserveTransients })}
              className="rounded-md border px-2.5 py-1.5 text-[11px] uppercase tracking-wide transition-colors"
              style={{
                background: params.retunePreserveTransients ? `${color}22` : '#14142b',
                borderColor: params.retunePreserveTransients ? `${color}66` : '#3a3a5c',
                color: params.retunePreserveTransients ? '#e8e8f0' : '#8888a8',
              }}
            >
              Preserve Attacks
            </button>
            <button
              onClick={() => updateParams({ retuneCollapseDuplicates: !params.retuneCollapseDuplicates })}
              className="rounded-md border px-2.5 py-1.5 text-[11px] uppercase tracking-wide transition-colors"
              style={{
                background: params.retuneCollapseDuplicates ? `${color}22` : '#14142b',
                borderColor: params.retuneCollapseDuplicates ? `${color}66` : '#3a3a5c',
                color: params.retuneCollapseDuplicates ? '#e8e8f0' : '#8888a8',
              }}
            >
              Collapse Duplicates
            </button>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-end justify-center gap-4">
        <Knob
          label="Strength"
          value={params.retuneStrength}
          onChange={(value) => updateParams({ retuneStrength: value })}
          color={color}
          displayValue={`${Math.round(params.retuneStrength * 100)}%`}
        />
        <Knob
          label="Key"
          value={params.retuneKey}
          onChange={(value) => handleKeyChange(Math.round(value))}
          color={color}
          min={0}
          max={11}
          displayValue={NOTE_NAMES[Math.round(params.retuneKey)]}
        />
        <Knob
          label="Texture"
          value={params.retuneTextureAmount}
          onChange={(value) => updateParams({ retuneTextureAmount: value })}
          color={color}
          displayValue={`${Math.round(params.retuneTextureAmount * 100)}%`}
        />
        <Knob
          label="Low Blend"
          value={params.retuneLowEndBlend}
          onChange={(value) => updateParams({ retuneLowEndBlend: value })}
          color={color}
          displayValue={`${Math.round(params.retuneLowEndBlend * 100)}%`}
        />
        <Knob
          label="Sub"
          value={params.retuneSubReinforcement}
          onChange={(value) => updateParams({ retuneSubReinforcement: value })}
          color={color}
          displayValue={`${Math.round(params.retuneSubReinforcement * 100)}%`}
        />
        <Knob
          label="Air"
          value={params.retuneAirMix}
          onChange={(value) => updateParams({ retuneAirMix: value })}
          color={color}
          min={0}
          max={1.5}
          displayValue={`${Math.round(params.retuneAirMix * 100)}%`}
        />
      </div>
    </EffectModule>
  );
}
