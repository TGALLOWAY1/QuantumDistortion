import { Knob } from './Knob';
import type { EngineParams } from '../audio/engine';
import { NOTE_NAMES } from '../audio/retune';

interface RetuneModuleProps {
  color: string;
  params: EngineParams;
  updateParams: (patch: Partial<EngineParams>) => void;
}

const SCALE_OPTIONS: EngineParams['retuneScale'][] = [
  'major',
  'minor',
  'dorian',
  'phrygian',
  'lydian',
  'mixolydian',
  'aeolian',
  'harmonic_minor',
  'custom',
];

const QUANTIZE_MODES = ['snap', 'glide', 'preserve attacks'] as const;

export function RetuneModule({ color, params, updateParams }: RetuneModuleProps) {
  const quantizeMode = params.retunePreserveTransients ? 'preserve attacks' : (params.retuneOutOfMaskMode === 'preserve' ? 'glide' : 'snap');

  return (
    <div
      className="flex flex-col rounded-xl overflow-hidden flex-shrink-0"
      style={{
        background: 'linear-gradient(180deg, #4891ff16 0%, #14142b 100%)',
        border: '1px solid #4891ff55',
        minWidth: 250,
      }}
    >
      <div className="flex items-center gap-2 px-3 py-2 border-b border-[#4891ff33]">
        <span className="text-[10px] text-[#9dbfff]">2</span>
        <span className="text-xs font-semibold tracking-wider uppercase text-text-primary">Retune</span>
        <div className="flex-1" />
        <button
          onClick={() => updateParams({ retuneEnabled: !params.retuneEnabled })}
          className="w-4 h-4 rounded-full border-2"
          style={{ borderColor: color, background: params.retuneEnabled ? color : 'transparent' }}
          aria-label="Toggle retune"
        />
      </div>

      <div className="px-3 py-3 flex flex-col gap-2">
        <div className="grid grid-cols-2 gap-2">
          <select
            value={params.retuneScale}
            onChange={(e) => updateParams({ retuneScale: e.target.value as EngineParams['retuneScale'] })}
            className="text-xs rounded bg-surface-2 border border-border px-2 py-1.5 text-text-primary"
          >
            {SCALE_OPTIONS.map((scale) => (
              <option key={scale} value={scale}>{scale.replace('_', ' ')}</option>
            ))}
          </select>

          <select
            value={params.retuneKey}
            onChange={(e) => updateParams({ retuneKey: Number(e.target.value) })}
            className="text-xs rounded bg-surface-2 border border-border px-2 py-1.5 text-text-primary"
          >
            {NOTE_NAMES.map((note, index) => (
              <option key={note} value={index}>{note}</option>
            ))}
          </select>
        </div>

        <select
          value={quantizeMode}
          onChange={(e) => {
            const mode = e.target.value as (typeof QUANTIZE_MODES)[number];
            if (mode === 'preserve attacks') {
              updateParams({ retunePreserveTransients: true, retuneOutOfMaskMode: 'nearest' });
            } else if (mode === 'glide') {
              updateParams({ retunePreserveTransients: false, retuneOutOfMaskMode: 'preserve' });
            } else {
              updateParams({ retunePreserveTransients: false, retuneOutOfMaskMode: 'nearest' });
            }
          }}
          className="text-xs rounded bg-surface-2 border border-border px-2 py-1.5 text-text-primary"
        >
          {QUANTIZE_MODES.map((mode) => (
            <option key={mode} value={mode}>{mode}</option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-2 gap-3 px-3 pb-3">
        <Knob
          label="Speed"
          value={params.retuneStrength}
          onChange={(value) => updateParams({ retuneStrength: value })}
          color={color}
          displayValue={`${Math.round(params.retuneStrength * 100)}%`}
        />
        <Knob
          label="Strength"
          value={params.retuneTextureAmount}
          onChange={(value) => updateParams({ retuneTextureAmount: value })}
          color={color}
          displayValue={`${Math.round(params.retuneTextureAmount * 100)}%`}
        />
        <Knob
          label="Octave"
          value={params.retuneAirMix}
          onChange={(value) => updateParams({ retuneAirMix: value })}
          color={color}
          min={0}
          max={1.5}
          displayValue={params.retuneAirMix.toFixed(2)}
        />
        <Knob
          label="Level"
          value={params.retuneLowEndBlend}
          onChange={(value) => updateParams({ retuneLowEndBlend: value })}
          color={color}
          displayValue={`${Math.round(params.retuneLowEndBlend * 100)}%`}
        />
      </div>
    </div>
  );
}
