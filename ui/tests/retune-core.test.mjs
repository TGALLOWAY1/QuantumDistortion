import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildScaleMask,
  normalizeRetuneParamPatch,
  PolyphonicRetuneCore,
} from '../public/worklets/retune-core.js';

const SAMPLE_RATE = 48_000;
const BLOCK_SIZE = 128;

function createCore(params = {}) {
  const core = new PolyphonicRetuneCore({ sampleRate: SAMPLE_RATE, maxChannels: 2 });
  core.setParams({ retuneEnabled: true, ...params });
  return core;
}

function renderSignal({ seconds = 1.4, params = {}, sampleAt }) {
  const core = createCore(params);
  const totalSamples = Math.ceil(seconds * SAMPLE_RATE);
  const totalBlocks = Math.ceil(totalSamples / BLOCK_SIZE);
  const output = [];
  const statuses = [];
  let inputEnergy = 0;

  for (let blockIndex = 0; blockIndex < totalBlocks; blockIndex++) {
    const left = new Float32Array(BLOCK_SIZE);
    const right = new Float32Array(BLOCK_SIZE);

    for (let i = 0; i < BLOCK_SIZE; i++) {
      const sampleIndex = blockIndex * BLOCK_SIZE + i;
      const time = sampleIndex / SAMPLE_RATE;
      const sample = sampleIndex < totalSamples ? sampleAt(sampleIndex, time) : 0;
      left[i] = sample;
      right[i] = sample;
      inputEnergy += sample * sample;
    }

    const processed = core.processBlock([left, right], 2, BLOCK_SIZE, { lowGain: 1, highGain: 1 });
    for (let i = 0; i < BLOCK_SIZE; i++) {
      output.push(processed[0][i]);
    }
    statuses.push(core.getStatus());
  }

  const outputEnergy = output.reduce((sum, sample) => sum + sample * sample, 0);
  return {
    core,
    status: core.getStatus(),
    statuses,
    inputRms: Math.sqrt(inputEnergy / (totalBlocks * BLOCK_SIZE)),
    outputRms: Math.sqrt(outputEnergy / output.length),
  };
}

function findNote(notes, predicate) {
  return notes.find(predicate);
}

test('normalizes legacy quantize params into retune params', () => {
  const normalized = normalizeRetuneParamPatch({
    quantizeEnabled: true,
    quantizeKey: 2,
    quantizeScale: 'minor',
    quantizeStrength: 0.82,
    quantizeSubLevel: 0.41,
  });

  assert.equal(normalized.retuneEnabled, true);
  assert.equal(normalized.retuneKey, 2);
  assert.equal(normalized.retuneScale, 'minor');
  assert.equal(normalized.retuneStrength, 0.82);
  assert.equal(normalized.retuneSubReinforcement, 0.41);
  assert.deepEqual(normalized.retuneTargetMask, buildScaleMask(2, 'minor'));
});

test('tracks low-end notes and remaps them to the nearest allowed pitch class', () => {
  const result = renderSignal({
    params: {
      retuneKey: 0,
      retuneScale: 'major',
      retuneLowEndBlend: 0.75,
      retuneSubReinforcement: 0,
    },
    sampleAt(sampleIndex) {
      return (
        Math.sin((2 * Math.PI * 46.2493028389543 * sampleIndex) / SAMPLE_RATE) * 0.7
        + Math.sin((2 * Math.PI * 92.4986056779086 * sampleIndex) / SAMPLE_RATE) * 0.2
      );
    },
  });

  const trackedLow = findNote(
    result.status.notes,
    (note) => note.isLowEnd && Math.abs(note.sourceMidi - 30) < 0.75,
  );

  assert.ok(trackedLow, 'expected a tracked low-end note near F#1');
  assert.equal(trackedLow.state, 'mapped');
  assert.ok(Math.abs(trackedLow.targetMidi - 31) < 0.75, `expected low-end target near G1, got ${trackedLow.targetMidi}`);
  assert.equal(result.status.lowEndLocked, true);
  assert.equal(result.status.analysisLatencyMs, 64);
});

test('keeps multiple tracked note objects for dyads', () => {
  const result = renderSignal({
    params: { retuneKey: 0, retuneScale: 'major' },
    sampleAt(sampleIndex) {
      return (
        Math.sin((2 * Math.PI * 440 * sampleIndex) / SAMPLE_RATE) * 0.35
        + Math.sin((2 * Math.PI * 523.2511306011972 * sampleIndex) / SAMPLE_RATE) * 0.35
      );
    },
  });

  const a4 = findNote(result.status.notes, (note) => !note.isLowEnd && Math.abs(note.sourceMidi - 69) < 1);
  const c5 = findNote(result.status.notes, (note) => !note.isLowEnd && Math.abs(note.sourceMidi - 72) < 1);

  assert.ok(a4, 'expected a tracked A4 note object');
  assert.ok(c5, 'expected a tracked C5 note object');
  assert.ok(result.status.noteCount >= 2, `expected multiple note groups, got ${result.status.noteCount}`);
});

test('preserve mode leaves out-of-mask notes alone while mute mode suppresses them', () => {
  const cOnlyMask = new Array(12).fill(false);
  cOnlyMask[0] = true;

  const preserveResult = renderSignal({
    params: {
      retuneTargetMask: cOnlyMask,
      retuneOutOfMaskMode: 'preserve',
    },
    sampleAt(sampleIndex) {
      return Math.sin((2 * Math.PI * 369.9944227116344 * sampleIndex) / SAMPLE_RATE) * 0.8;
    },
  });

  const preserved = findNote(
    preserveResult.status.notes,
    (note) => !note.isLowEnd && Math.abs(note.sourceMidi - 66) < 1,
  );
  assert.ok(preserved, 'expected a tracked F#4 note');
  assert.equal(preserved.state, 'preserved');
  assert.ok(Math.abs(preserved.targetMidi - preserved.sourceMidi) < 0.75);

  const muteResult = renderSignal({
    params: {
      retuneTargetMask: cOnlyMask,
      retuneOutOfMaskMode: 'mute',
    },
    sampleAt(sampleIndex) {
      return Math.sin((2 * Math.PI * 369.9944227116344 * sampleIndex) / SAMPLE_RATE) * 0.8;
    },
  });

  const muted = findNote(
    muteResult.status.notes,
    (note) => !note.isLowEnd && Math.abs(note.sourceMidi - 66) < 1,
  );
  assert.ok(muted, 'expected a muted F#4 note object');
  assert.equal(muted.state, 'muted');
  assert.ok(muteResult.outputRms < muteResult.inputRms * 0.2, `expected muted output to drop strongly, got ${muteResult.outputRms}`);
});

test('transient preservation reports attack bypass on note onsets', () => {
  const result = renderSignal({
    params: { retunePreserveTransients: true },
    sampleAt(sampleIndex, time) {
      const env = time < 0.2 ? 0 : time < 0.22 ? (time - 0.2) / 0.02 : 1;
      return env * (
        Math.sin((2 * Math.PI * 440 * sampleIndex) / SAMPLE_RATE) * 0.45
        + Math.sin((2 * Math.PI * 554.3652619537442 * sampleIndex) / SAMPLE_RATE) * 0.3
      );
    },
  });

  assert.ok(
    result.statuses.some((status) => status.transientBypassActive),
    'expected transient bypass to activate during the attack window',
  );
});

test('noise stays low-confidence and near-unity ratio', () => {
  let seed = 1;
  const result = renderSignal({
    params: { retuneKey: 0, retuneScale: 'major' },
    sampleAt() {
      seed = (seed * 1_664_525 + 1_013_904_223) >>> 0;
      return ((seed / 4_294_967_295) * 2 - 1) * 0.35;
    },
  });

  assert.ok(result.status.confidence < 0.2, `expected low confidence on noise, got ${result.status.confidence}`);
  assert.ok(Math.abs(result.status.smoothedRatio - 1) < 0.05, `expected near-unity ratio on noise, got ${result.status.smoothedRatio}`);
});
