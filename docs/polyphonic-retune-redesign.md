# Electron-First Polyphonic Retune Redesign

## Goal

Make the Electron app the persistent product UI and replace the current split Python/JS pitch logic with one unified interactive backend that can grow into Pitchmap-style polyphonic note manipulation.

This document is intentionally opinionated:

- `spectral_bins` stays only as a creative spectral effect, not the correction core.
- the current worklet retuner is treated as a realtime prototype, not the long-term engine.
- the future backend is driven by the Electron UI contract first.

## Product Scope

### In Scope

- interactive Electron-first workflow
- polyphonic note-aware remapping
- explicit note-target control
- low-end handling that does not rely on a fixed sub oscillator as the source of truth
- live engine telemetry in the UI

### Out of Scope

- preserving the current CLI as a product surface
- presenting the current monophonic worklet as the final DSP design
- treating FFT-bin attraction as autotune

## Unified Architecture

Target runtime layout:

1. Electron/React UI
2. unified backend host
3. shared retune engine
4. realtime status channel back to the UI

The UI should never infer backend state from knob positions alone. The backend must report:

- detected note groups
- target note groups
- confidence / lock state
- low-end assignment strategy
- engine mode
- latency / CPU health

## Backend Stages

### Stage 1: Analysis Front End

Use a polyphonic analysis front end instead of YIN:

- constant-Q or log-frequency analysis for note-space stability
- transient detection to protect attacks from smearing
- harmonic/percussive separation or residual split
- multi-pitch candidate extraction per frame
- continuity tracking across frames so notes are stable objects, not isolated peaks

Outputs:

- note candidates with pitch, strength, bandwidth, and harmonic support
- residual/unassigned energy
- low-frequency note candidates

### Stage 2: Note Assignment / Mapping

Each tracked source note should be assignable to:

- nearest in-key note
- explicit user note map
- chord or interval map
- mute / preserve / collapse behaviors

This is where Pitchmap-like behavior starts. The engine needs note objects, not one global pitch ratio.

### Stage 3: Resynthesis

Resynthesis should be note-group aware:

- phase-consistent partial remapping for tonal components
- transient bypass or protected transient lane
- residual lane for noisy content
- optional texture coloration after retune, not baked into it

## Low-End Strategy

The current fixed-note synthetic sub is acceptable as an effect, but it cannot be the corrective low-end strategy.

Recommended low-end design:

1. Split low end into three lanes:
   - transient low-frequency energy
   - stable tonal low-frequency note groups
   - residual low-frequency noise
2. Track low notes separately with a long-window log-frequency estimator or harmonic template matcher.
3. Rebuild the tonal low end from assigned note groups:
   - phase-locked sinusoidal bank
   - oscillator-bank reinforcement tied to tracked source notes
   - optional hybrid of shifted original low mids plus resynthesized fundamental
4. Use the old fixed-note sub oscillator only as an optional reinforcement module.

This keeps low-end note identity tied to the mapped note groups instead of a static root tone.

## UI Contract

The Electron UI should drive a backend contract shaped like this:

- `engineMode`
- `mappingMode`
- `lowEndStrategy`
- `preserveTransients`
- `retuneStrength`
- `textureAmount`
- `subReinforcement`
- `noteMap`
- `status`

The UI should receive:

- active source notes
- assigned target notes
- confidence per note
- low-end lock state
- backend label
- latency / overload flags

## Migration Plan

### Phase 0

- keep the current worklet alive as the realtime prototype
- expose live retune telemetry in the Electron UI
- remove or fix misleading UI behaviors so the frontend matches the current singleton backend

### Phase 1

- define the shared backend contract around note groups and status reporting
- move away from separate Python/JS product behaviors
- make the Electron UI the only primary control surface

### Phase 2

- replace the monophonic detector with a polyphonic analysis front end
- add note-group tracking and mapping
- keep the current distortion / delay / modulation stages downstream of retune

### Phase 3

- replace the body-band delay-line pitch shifter with note-aware resynthesis
- redesign low-end handling around tracked note groups
- relegate the synthetic sub to an explicit reinforcement module

## Immediate Code Implications

- the current worklet status channel should be treated as the long-term UI telemetry path
- module visibility/removal in the Electron UI must match actual backend behavior
- the UI should stop implying that the current backend is already a true polyphonic mapper
