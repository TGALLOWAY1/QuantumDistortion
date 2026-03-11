# Milestone M6: Performance Validation & Quantization Bug Fix

## Summary

This milestone focused on diagnosing and fixing a critical bug where spectral quantization was not effectively reducing cents offset from musical scales. The root cause was identified as improper phase handling when moving energy between frequency bins during quantization. The fix ensures that phases are properly combined using weighted averages when multiple bins contribute energy to the same target bin, preserving the phase-frequency relationship during STFT/iSTFT roundtrips.

## Prompts Executed

### PROMPT — Diagnose & Fix "Identical Input/Output Alignment" Bug

#### Step 1 — Make the validation script print the actual DSP settings
- Updated `scripts/validate_dsp_metrics.py` to print all DSP parameters before processing
- **Debug output block**:
  - Displays key, scale, snap_strength, smear, bin_smoothing
  - Shows pre_quant, post_quant flags
  - Lists distortion_mode and distortion_params
  - Displays limiter_on, limiter_ceiling_db, dry_wet
- Ensures parameters in `process_audio()` call match printed values
- Helps verify correct parameters are being used during validation

#### Step 2 — Add a separate "quantizer-only" validation path
- Added quantizer-only diagnostic block to `scripts/validate_dsp_metrics.py`
- **Diagnostic test**:
  - Processes audio with maximum quantization (snap_strength=1.0, smear=0.0)
  - No distortion or limiter applied
  - Measures cents offset of quantizer-only output
- Provides three metrics per run:
  - Input avg abs cents offset
  - Output avg abs cents offset (full pipeline)
  - Quantizer-only output avg abs cents offset
- Helps isolate quantization effects from distortion/limiter effects

#### Step 3 — Add a focused integration test for quantization integration
- Created `tests/test_quantization_integration.py`
- **Test function**: `test_apply_spectral_quantization_block_reduces_cents_offset()`
- **Test setup**:
  - Generates detuned sine wave (A4 + 30 cents)
  - Measures baseline cents offset
  - Applies spectral quantization with strong settings
  - Verifies output cents offset is reduced compared to input
- **Test validation**:
  - Ensures quantization reduces cents offset (not just preserves it)
  - Accounts for STFT resolution limits in assertions
  - Provides automatic verification that quantization is working

#### Step 4 — Inspect and fix `_apply_spectral_quantization_block`
- **Root cause identified**: Phase handling bug in `quantize_spectrum()`
  - Original implementation: Phases remained unchanged when moving energy
  - Problem: When energy moved from source bin to target bin, phase at target bin was not updated
  - Result: iSTFT reconstruction didn't preserve quantization changes
- **Fix implemented** in `quantum_distortion/dsp/quantizer.py`:
  - Added phase tracking for energy contributions to target bins
  - Uses weighted average of contributing phases when multiple bins target same bin
  - Properly combines phases using complex number arithmetic: `energy * exp(1j * phase)`
  - Updates phases at target bins based on weighted contributions
  - Preserves phase-frequency relationship during quantization
- **Code changes**:
  - Added `target_energy` array to track energy contributions
  - Added `target_phase_sum` complex array for phase-weighted contributions
  - Modified energy movement loop to accumulate phase contributions
  - Added phase update loop after energy movement
  - Handles both direct target and smeared energy contributions

#### Step 5 — Run tests + validation script
- **Test results**: All 21 tests pass, including new integration test
- **Validation script output**:
  - Debug parameters correctly displayed
  - Quantizer-only diagnostic provides additional metrics
  - Quantization now works correctly for simple tones
- **Verification**:
  - Integration test confirms quantization reduces cents offset
  - Signal changes verified (not identical to input)
  - Phase fix ensures STFT/iSTFT roundtrip preserves quantization

## Files Created/Modified

### New Files
- `tests/test_quantization_integration.py` - Integration test for quantization effectiveness
- `pyrightconfig.json` - Pyright configuration for linter (created during bug fix session)

### Modified Files
- `scripts/validate_dsp_metrics.py` - Added debug output and quantizer-only diagnostic
- `quantum_distortion/dsp/quantizer.py` - Fixed phase handling in `quantize_spectrum()`
- `quantum_distortion/io/audio_io.py` - Added type ignore comments for linter
- `README.md` - Updated with validation and profiling sections (from earlier work)

## Key Bug Fix Details

### The Problem
- Quantization appeared to work at the frame level (energy moved correctly)
- After iSTFT reconstruction, dominant frequencies returned to original bins
- Cents offset analysis showed identical input/output values
- Signal was changing, but frequency content wasn't being quantized effectively

### Root Cause
- **Phase mismatch**: When moving energy from source bin to target bin:
  - Magnitude was correctly moved
  - Phase at target bin was not updated to match moved energy
  - iSTFT reconstruction used incorrect phase information
  - Result: Quantization changes were lost during reconstruction

### The Solution
- **Weighted phase combination**: When energy moves to target bins:
  - Track all energy contributions to each target bin
  - Accumulate phase-weighted contributions: `energy * exp(1j * phase)`
  - Update target bin phase using: `angle(sum_of_contributions)`
  - Preserves phase-frequency relationship
  - Handles multiple sources contributing to same target

### Technical Implementation
```python
# Track contributions
target_energy = np.zeros(len(mags), dtype=float)
target_phase_sum = np.zeros(len(mags), dtype=complex)

# Accumulate during energy movement
target_phase_sum[target_bin] += base_energy * np.exp(1j * phases[i])

# Update phases after all energy movement
for i in range(len(mags)):
    if target_energy[i] > 0.0:
        new_phases[i] = np.angle(target_phase_sum[i])
```

## Testing & Validation

### Integration Test
- **Test**: `test_apply_spectral_quantization_block_reduces_cents_offset()`
- **Setup**: Detuned sine wave (A4 + 30 cents)
- **Verification**: Output cents offset < input cents offset
- **Result**: ✅ Passes - quantization reduces cents offset

### Validation Script
- **Debug output**: All parameters correctly displayed
- **Quantizer-only diagnostic**: Provides isolated quantization metrics
- **Full pipeline**: Shows combined effects of quantization + distortion + limiter

### Test Suite
- **All tests pass**: 21/21 tests passing
- **No regressions**: Existing functionality preserved
- **New test added**: Integration test validates quantization effectiveness

## Performance Impact

### Before Fix
- Quantization had no measurable effect on cents offset
- Input/output alignment identical (bug)
- Signal changed but frequency content unchanged

### After Fix
- Quantization reduces cents offset for simple tones
- Phase handling preserves quantization through STFT/iSTFT
- Signal changes reflect frequency quantization correctly

### Limitations
- **Complex audio**: For multi-frequency content, quantization effect may be subtle
- **STFT resolution**: Limited by FFT bin resolution (frequency precision)
- **Analysis sensitivity**: Cents offset measurement may not detect all quantization effects on complex signals

## Technical Details

### Phase Handling Algorithm
1. **Track contributions**: For each source bin moving energy:
   - Calculate energy to move: `energy_to_move = mag * snap_strength`
   - Split into direct target and smeared components
   - Accumulate energy and phase-weighted contributions

2. **Combine phases**: After all energy movement:
   - For each target bin with contributions:
     - Extract phase from accumulated complex sum: `angle(sum_of_contributions)`
     - Update phase at target bin

3. **Preserve relationships**: 
   - Phase-frequency relationship maintained
   - Energy conservation preserved
   - Multiple sources properly combined

### Code Quality
- Proper handling of edge cases (empty contributions, invalid bins)
- Energy conservation verified
- Phase updates only when contributions exist
- Handles both direct and smeared energy contributions

## Verification

- ✅ Integration test passes
- ✅ All 21 tests pass
- ✅ Quantization reduces cents offset for simple tones
- ✅ Phase handling preserves quantization through roundtrip
- ✅ Signal changes verified (not identical to input)
- ✅ Debug output shows correct parameters
- ✅ Quantizer-only diagnostic provides useful metrics
- ✅ No regressions in existing functionality

## Usage

### Running Validation Script
```bash
PYTHONPATH=. python scripts/validate_dsp_metrics.py \
  --infile examples/example_bass.wav \
  --key C \
  --scale minor
```

### Running Integration Test
```bash
pytest tests/test_quantization_integration.py -v
```

### Debug Output
The validation script now shows:
- All DSP parameters being used
- Input cents offset
- Output cents offset (full pipeline)
- Quantizer-only cents offset (isolated quantization effect)

## Next Steps

The quantization bug is fixed and verified. Future improvements could include:
1. Enhanced quantization algorithms for complex audio
2. Higher resolution STFT for finer frequency control
3. Adaptive quantization based on signal characteristics
4. Improved analysis metrics for complex signals
5. Real-time quantization visualization
6. Performance optimization for longer audio files

