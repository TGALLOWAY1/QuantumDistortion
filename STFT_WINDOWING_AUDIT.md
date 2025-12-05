# STFT/ISTFT and Windowing Diagnostic Report

## Executive Summary

This audit examines all STFT/ISTFT and windowing implementations in the QuantumDistortion codebase to assess correctness of overlap-add (OLA) reconstruction and identify any violations of the `hop_length = n_fft // 4` constraint.

---

## 1. STFT Implementation Location

**Primary Implementation**: `quantum_distortion/dsp/stft_utils.py`

- **`stft_mono()`** (lines 11-40): Wrapper around `librosa.stft()`
- **`istft_mono()`** (lines 43-67): Wrapper around `librosa.istft()`

**Secondary Usage**:
- `quantum_distortion/dsp/pipeline.py`: Uses `stft_mono()` and `istft_mono()` throughout the processing pipeline
- `quantum_distortion/dsp/analyses.py`: Direct `librosa.stft()` call in `avg_cents_offset_from_scale()` (line 98)

**Note**: The codebase does NOT implement custom STFT/ISTFT. All operations delegate to librosa.

---

## 2. Hop Length Calculation

**Default Calculation**: `hop_length = n_fft // 4`

**Locations**:
- `stft_utils.py` line 30: `hop_length = n_fft // 4` (when `None`)
- `stft_utils.py` line 58: `hop_length = n_fft // 4` (when `None`)
- `pipeline.py` line 42: `HOP_LENGTH_DEFAULT = N_FFT_DEFAULT // 4  # 512`
- `analyses.py` line 95: `hop_length = frame_length // 4` (when `None`)

**Current Values**:
- `N_FFT_DEFAULT = 2048`
- `HOP_LENGTH_DEFAULT = 512` (which equals `2048 // 4`)

**Consistency**: ✅ All STFT/ISTFT calls use `HOP_LENGTH_DEFAULT` or calculate `n_fft // 4` when hop_length is `None`.

---

## 3. Analysis and Synthesis Window Application

**Window Type**: `"hann"` (Hann/Hanning window)

**Window Specification**:
- `pipeline.py` line 43: `WINDOW_DEFAULT = "hann"`
- All STFT calls pass `window=WINDOW_DEFAULT` or `window="hann"`
- All ISTFT calls pass `window=WINDOW_DEFAULT` or `window="hann"`

**Window Application Method**:
- **No custom windowing code exists** in the codebase
- Window application is handled entirely by `librosa.stft()` and `librosa.istft()`
- The window parameter is passed as a string identifier to librosa, which constructs the window internally

**Critical Finding**: The codebase does NOT explicitly manage analysis vs. synthesis windows. It relies on librosa's default behavior, which uses the **same window** for both analysis (STFT) and synthesis (ISTFT).

---

## 4. Overlap-Add (OLA) Mathematical Correctness

### Current Implementation

**OLA is handled entirely by librosa** - there is no custom overlap-add code in the codebase.

### Theoretical Requirements for Perfect Reconstruction

For perfect reconstruction with OLA, the following condition must be satisfied:

```
∑[w(n - m*hop_length)²] = constant  (for all n)
```

Where:
- `w(n)` is the analysis/synthesis window
- `hop_length` is the hop size
- The sum is over all frames `m` that overlap at sample `n`

### Hann Window + hop_length = n_fft // 4

For a **Hann window** with **hop_length = n_fft // 4** (75% overlap):
- The COLA (Constant Overlap-Add) condition is **approximately satisfied** but not perfect
- Perfect reconstruction requires either:
  1. **Different analysis and synthesis windows** (e.g., analysis Hann, synthesis scaled to satisfy COLA)
  2. **Modified hop_length** to exactly satisfy COLA for the given window
  3. **Window normalization** during OLA reconstruction

### Librosa's Behavior

`librosa.istft()` uses the **same window** for synthesis as was used for analysis. For a Hann window with 75% overlap, librosa applies a normalization factor internally to approximate perfect reconstruction, but this is not mathematically exact.

**Assessment**: ⚠️ **The current implementation relies on librosa's approximation** rather than mathematically exact OLA. This may introduce:
- Small reconstruction errors
- Potential artifacts in processed audio
- Phase inconsistencies

---

## 5. Library Usage: librosa vs. scipy vs. Custom

**Library Used**: **librosa** (not scipy, not custom)

**Evidence**:
- `stft_utils.py` line 8: `import librosa`
- `stft_utils.py` line 32: `S = librosa.stft(...)`
- `stft_utils.py` line 60: `y = librosa.istft(...)`
- `analyses.py` line 13: `import librosa`
- `analyses.py` line 98: `S = librosa.stft(...)`

**No scipy.signal.stft or scipy.signal.istft usage found.**

**No custom STFT/ISTFT implementation found.**

---

## 6. Violations of hop = n_fft // 4 or OLA Consistency

### Hop Length Violations

✅ **No violations found**. All STFT/ISTFT operations use:
- `hop_length = n_fft // 4` when `hop_length` is `None`
- `HOP_LENGTH_DEFAULT = 512` (which equals `2048 // 4`) when explicitly specified

### OLA Consistency Issues

⚠️ **Potential issues identified**:

1. **Same Window for Analysis and Synthesis**
   - Current: Both STFT and ISTFT use the same "hann" window
   - Issue: For perfect OLA reconstruction with 75% overlap, the synthesis window should be normalized or different from the analysis window
   - Impact: May cause small reconstruction errors

2. **No Explicit Window Normalization**
   - The codebase does not verify or enforce COLA conditions
   - Relies entirely on librosa's internal handling
   - No validation that windows satisfy the overlap-add constraint

3. **Inconsistent Window Usage in Analyses**
   - `analyses.py` uses `window="hann"` directly (line 102)
   - `pipeline.py` uses `WINDOW_DEFAULT = "hann"`
   - While both use the same window type, the lack of a shared constant could lead to future inconsistencies

4. **No Verification of Perfect Reconstruction**
   - No tests found that verify `istft(stft(x)) ≈ x` (identity test)
   - No validation of OLA correctness

---

## Summary of Findings

### ✅ Strengths
1. Consistent hop_length calculation (`n_fft // 4`)
2. All STFT/ISTFT calls use the same parameters
3. Centralized configuration via `pipeline.py` constants
4. No violations of the `hop = n_fft // 4` rule

### ⚠️ Concerns
1. **No explicit OLA correctness guarantee**: Relies on librosa's approximation
2. **Same window for analysis and synthesis**: May not satisfy exact COLA condition
3. **No identity tests**: No verification that STFT→ISTFT roundtrip preserves signal
4. **No window normalization**: No explicit handling of window scaling for perfect reconstruction

### 🔍 Recommendations for Further Investigation
1. Add identity test: `istft(stft(x))` should reconstruct `x` with minimal error
2. Verify librosa's OLA implementation details for Hann window + 75% overlap
3. Consider explicit window normalization or different synthesis window if perfect reconstruction is required
4. Check for audible artifacts in processed audio that might indicate OLA issues

---

## Files Examined

- `quantum_distortion/dsp/stft_utils.py` (67 lines)
- `quantum_distortion/dsp/pipeline.py` (742 lines, multiple STFT/ISTFT calls)
- `quantum_distortion/dsp/analyses.py` (152 lines, one STFT call)
- `quantum_distortion/ui/visualizers.py` (uses `np.hanning()` for visualization only, not STFT)

---

**Report Generated**: Based on codebase analysis
**Date**: Current codebase state on branch `M10---overlap-add-and-quality-refinement`

