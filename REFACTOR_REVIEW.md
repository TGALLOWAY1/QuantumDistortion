# Quantum Distortion — Codebase Refactor Review

**Date:** 2026-03-11
**Reviewer:** Claude Opus 4.6 (Senior Software Engineer perspective)
**Branch:** `master` (68/68 tests passing)
**Codebase:** ~4,500 lines of Python across 40+ files

---

## 1. Architecture Assessment

### Current Architecture

Quantum Distortion is a Python prototype for a spectral pitch quantization + distortion audio DSP engine. The architecture follows a **monolithic pipeline pattern** with supporting modules:

```
quantum_distortion/          # Main package
├── dsp/                     # DSP processing core
│   ├── pipeline.py          # Main orchestrator (~1,035 lines) — THE central module
│   ├── quantizer.py         # Spectral quantization (scale-aware bin attraction)
│   ├── spectral_fx.py       # Frequency-domain creative FX (bitcrush, phase dispersal, bin scramble)
│   ├── distortion.py        # Time-domain wavefold + tube distortion
│   ├── saturation.py        # Low-band saturation + mono-maker
│   ├── limiter.py           # Peak limiter with lookahead
│   ├── crossover.py         # Linkwitz-Riley multiband split
│   ├── stft_utils.py        # Custom OLA-compliant STFT/ISTFT
│   ├── analyses.py          # Pitch analysis (uses librosa)
│   └── harness.py           # File-to-file processing convenience wrapper
├── io/
│   └── audio_io.py          # Load/save audio via soundfile
├── ui/
│   ├── app_streamlit.py     # Streamlit UI (~900 lines, V1+V2 combined)
│   └── visualizers.py       # Matplotlib-based spectrum/oscilloscope/phase plots
├── config.py                # Global default constants
└── presets.py               # JSON preset loader with caching
```

### Major Architectural Issues

**1. `pipeline.py` is a god module (1,035 lines)**

This single file contains:
- A private config class (`_SpectralFXConfig`)
- A spectral FX dispatcher (`apply_spectral_fx`)
- Per-frame STFT processing logic (`_apply_spectral_quantization_to_stft`)
- Branch alignment logic (`_align_branches`)
- Passthrough test logic
- Timing/profiling infrastructure
- The main 190+ line `process_audio()` function with extensive conditional branching
- A dataclass-like `_RenderTiming` class

This violates single responsibility and makes the pipeline difficult to understand, test, and extend. New features (M12: spectral freeze, formant shift, harmonic locking) will make this worse.

**2. Duplicated `soft_tube` implementations**

There are **two separate** `soft_tube` functions:
- `quantum_distortion/dsp/distortion.py:66-95` — takes `drive` and `warmth`, uses shape parameter `a = 1 + 4*warmth`
- `quantum_distortion/dsp/saturation.py:6-61` — takes only `drive`, uses fixed shape `a = 3.0`

These serve different purposes (distortion vs. low-band saturation) but share the same name and nearly identical logic. This is confusing and error-prone.

**3. `process_audio()` has 20+ parameters**

The main entry point accepts 20+ keyword arguments passed individually. This creates:
- Massive function signatures
- Copy-paste parameter threading in `harness.py`, `render_preset.py`, `render_cli.py`, and the Streamlit UI
- Fragile call sites that must be updated whenever a parameter is added

**4. Tight coupling between UI and DSP**

`app_streamlit.py` directly calls `process_audio()` and manually constructs the 20+ parameter dict. The UI also contains DSP-aware logic (mono conversion, tap buffer handling) that should live in the pipeline layer.

**5. `analyses.py` depends on librosa while the rest of the codebase does not**

The STFT was rewritten to a custom implementation (M10), but `analyses.py` still imports librosa directly. This is the only module that requires librosa at runtime, creating an unnecessary heavy dependency for a single analysis function.

---

## 2. File System Refactor

### Current Issues

| Issue | Details |
|---|---|
| `__pycache__` directories checked in | Multiple `__pycache__/` dirs and `.pyc` files are in the repo despite `.gitignore` listing them |
| `.DS_Store` in root | macOS artifact checked into repo |
| `~$ - PRD.docx` temp file | Office temp file in project root |
| `V2 - PRD.docx` and `V2 - Dev Plan .docx` | Binary Office documents in project root — not version-controllable |
| `0. GitHelpers/` folder | Contains a single markdown file with commit instructions — oddly named, doesn't belong |
| `test_data/` in project root | 11 WAV files (~large) duplicating the purpose of `tests/data/` |
| `examples/` contains both input and output WAVs | Mixing source material with generated output |
| `examples/visualizations/` | Generated PNG files that should not be in version control |
| Docs scattered across root | `STFT_WINDOWING_AUDIT.md`, `V2_STATUS_ASSESSMENT.md` in root alongside `docs/` |
| 9 milestone note files | Internal development log that adds clutter without aiding contributors |
| `presets/` as a top-level directory | Contains a single JSON file — could live inside the package |

### Recommended Directory Structure

```
quantum_distortion/
├── __init__.py
├── config.py                    # Global defaults
├── presets.py                   # Preset loader (move presets.json alongside)
├── presets.json                 # Move from presets/ into package
├── dsp/
│   ├── __init__.py
│   ├── pipeline.py              # Simplified orchestrator (<200 lines)
│   ├── processors.py            # NEW: extracted processing stages
│   ├── quantizer.py
│   ├── spectral_fx.py
│   ├── distortion.py            # Merge saturation.py into this
│   ├── limiter.py
│   ├── crossover.py
│   ├── stft_utils.py
│   └── analyses.py
├── io/
│   ├── __init__.py
│   └── audio_io.py
└── ui/
    ├── __init__.py
    ├── app_streamlit.py
    └── visualizers.py

scripts/
├── render_cli.py
├── render_preset.py
├── profile_pipeline.py
├── validate_dsp_metrics.py
├── generate_test_fixtures.py
└── preview_visualizers.py

tests/
├── data/                        # Test fixtures only
│   └── README.md
├── utils/
│   ├── __init__.py
│   └── audio_test_utils.py
├── test_*.py

docs/
├── architecture.md              # High-level architecture and signal flow
├── api.md                       # process_audio API reference
├── development.md               # Setup, testing, contributing
├── presets.md                    # Preset system documentation
└── milestone-notes/             # Keep but flatten naming (optional, could archive)
```

### Files to Remove

| File/Directory | Reason |
|---|---|
| `0. GitHelpers/` | Not a contributor-facing resource; commit workflow is standard git |
| `~$ - PRD.docx` | Office temp file |
| `.DS_Store` | macOS artifact (add to `.gitignore`) |
| `V2 - PRD.docx`, `V2 - Dev Plan .docx` | Binary docs don't belong in a code repo; move to external wiki/drive |
| `test_data/` (root-level) | Duplicate purpose of `tests/data/`; 11 WAV files taking space |
| `examples/visualizations/` | Generated output; regenerate with script |
| `STFT_WINDOWING_AUDIT.md` | **Completely stale** — describes old librosa-based STFT that no longer exists |
| `V2_STATUS_ASSESSMENT.md` | Point-in-time assessment; archive or move to docs |
| `presets/` directory | Merge single JSON file into package |
| All `__pycache__/` directories | Should never be committed |
| `scripts/test_null_passthrough.py` | Duplicates `tests/test_passthrough_null.py` |

### Why This Structure Is Better

1. **Package is self-contained** — presets JSON lives inside the package, eliminating path-resolution gymnastics
2. **No generated artifacts in VCS** — visualization PNGs and processed WAVs are excluded
3. **Clear separation** — `scripts/` for CLI tools, `tests/` for tests, `docs/` for documentation
4. **Flat docs structure** — single source of truth for each topic instead of scattered markdown files
5. **No binary documents** — removes `.docx` files that can't be meaningfully diffed

---

## 3. Dead Code Report

### Confirmed Dead Code

| Item | Location | Evidence |
|---|---|---|
| `formant_shift_frame()` | `spectral_fx.py:116-195` | **Never called anywhere** in the codebase. Not imported by any module. Not tested. UI shows "Feature not yet implemented" message. This is a complete unused implementation. |
| `build_harmonic_target_bins()` | `quantizer.py:204-255` | **Never called** from pipeline or any other module. Exists as dead code from M12 planning. No test calls it directly. |
| `_RenderTiming` dataclass | `pipeline.py` (within large file) | Used only for `print()` debug logging inside `process_audio`. The timing data is never returned or consumed by callers. |
| `SPECTRAL_FX_PRESETS` dict | `spectral_fx.py:34-113` | Only consumed by `scripts/quick_regression_suite.py`. Not used by the main pipeline, UI, or preset system. These are "shadow presets" disconnected from the actual preset JSON. |
| `target_sr` parameter | `audio_io.py:10` | Parameter accepted but **never used** — function always returns audio at original sample rate |
| `window` parameter | `stft_utils.py:16,111` | Parameter accepted but **always ignored** — function hardcodes Hann window |
| `hop_length` parameter | `stft_utils.py:15,110` | Parameter accepted but **always ignored** — function hardcodes `n_fft // 4` |
| `scripts/test_null_passthrough.py` | `scripts/` | Functionality fully duplicated by `tests/test_passthrough_null.py` |
| `0. GitHelpers/commitPush.md` | Root | Internal workflow note, not code |

### Potential Dead Code — Requires Confirmation

| Item | Location | Reason for Uncertainty |
|---|---|---|
| `test_data/` (root-level WAV files) | Project root | 11 WAV files that appear to be real-world test clips. Not referenced by any test or script. May be used manually for listening tests. |
| `examples/*.wav` files | `examples/` | Some may be referenced in README demo commands (`example_bass.wav`), others appear to be generated output files |
| `docs/ui_streamlit_v2_plan.md` | `docs/` | V2 UI has been implemented; this planning doc may be obsolete |
| `docs/MilestoneNotes/M0-M8` | `docs/MilestoneNotes/` | Internal development history. May have value for project context but adds cognitive load for new contributors |

---

## 4. Code Quality Issues

### 4.1 — God Function: `process_audio()` (~190 lines of branching logic)

**Problem:** `process_audio()` in `pipeline.py` handles single-band processing, multiband processing, passthrough testing, preview mode, timing, and tap buffer management all in one function with deeply nested conditional branches.

**Why it's problematic:** Adding new features (M12: spectral freeze, formant shift) requires modifying this already-complex function. Each new feature adds another branch, increasing cyclomatic complexity and making the function harder to test in isolation.

**Recommended improvement:** Extract a `PipelineConfig` dataclass and refactor to a processor chain pattern:

```python
@dataclass
class PipelineConfig:
    key: str = "D"
    scale: str = "minor"
    snap_strength: float = 1.0
    smear: float = 0.1
    bin_smoothing: bool = True
    pre_quant: bool = True
    post_quant: bool = True
    distortion_mode: str = "wavefold"
    distortion_params: dict = field(default_factory=dict)
    limiter_on: bool = True
    limiter_ceiling_db: float = -1.0
    dry_wet: float = 1.0
    use_multiband: bool = False
    crossover_hz: float = 300.0
    lowband_drive: float = 1.0
    spectral_fx_mode: str | None = None
    spectral_fx_strength: float = 0.0
    spectral_fx_params: dict = field(default_factory=dict)

def process_audio(audio: np.ndarray, sr: int, config: PipelineConfig) -> tuple[np.ndarray, dict]:
    ...
```

### 4.2 — Duplicated Mono Conversion Pattern

**Problem:** The pattern of converting audio to mono float32 is copy-pasted in at least 6 places:

- `pipeline.py` (lines ~770-780)
- `harness.py` (lines 66-69)
- `render_preset.py` (lines 44-46)
- `render_cli.py` (implicit — passes raw audio)
- `profile_pipeline.py` (lines 28-30)
- `validate_dsp_metrics.py` (lines 29-31)
- `app_streamlit.py` (multiple locations)

```python
# This exact pattern appears 6+ times:
x = np.asarray(audio, dtype=np.float32)
if x.ndim == 2:
    x = x.mean(axis=1).astype(np.float32)
```

**Why it's problematic:** If the mono conversion logic needs to change (e.g., to support channel selection instead of averaging), every call site must be updated.

**Recommended improvement:** Add `ensure_mono_float32()` to `audio_io.py` and use it everywhere:

```python
def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32)
    return x
```

### 4.3 — Duplicate `soft_tube` Implementations

**Problem:** Two functions with the same name `soft_tube` exist in different modules with different signatures and behavior:

- `distortion.py:soft_tube(audio, drive, warmth)` — shape varies with warmth: `a = 1 + 4*warmth`
- `saturation.py:soft_tube(x, drive)` — fixed shape: `a = 3.0`

**Why it's problematic:** The name collision is confusing. `pipeline.py` imports both: `from quantum_distortion.dsp.distortion import apply_distortion` (which wraps distortion's `soft_tube`) and `from quantum_distortion.dsp.saturation import soft_tube` (the saturation version). A developer reading the code must track which `soft_tube` is in scope at each call site.

**Recommended improvement:** Rename `saturation.py:soft_tube` to `saturate_lowband()` to clearly distinguish its purpose. Or merge both into `distortion.py` as `soft_tube_distortion()` and `soft_tube_saturation()`.

### 4.4 — `_ensure_mono_float` vs Other Mono Checks

**Problem:** `distortion.py` has `_ensure_mono_float()` that raises on non-1D input. Meanwhile, `saturation.py` explicitly handles both 1D and 2D inputs. `crossover.py` also handles both. There's no consistent contract for what shape audio functions expect.

**Why it's problematic:** Callers must know each module's expectation individually. Some modules silently handle stereo, others crash.

**Recommended improvement:** Standardize: either all DSP functions accept mono-only (and mono conversion happens once at the entry point), or all accept both mono and stereo.

### 4.5 — `app_streamlit.py` Contains Both V1 and V2 UI (~900 lines)

**Problem:** The Streamlit app file contains both the V1 UI (`render_v1_ui()`) and V2 UI (`render_v2_ui()`), selected by a feature flag. This makes the file nearly 900 lines long.

**Why it's problematic:** The V2 UI is now the default. The V1 code is dead weight that inflates the file and confuses contributors about which code path is active.

**Recommended improvement:** Remove V1 UI code entirely. If V1 is needed for reference, it exists in git history.

### 4.6 — `sys.path` Manipulation in `app_streamlit.py`

**Problem:** Lines 35-37 of `app_streamlit.py`:
```python
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
```

**Why it's problematic:** This is a hack to make imports work when running via `streamlit run`. It's fragile and breaks if the file moves.

**Recommended improvement:** Use a proper `pyproject.toml` or `setup.py` with `pip install -e .` so the package is on the path naturally.

### 4.7 — Ignored Parameters That Suggest Incomplete API Design

**Problem:** Several functions accept parameters they completely ignore:

- `stft_utils.py`: `hop_length` and `window` are accepted but always overridden
- `audio_io.py`: `target_sr` is accepted but never used

**Why it's problematic:** Callers may believe they're customizing behavior when they're not. This is a lie in the API surface.

**Recommended improvement:** Remove ignored parameters. If configurability is planned for the future, add them when implementing the feature — not before.

### 4.8 — Peak Limiter Uses Python Loop Over Every Sample

**Problem:** `limiter.py:62-77` iterates over every sample with a Python `for` loop. For a 5-second clip at 48kHz, that's 240,000 iterations in pure Python.

**Why it's problematic:** This is extremely slow. The limiter is called on every render and contributes non-trivially to processing time.

**Recommended improvement:** Vectorize using NumPy's `np.maximum.accumulate` or use Numba (already a dependency) to JIT-compile the loop.

---

## 5. Readability Improvements

### 5.1 — Naming Conventions

| Current Name | Issue | Suggested Name |
|---|---|---|
| `_SpectralFXConfig` | Leading underscore suggests private, but it's a key config type | `SpectralFXConfig` |
| `soft_tube` (saturation.py) | Collides with `soft_tube` in distortion.py | `saturate_lowband` |
| `_ensure_mono_float` | Only used in distortion.py — fine as private, but name should match the pattern elsewhere | `_to_mono_float32` (consistent with float32 output) |
| `_select_segment` | Generic name for a visualization utility | `select_display_segment` |
| `apply_distortion` | Too generic — could mean any kind of distortion | `apply_time_domain_distortion` |
| `process_audio` | Overly generic for a 20-parameter function | Keep name, but reduce parameter count via config object |
| `_load_presets_raw` | "raw" is unclear — all presets are loaded from JSON | `_load_presets` |
| `formant_shift_frame` | Function exists but is dead code | Remove entirely |
| `TapSource` | Good name, but defined in `visualizers.py` which limits reuse | Move to `config.py` or a shared types module |
| `ScaleName` | Defined in both `quantizer.py` and `analyses.py` | Define once, import everywhere |

### 5.2 — Function Length

| Function | Lines | Recommendation |
|---|---|---|
| `process_audio()` | ~190 | Split into `_process_singleband()`, `_process_multiband()`, and a thin dispatcher |
| `render_v2_ui()` | ~450 | Split into panel-rendering functions (already partially done but panels are too large) |
| `quantize_spectrum()` | ~170 | Already reasonably structured; the Numba fallback block could be extracted |
| `_apply_spectral_quantization_to_stft()` | ~120 | Reasonable for its complexity |
| `process_file_to_file()` | ~70 | Reasonable, but duplicates parameter threading from `render_preset.py` |

### 5.3 — Inconsistent Import Style

- Some files use `from __future__ import annotations` (good), others don't
- Some files put blank lines between every import group, others don't
- `typing` imports mix old-style (`Dict`, `Tuple`, `Union`) with new-style (`dict`, `tuple`, `str | None`) — this is due to Python 3.7 compatibility target, but the codebase uses `tuple[...]` syntax in newer files which breaks on 3.7

### 5.4 — Python 3.7 Target Is Outdated

`pyrightconfig.json` specifies `pythonVersion: "3.7"`. Python 3.7 reached end-of-life in June 2023. Meanwhile:
- The code uses `tuple[...]` and `str | None` syntax (requires 3.10+)
- The `from __future__ import annotations` import partially mitigates this, but not at runtime
- NumPy, SciPy, and other dependencies have dropped 3.7 support

**Recommendation:** Update target to Python 3.10+ and remove all `try: from typing import Literal / except: from typing_extensions` blocks.

---

## 6. Documentation Reorganization

### Current State

Documentation is scattered across 15+ locations:

| Location | Content | Status |
|---|---|---|
| `README.md` | Usage instructions, CLI examples, preset demos | **Current but minimal** |
| `STFT_WINDOWING_AUDIT.md` | OLA/windowing analysis | **Completely stale** — describes old librosa-based code |
| `V2_STATUS_ASSESSMENT.md` | Milestone-by-milestone status report | **Point-in-time snapshot**, useful but not a living doc |
| `docs/audio_pipeline_overview.md` | API reference for pipeline and scripts | **Mostly current** |
| `docs/ui_streamlit_v2_plan.md` | V2 UI planning document | **Obsolete** — V2 is implemented |
| `docs/MilestoneNotes/M0-M8` | 9 milestone development logs | **Historical** — internal dev notes |
| `0. GitHelpers/commitPush.md` | Git commit workflow for AI assistant | **Internal tooling**, not user-facing |
| `tests/data/README.md` | Test fixture description | Minimal but fine |
| Inline docstrings | Function/module docstrings | Generally good in DSP modules, sparse in scripts |

### Proposed Documentation Structure

```
docs/
├── architecture.md       # Signal flow, module map, design decisions
├── api.md                # process_audio() reference, config parameters, presets
├── development.md        # Setup, dependencies, running tests, adding features
├── changelog.md          # Consolidate milestone notes into a changelog format
```

**What goes where:**

| Document | Content Sources |
|---|---|
| `architecture.md` | Pipeline diagram from `V2_STATUS_ASSESSMENT.md`, module descriptions from `audio_pipeline_overview.md`, design rationale |
| `api.md` | `process_audio()` signature and parameter docs from `audio_pipeline_overview.md`, preset system from README, spectral FX modes |
| `development.md` | Setup instructions (currently missing), dependency installation, test running, fixture generation, profiling |
| `changelog.md` | Condensed summaries from `docs/MilestoneNotes/M0-M8` in reverse chronological order |

**Files to remove after consolidation:**

- `STFT_WINDOWING_AUDIT.md` — stale, references non-existent code
- `docs/ui_streamlit_v2_plan.md` — V2 is implemented
- `0. GitHelpers/commitPush.md` — internal AI assistant prompt
- `docs/audio_pipeline_overview.md` — content moves to `api.md`

**Files to archive (move to `docs/archive/` or delete):**

- `docs/MilestoneNotes/M0-M8` — content condensed into `changelog.md`
- `V2_STATUS_ASSESSMENT.md` — point-in-time assessment, archive

---

## 7. Step-by-Step Refactor Plan

### Step 1 — Clean Up Repository Hygiene

**What to change:**
- Remove all `__pycache__/` directories from git tracking
- Remove `.DS_Store`, `~$ - PRD.docx`
- Remove or move `V2 - PRD.docx`, `V2 - Dev Plan .docx` out of repo
- Remove `0. GitHelpers/` directory
- Update `.gitignore` to include: `.DS_Store`, `*.docx`, `~$*`, `examples/visualizations/`, `test_data/`, `tests/data/processed/`
- Remove `examples/visualizations/*.png` (generated files)
- Remove or relocate `test_data/` root-level WAV files

**Why:** Reduces repository size and noise. Binary files and generated output don't belong in version control.

**Risk level:** LOW — no code changes, purely cleanup.

---

### Step 2 — Remove Dead Code and Stale Documentation

**What to change:**
- Delete `formant_shift_frame()` from `spectral_fx.py` (dead code, never called)
- Delete `STFT_WINDOWING_AUDIT.md` (describes code that no longer exists)
- Delete `docs/ui_streamlit_v2_plan.md` (V2 is implemented)
- Delete `scripts/test_null_passthrough.py` (duplicated by `tests/test_passthrough_null.py`)
- Remove unused `target_sr` parameter from `load_audio()` in `audio_io.py`
- Remove ignored `hop_length` and `window` parameters from `stft_mono()` and `istft_mono()`

**Why:** Dead code and stale docs mislead contributors and increase cognitive load.

**Risk level:** LOW — removing unused code. Run tests to confirm nothing breaks.

---

### Step 3 — Extract Shared Utilities

**What to change:**
- Add `ensure_mono_float32()` to `audio_io.py`
- Replace all 6+ instances of the mono conversion pattern with this utility
- Move duplicated `ScaleName` type alias to `config.py` (currently defined in both `quantizer.py` and `analyses.py`)
- Move `TapSource` type alias to `config.py`

**Why:** Eliminates copy-paste code and establishes a single source of truth for shared types.

**Risk level:** LOW — mechanical refactor with no behavior change. Run tests to confirm.

---

### Step 4 — Resolve `soft_tube` Naming Conflict

**What to change:**
- Rename `saturation.py:soft_tube()` to `saturate_lowband()`
- Update the single import in `pipeline.py`
- Update tests that reference this function

**Why:** Two functions with the same name in the same package is confusing and error-prone.

**Risk level:** LOW — rename only, no behavior change.

---

### Step 5 — Introduce `PipelineConfig` Dataclass

**What to change:**
- Create a `PipelineConfig` dataclass in `config.py` (or a new `pipeline_config.py`)
- The dataclass captures all 20+ parameters of `process_audio()`
- Add a `from_preset(name: str)` class method that loads a preset into the config
- Update `process_audio()` to accept `config: PipelineConfig` (keep backward-compatible kwargs for now)
- Update `harness.py` to build a `PipelineConfig` instead of manually threading parameters
- Update CLI scripts to build `PipelineConfig`

**Why:** Reduces parameter threading, makes the API self-documenting, and enables config serialization/comparison.

**Risk level:** MEDIUM — touches multiple call sites. Must maintain backward compatibility during transition.

---

### Step 6 — Split `process_audio()` into Smaller Functions

**What to change:**
- Extract passthrough test logic into `_passthrough_test()`
- Extract single-band processing into `_process_singleband(audio, sr, config)`
- Extract multiband processing into `_process_multiband(audio, sr, config)`
- Make `process_audio()` a thin dispatcher (~30 lines)

**Why:** The current function is 190+ lines of deeply nested conditionals. Splitting it makes each path independently testable and readable.

**Risk level:** MEDIUM — refactoring core logic. Requires thorough regression testing.

---

### Step 7 — Remove V1 UI Code

**What to change:**
- Remove `render_v1_ui()` function and all V1-only code from `app_streamlit.py`
- Remove the `USE_V2_UI` feature flag
- Remove the V1/V2 dispatch in `main()`

**Why:** V2 is the default and only active code path. V1 is dead weight (~200 lines).

**Risk level:** LOW — removing unused code path. V1 exists in git history if needed.

---

### Step 8 — Normalize Naming and Style Conventions

**What to change:**
- Update `pyrightconfig.json` to target Python 3.10+
- Remove all `try: from typing import Literal` blocks (use direct imports)
- Remove all `from __future__ import annotations` imports (no longer needed with 3.10+)
- Standardize import ordering across all files (stdlib → third-party → local)
- Remove extraneous blank lines in `render_cli.py` and `audio_io.py` (inconsistent with other files)

**Why:** Consistent style reduces cognitive load and eliminates compatibility shims for an EOL Python version.

**Risk level:** LOW — mechanical changes. Must verify no runtime code depends on 3.7 annotations behavior.

---

### Step 9 — Consolidate Documentation

**What to change:**
- Create `docs/architecture.md` (from V2 assessment + pipeline overview)
- Create `docs/api.md` (from audio_pipeline_overview.md)
- Create `docs/development.md` (new: setup, testing, contributing)
- Create `docs/changelog.md` (condensed from MilestoneNotes)
- Update `README.md` to link to docs/ instead of duplicating content
- Archive `docs/MilestoneNotes/` to `docs/archive/milestone-notes/`
- Delete consolidated source files

**Why:** Contributors should find all documentation in a predictable, organized structure.

**Risk level:** LOW — documentation only, no code changes.

---

### Step 10 — Add `pyproject.toml` and Eliminate `sys.path` Hacks

**What to change:**
- Add `pyproject.toml` with project metadata, dependencies, and entry points
- Move `requirements.txt` content into `pyproject.toml` `[project.dependencies]`
- Remove `sys.path` manipulation in `app_streamlit.py`
- Remove `sys.path` manipulation in `quick_regression_suite.py`

**Why:** A proper Python package with `pip install -e .` eliminates path hacks and makes the project installable.

**Risk level:** LOW — standard Python packaging. Must verify all imports still resolve.

---

## 8. Optional Improvements

### 8.1 — Performance: Vectorize Peak Limiter

The peak limiter in `limiter.py` uses a pure Python loop over every sample. Since `numba` is already a dependency, JIT-compiling the limiter loop would be a straightforward performance win. Alternatively, a vectorized approach using `np.maximum.accumulate` for the attack phase would remove the loop entirely.

### 8.2 — Performance: Cache Target Bins

As identified in `V2_STATUS_ASSESSMENT.md`, `build_target_bins_for_freqs()` is recomputed every STFT frame despite returning the same result (the frequency-to-scale mapping is static for a given key/scale). Caching this outside the frame loop would reduce render time from ~13s to ~2-3s for a 5s clip. This is the single highest-impact optimization available.

### 8.3 — Make `librosa` Optional

`analyses.py` is the only module that requires `librosa`. Since analysis is not part of the core processing pipeline, `librosa` could be made an optional dependency:

```python
try:
    import librosa
except ImportError:
    librosa = None

def avg_cents_offset_from_scale(...):
    if librosa is None:
        raise ImportError("librosa is required for analysis functions: pip install librosa")
    ...
```

This reduces the mandatory dependency footprint significantly (`librosa` pulls in many transitive dependencies).

### 8.4 — Add `setup.cfg` or `pyproject.toml` Entry Points

Define CLI entry points so users can run `qd-render` instead of `python scripts/render_cli.py`:

```toml
[project.scripts]
qd-render = "scripts.render_cli:main"
qd-render-preset = "scripts.render_preset:main"
qd-profile = "scripts.profile_pipeline:main"
```

### 8.5 — Type Safety: Replace `Dict[str, Any]` Pattern

The `distortion_params`, `spectral_fx_params`, and preset configs all use `Dict[str, Any]` — completely untyped. Introducing typed dataclasses for each parameter group would catch errors at development time:

```python
@dataclass
class WavefoldParams:
    fold_amount: float = 1.0
    bias: float = 0.0
    threshold: float = 1.0

@dataclass
class TubeParams:
    drive: float = 1.0
    warmth: float = 0.5
```

### 8.6 — Improve `.gitignore`

The current `.gitignore` is minimal (4 lines). Recommended additions:

```
# OS files
.DS_Store
Thumbs.db

# Office temp files
~$*

# Generated output
examples/visualizations/
tests/data/processed/
*.wav  # Or be selective

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.pytest_cache/
.mypy_cache/

# IDE
.vscode/
.idea/
```

---

## Summary of Priorities

| Priority | Action | Impact | Effort |
|---|---|---|---|
| **P0** | Cache `build_target_bins_for_freqs()` outside frame loop | 5-10x performance improvement | Small |
| **P0** | Remove `__pycache__`, `.DS_Store`, temp files from repo | Repo hygiene | Small |
| **P1** | Remove dead code (`formant_shift_frame`, stale docs, unused params) | Reduce confusion | Small |
| **P1** | Resolve `soft_tube` naming conflict | Eliminate ambiguity | Small |
| **P1** | Extract `ensure_mono_float32()` utility | Eliminate 6+ duplications | Small |
| **P2** | Introduce `PipelineConfig` dataclass | Simplify API surface | Medium |
| **P2** | Split `process_audio()` into smaller functions | Improve maintainability | Medium |
| **P2** | Remove V1 UI code | Remove ~200 lines of dead code | Small |
| **P2** | Update Python target to 3.10+ | Remove compatibility shims | Small |
| **P3** | Consolidate documentation | Improve contributor experience | Medium |
| **P3** | Add `pyproject.toml` | Proper packaging | Medium |
| **P3** | Vectorize peak limiter | Performance improvement | Medium |
| **P3** | Make `librosa` optional | Reduce dependency footprint | Small |

---

*Review generated by Claude Opus 4.6 — 2026-03-11*
