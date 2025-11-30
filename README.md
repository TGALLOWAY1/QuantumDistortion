# Quantum Distortion (MVP)

Mockup
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/2000d791-a24c-44bf-aecd-5eb478be0ce0" />



Experimental DSP engine combining spectral pitch quantization with time-domain distortion.



MVP features:

- Load audio files

- Process through pre-quant → distortion → post-quant → limiter

- Visualize tap points (spectrum, oscilloscope, phase scope)



This MVP is a Python prototype intended to validate the DSP design before building a JUCE VST plugin.

## Usage

### CLI Offline Render

```bash
python scripts/render_cli.py \
  --infile examples/example_bass.wav \
  --outfile examples/example_bass_qd.wav
```

### Streamlit UI

```bash
streamlit run quantum_distortion/ui/app_streamlit.py
```

Then open the provided local URL in your browser, load an audio file, tweak the controls, and hit Render to hear and visualize the results.

## Validation & Profiling

### Runtime Profiling

Profile the full DSP pipeline on an input file:

```bash
python scripts/profile_pipeline.py \
  --infile examples/example_bass.wav
```

This prints processing time and an approximate real-time factor.

### Scale Alignment Metric

Check how tightly the processed audio aligns to the chosen key/scale (average cents offset of dominant peaks):

```bash
python scripts/validate_dsp_metrics.py \
  --infile examples/example_bass.wav \
  --key C \
  --scale minor
```

The script reports the average absolute cents offset for:

- The input signal
- The processed output

and the difference between them.

## Presets & Demo Recipes

Quantum Distortion ships with a small set of named presets in:

- `presets/quantum_distortion_presets.json`

You can list them and render audio using any preset via the CLI:

```bash
# List all available presets
python scripts/render_preset.py --list-presets
```

Example demo commands:

```bash
# 1) Chordal Noise Wash
python scripts/render_preset.py \
  --infile examples/example_noise.wav \
  --outfile examples/example_noise_chordal_wash.wav \
  --preset "Chordal Noise Wash"

# 2) Controlled Dubstep Growl
python scripts/render_preset.py \
  --infile examples/example_bass.wav \
  --outfile examples/example_bass_growl.wav \
  --preset "Controlled Dubstep Growl"

# 3) Perc To Tonal Clang
python scripts/render_preset.py \
  --infile examples/example_perc.wav \
  --outfile examples/example_perc_clang.wav \
  --preset "Perc To Tonal Clang"
```

These are good "portfolio-ready" demonstrations to showcase:

- Pre/Post spectral quantization
- Smear + bin smoothing
- Wavefold / tube distortion
- Limiter safety net keeping things in check
