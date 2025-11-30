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
