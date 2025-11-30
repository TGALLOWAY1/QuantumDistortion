from pathlib import Path

import numpy as np



from quantum_distortion.io.audio_io import load_audio, save_audio

from quantum_distortion.dsp.pipeline import process_audio



def test_roundtrip(tmp_path: Path) -> None:

    sr = 44100

    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)

    sine = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)



    in_path = tmp_path / "in.wav"

    out_path = tmp_path / "out.wav"



    save_audio(in_path, sine, sr)

    audio, sr_loaded = load_audio(in_path)



    assert sr_loaded == sr

    assert audio.shape == sine.shape



    processed, taps = process_audio(audio, sr=sr)

    save_audio(out_path, processed, sr)



    assert set(taps.keys()) == {"input", "pre_quant", "post_dist", "output"}
