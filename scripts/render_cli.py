from pathlib import Path

import argparse



from quantum_distortion.io.audio_io import load_audio, save_audio

from quantum_distortion.dsp.pipeline import process_audio



def main() -> None:
    """
    Quantum Distortion - Offline Renderer

    Loads an audio file, runs it through the full DSP pipeline with default
    parameters (pre/post quantization, distortion, limiter), and saves the result.
    """
    parser = argparse.ArgumentParser(description="Quantum Distortion - Offline Renderer")

    parser.add_argument("--infile", "-i", required=True)

    parser.add_argument("--outfile", "-o", required=True)

    args = parser.parse_args()



    audio, sr = load_audio(args.infile)

    processed, taps = process_audio(audio, sr)



    save_audio(args.outfile, processed, sr)



    print(f"Loaded: {args.infile} (sr={sr}, shape={audio.shape})")

    print(f"Saved processed output → {args.outfile}")

    print("Tap buffers:", list(taps.keys()))



if __name__ == "__main__":

    main()
