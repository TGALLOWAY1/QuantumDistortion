from __future__ import annotations


from pathlib import Path


import argparse


import matplotlib.pyplot as plt


from quantum_distortion.io.audio_io import load_audio
from quantum_distortion.dsp.pipeline import process_audio
from quantum_distortion.ui.visualizers import (
    plot_spectrum,
    plot_oscilloscope,
    plot_phase_scope,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Quantum Distortion visualizations.")
    parser.add_argument("--infile", "-i", required=True, help="Input audio file (wav/aif/aiff)")
    parser.add_argument("--outdir", "-o", default="examples/visualizations", help="Output directory for PNGs")
    args = parser.parse_args()

    audio, sr = load_audio(args.infile)
    processed, taps = process_audio(audio, sr=sr)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure taps include the processed output as well
    taps = dict(taps)
    taps["output"] = processed

    for tap_name, buf in taps.items():
        # Spectrum
        fig_spec = plot_spectrum(
            audio=buf,
            sr=sr,
            tap_source=tap_name,  # type: ignore[arg-type]
            key="C",
            scale="minor",
            show_scale_lines=True,
            max_freq=sr / 2.0,
        )
        spec_path = outdir / f"{tap_name}_spectrum.png"
        fig_spec.savefig(spec_path, dpi=120, bbox_inches="tight")
        plt.close(fig_spec)

        # Oscilloscope
        fig_osc = plot_oscilloscope(
            audio=buf,
            sr=sr,
            tap_source=tap_name,  # type: ignore[arg-type]
            duration=0.02,
        )
        osc_path = outdir / f"{tap_name}_osc.png"
        fig_osc.savefig(osc_path, dpi=120, bbox_inches="tight")
        plt.close(fig_osc)

        # Phase scope
        fig_phase = plot_phase_scope(
            audio=buf,
            sr=sr,
            tap_source=tap_name,  # type: ignore[arg-type]
        )
        phase_path = outdir / f"{tap_name}_phase.png"
        fig_phase.savefig(phase_path, dpi=120, bbox_inches="tight")
        plt.close(fig_phase)

        print(f"Saved visualizations for tap '{tap_name}' to {outdir}")

    print("Done.")


if __name__ == "__main__":
    main()

