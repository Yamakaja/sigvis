"""Render a synthetic NRZ eye diagram and save as PNG."""

import sys
import pathlib
import numpy as np
from PIL import Image
import matplotlib.cm as mcm

BUILD = pathlib.Path(__file__).parent.parent / "build" / "pyrendering"
sys.path.insert(0, str(BUILD))
import pyrendering as pr


_inferno = mcm.get_cmap("inferno")


def apply_colormap(
    hist: np.ndarray,
    max_intensity: float | None = None,
    log_min: float = 0.1,
) -> np.ndarray:
    """Map a (H, W) float32 histogram to (H, W, 4) uint8 RGBA via inferno on a log scale.

    log_min: absolute histogram value that maps to 0 (black). Values below are clipped.
    """
    peak = max_intensity if max_intensity is not None else hist.max()
    if peak > log_min:
        log_range = np.log(peak) - np.log(log_min)
        normalized = (np.log(np.clip(hist, log_min, peak)) - np.log(log_min)) / log_range
    else:
        normalized = np.zeros_like(hist)
    rgba = (_inferno(normalized) * 255).astype(np.uint8)
    return rgba


def make_eye_samples(
    n_traces: int = 4000,
    samples_per_trace: int = 64,
    rise_time: float = 0.15,
    noise_rms: float = 0.035,
    jitter_rms: float = 0.012,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate NRZ eye diagram samples.
    Returns float32 array of shape (n_traces, samples_per_trace, 2): [x, y].
    """
    if rng is None:
        rng = np.random.default_rng(0)

    spt = samples_per_trace
    prev_bits = rng.integers(0, 2, size=n_traces).astype(np.float32) * 2 - 1
    curr_bits = rng.integers(0, 2, size=n_traces).astype(np.float32) * 2 - 1

    t = np.tile(np.linspace(0.0, 1.0, spt, dtype=np.float32), (n_traces, 1))
    jitter = rng.standard_normal(n_traces).astype(np.float32) * jitter_rms
    t = np.clip(t + jitter[:, None], 0.0, 1.0)

    def smooth_step(t_local, rt):
        edge = np.clip((t_local - 0.5) / rt + 0.5, 0.0, 1.0)
        return edge * edge * (3.0 - 2.0 * edge)

    alpha = smooth_step(t, rise_time).astype(np.float32)
    y = prev_bits[:, None] * (1.0 - alpha) + curr_bits[:, None] * alpha
    y += rng.standard_normal((n_traces, spt)).astype(np.float32) * noise_rms

    return np.stack([t, y], axis=-1)  # (n_traces, spt, 2)


def make_sinusoid_traces(
    n_traces: int = 5,
    samples_per_trace: int = 64,
    amplitude_variation: float = 0.1,
) -> np.ndarray:
    """n_traces sinusoids with random amplitude variation. Shape: (n_traces, spt, 2)."""
    t = np.linspace(0.0, 1.0, samples_per_trace, dtype=np.float32)
    amplitudes = (1+amplitude_variation*np.random.randn(n_traces)).astype(np.float32)
    y = amplitudes[:, None] * np.sin(2.0 * np.pi * t).astype(np.float32)
    return np.stack([np.tile(t, (n_traces, 1)), y], axis=-1)


def main():
    pr.init(validation_layers=True)

    width, height = 2*1920, 2*1080
    renderer = pr.EyeDiagramRenderer(width, height)

    samples = make_sinusoid_traces(n_traces=10000, samples_per_trace=64, amplitude_variation=0.05)
    print(f"Samples: {samples.shape}")

    renderer.set_samples(samples)

    hist = renderer.render_histogram(
        x_range=(0.0, 1.0),
        y_range=(-1.5, 1.5),
        line_width=5e-3,
    )
    print(f"Histogram: shape={hist.shape}, min={hist.min():.4f}, max={hist.max():.4f}")

    frame = apply_colormap(hist)

    out = pathlib.Path(__file__).parent / "eye_diagram.png"
    Image.fromarray(frame, "RGBA").save(out)
    print(f"Saved → {out}")

    del renderer
    pr.shutdown()


if __name__ == "__main__":
    main()
