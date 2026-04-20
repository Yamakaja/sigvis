"""Render a long 1-D waveform as a phosphor-style accumulation histogram."""

import sys
import pathlib
import numpy as np
from PIL import Image
import matplotlib.cm as mcm

EXAMPLES = pathlib.Path(__file__).parent
BUILD = EXAMPLES.parent / "build" / "pyrendering"
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(EXAMPLES))
import pyrendering as pr


_inferno = mcm.get_cmap("inferno")


def apply_colormap(hist: np.ndarray, log_min: float = 0.05) -> np.ndarray:
    peak = hist.max()
    if peak > log_min:
        log_range = np.log(peak) - np.log(log_min)
        normalized = (np.log(np.clip(hist, log_min, peak)) - np.log(log_min)) / log_range
    else:
        normalized = np.zeros_like(hist)
    return (_inferno(normalized) * 255).astype(np.uint8)


def render_square_wave(width: int = 1920, height: int = 1080) -> None:
    rng = np.random.default_rng(42)
    n = 1_000_000

    # 2 periods of a square wave, noise std = 0.1
    half_period = n // 4  # 4 half-periods = 2 full periods
    square = np.where((np.arange(n) // half_period) % 2 == 0, 1.0, -1.0).astype(np.float32)
    signal = square + 0.1 * rng.standard_normal(n).astype(np.float32)

    hist = pr.Histogram(width, height)
    hist.clear()
    hist.draw_waveform(signal, x_range=(0.0, 1.0), y_range=(-1.5, 1.5), line_width=1e-3)

    data = hist.download()
    print(f"Histogram: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")

    frame = apply_colormap(data)
    out = EXAMPLES / "waveform_square.png"
    Image.fromarray(frame, "RGBA").save(out)
    print(f"Saved → {out}")
    del hist


def main():
    pr.init(validation_layers=False)
    render_square_wave(width=2*1920, height=2*1080)
    pr.shutdown()


if __name__ == "__main__":
    main()
