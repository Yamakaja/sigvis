"""Render eye diagrams and save as PNG."""

import sys
import pathlib
import time
import numpy as np
from PIL import Image
import matplotlib.cm as mcm

EXAMPLES = pathlib.Path(__file__).parent
BUILD = EXAMPLES.parent / "build" / "pyrendering"
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(EXAMPLES))
import pyrendering as pr
import sigutils


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


def render_oneshot(width: int, height: int, n_traces: int = 1_000_000) -> None:
    renderer = pr.EyeDiagramRenderer(width, height)
    samples = make_sinusoid_traces(n_traces=n_traces, samples_per_trace=64, amplitude_variation=0.05)
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


def render_batched(
    width: int,
    height: int,
    total_traces: int = 2_000_000,
    batch_size: int = 100_000,
    samples_per_trace: int = 64,
) -> None:
    hist = pr.Histogram(width, height)
    hist.clear()

    rng = np.random.default_rng(0)
    n_batches = (total_traces + batch_size - 1) // batch_size
    rendered = 0
    t_dsp_total = 0.0
    t_render_total = 0.0

    for i in range(n_batches):
        n = min(batch_size, total_traces - rendered)

        t0 = time.perf_counter()
        batch = make_sinusoid_traces(n_traces=n, samples_per_trace=samples_per_trace,
                                     amplitude_variation=0.05)
        t_dsp_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        hist.draw(batch, x_range=(0.0, 1.0), y_range=(-1.5, 1.5), line_width=5e-3)
        t_render_total += time.perf_counter() - t0

        rendered += n
        print(f"  [{i+1}/{n_batches}] {rendered:>10,} / {total_traces:,} traces", flush=True)

    t0 = time.perf_counter()
    data = hist.download()
    t_download = time.perf_counter() - t0

    print(f"Histogram: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
    print(f"  DSP:      {t_dsp_total*1e3:7.1f} ms total")
    print(f"  draw():   {t_render_total*1e3:7.1f} ms total  (memcpy + submit, GPU runs async)")
    print(f"  download:{t_download*1e3:7.1f} ms  (flush + readback)")
    frame = apply_colormap(data)
    out = pathlib.Path(__file__).parent / "eye_diagram_batched.png"
    Image.fromarray(frame, "RGBA").save(out)
    print(f"Saved → {out}")
    del hist


def make_rcf_filter(sps: int, beta: float, n_syms: int = 8) -> np.ndarray:
    """Raised cosine filter kernel spanning ±n_syms/2 symbol periods."""
    half = n_syms // 2
    t = np.linspace(-half, half, n_syms * sps + 1)
    h, _ = sigutils.rcf(t, np.ones(1), beta)
    return h


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _overlap_save_circular(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Overlap-save convolution treating x as periodic: the first block's overlap
    is taken from the tail of x so every output sample sees steady-state filter
    context — no startup transients at the trace boundaries in the eye diagram.
    All blocks are processed in one batched FFT call (no Python loop).
    """
    M = len(h) - 1                          # filter memory
    block_size = _next_pow2(4 * len(h))     # power-of-2 FFT size
    step = block_size - M                   # new samples per block
    N = len(x)

    # Pad signal to a multiple of step, wrapping circularly.
    pad = (-N) % step
    x_pad = np.concatenate([x, x[:pad]]) if pad else x
    N_p = N + pad

    # Prepend the last M samples for circular first-block overlap.
    x_ext = np.concatenate([x_pad[-M:], x_pad])  # length M + N_p

    H = np.fft.rfft(h, n=block_size)

    # Build all blocks as a strided view (no copy) → batch FFT in one call.
    blocks = np.lib.stride_tricks.sliding_window_view(x_ext, block_size)[::step]
    # shape: (N_p // step, block_size)
    Y = np.fft.irfft(np.fft.rfft(blocks, axis=1) * H[None, :], n=block_size, axis=1)
    return Y[:, M:].ravel()[:N].astype(np.float32)


def make_pam4_eye_traces(
    n_symbols: int = 65536,
    sps: int = 64,
    noise_sigma: float = 0.02,
    rng: np.random.Generator | None = None,
    h: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate PAM4 eye diagram traces via raised cosine filtering.
    Returns float32 array of shape (n_symbols//2, 2*sps, 2): [x, y].
    Each trace spans two symbol periods; x is normalised to [-1, 1].
    Pass precomputed h (from make_rcf_filter) to avoid recomputing it each call.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if h is None:
        h = make_rcf_filter(sps, beta=0.8)

    C = np.array([-3, -1, 1, 3], dtype=np.float64) / np.sqrt(5)
    symbols = C[rng.integers(0, 4, n_symbols)]
    if noise_sigma > 0:
        symbols += noise_sigma * rng.standard_normal(n_symbols)

    upsampled = np.zeros(n_symbols * sps)
    upsampled[::sps] = symbols

    filtered = _overlap_save_circular(upsampled, h)

    traces = filtered.reshape(n_symbols // 2, 2 * sps)
    x = np.broadcast_to(
        np.linspace(-1.0, 1.0, 2 * sps, endpoint=False, dtype=np.float32),
        traces.shape,
    )
    return np.stack([x, traces], axis=-1).astype(np.float32)


def render_pam4_eye(
    width: int,
    height: int,
    total_symbols: int = 1_000_000,
    batch_size: int = 100_000,
    sps: int = 64,
) -> None:
    hist = pr.Histogram(width, height)
    hist.clear()

    rng = np.random.default_rng(0)
    n_batches = (total_symbols + batch_size - 1) // batch_size
    rendered = 0
    t_dsp_total = 0.0
    t_render_total = 0.0

    h = make_rcf_filter(sps, beta=0.8)

    for i in range(n_batches):
        n = min(batch_size, total_symbols - rendered)

        t0 = time.perf_counter()
        batch = make_pam4_eye_traces(n_symbols=n, sps=sps, rng=rng,
                                     noise_sigma=0.004, h=h)
        t_dsp_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        hist.draw(batch, x_range=(-1.0, 1.0), y_range=(-2.0, 2.0), line_width=1e-3)
        t_render_total += time.perf_counter() - t0

        rendered += n
        print(f"  [{i+1}/{n_batches}] {rendered:>10,} / {total_symbols:,} symbols", flush=True)

    t0 = time.perf_counter()
    data = hist.download()
    t_download = time.perf_counter() - t0

    print(f"Histogram: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
    print(f"  DSP:      {t_dsp_total*1e3:7.1f} ms total")
    print(f"  draw():   {t_render_total*1e3:7.1f} ms total  (memcpy + submit, GPU runs async)")
    print(f"  download:{t_download*1e3:7.1f} ms  (flush + readback)")
    frame = apply_colormap(data)
    out = pathlib.Path(__file__).parent / "pam4_eye.png"
    Image.fromarray(frame, "RGBA").save(out)
    print(f"Saved → {out}")
    del hist


def main():
    pr.init(validation_layers=False)
    caps = pr.capabilities()
    print("Hardware capabilities:")
    for k, v in caps.items():
        print(f"  {k}: {v}")

    width, height = 2*1920, 2*1080

    print("\n--- oneshot ---")
    # render_oneshot(width, height)

    print("\n--- batched (2M traces) ---")
    # render_batched(width, height)

    print("\n--- PAM4 eye (1M symbols) ---")
    render_pam4_eye(width, height)

    pr.shutdown()


if __name__ == "__main__":
    main()
