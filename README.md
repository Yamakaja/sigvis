# sigvis

GPU-accelerated signal visualization. Renders eye diagrams, density histograms, and long 1-D
waveforms using Vulkan mesh shaders with additive accumulation (phosphor-oscilloscope style).
Headless by design — the primary consumer is the `pyrendering` Python extension.

## Requirements

A Vulkan 1.3 driver with **mesh shader support** (`VK_EXT_mesh_shader`) is required. On modern
hardware this means a recent NVIDIA, AMD (RDNA2+), or Intel Arc GPU with up-to-date drivers.

### System packages

The build needs a C++20 compiler, CMake ≥ 3.25, the Vulkan SDK, `glslc` (from shaderc), and the
C++ libraries `fmt`, `glm`, and `VulkanMemoryAllocator`. On Arch Linux:

```bash
sudo pacman -S base-devel cmake vulkan-headers vulkan-icd-loader shaderc \
               vulkan-memory-allocator fmt glm pybind11 python
```

On Debian/Ubuntu the equivalents are `build-essential cmake libvulkan-dev glslc
libvulkan-memory-allocator-dev libfmt-dev libglm-dev pybind11-dev python3-dev`.

### Python packages

For the examples:

```bash
pip install numpy pillow matplotlib scipy
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSIGVIS_PYTHON=ON
cmake --build build --target pyrendering -j
```

The built extension lands at `build/pyrendering/pyrendering.cpython-*.so`. Shaders are compiled
by `glslc` at CMake configure time and embedded directly into the binary; no runtime shader
files are needed.

To build with Vulkan validation layers enabled (for development):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DSIGVIS_VALIDATION=ON -DSIGVIS_PYTHON=ON
cmake --build build --target pyrendering -j
```

## Using `pyrendering` from Python

The extension is not installed to `site-packages`; add the build directory to `sys.path`:

```python
import sys
sys.path.insert(0, "build/pyrendering")
import pyrendering as pr

pr.init(validation_layers=False)      # call once before creating any object
print(pr.capabilities())              # device name + mesh shader limits

# --- Low-level accumulation renderer ---
hist = pr.Histogram(width=3840, height=2160)
hist.clear()

# 2-D traces: shape (n_traces, trace_length, 2) float32, [x, y] per sample
hist.draw(samples, x_range=(-1.0, 1.0), y_range=(-2.0, 2.0), line_width=1e-3)

# 1-D waveform: shape (N,) float32 with uniform x in [0, 1]
hist.draw_waveform(signal, x_range=(0.0, 1.0), y_range=(-1.5, 1.5), line_width=3e-3)

data = hist.download()                # (H, W) float32 — blocks until GPU is done

# --- High-level renderer: histogram + log/colormap composite ---
renderer = pr.EyeDiagramRenderer(width=1920, height=1080)
renderer.set_samples(samples)
rgba = renderer.render(x_range=(-1, 1), y_range=(-1, 1),
                       line_width=1.0, max_intensity=1.0)   # (H, W, 4) uint8

pr.shutdown()
```

`draw()` is asynchronous: it copies into a persistent staging buffer and returns immediately
while the GPU works in the background. `download()` blocks until all pending work is complete.

## Examples

Self-contained examples live in `examples/`:

- `render_eye.py` — PAM4 eye diagram and batched sinusoids over millions of traces
- `render_waveform.py` — long 1-D phosphor-style waveform

Run them from the repo root after the build completes:

```bash
python examples/render_eye.py
python examples/render_waveform.py
```

Each script saves a PNG next to itself.

For interactive exploration there are also Jupyter notebooks demonstrating typical workflows:

- [`examples/rendering_eye_diagrams.ipynb`](examples/rendering_eye_diagrams.ipynb) — building
  PAM4 eye diagrams from raised-cosine-shaped symbol streams
- [`examples/rendering_tests.ipynb`](examples/rendering_tests.ipynb) — assorted rendering
  experiments (waveforms, histograms, colormap tuning)

Launch them with `jupyter lab examples/` from the repo root.

## Project layout

```
sigvis/
├── vulkan_engine/    # vke::  — RAII Vulkan wrappers (instance, device, buffers, …)
├── plot_renderer/    # plot:: — Histogram, EyeDiagramRenderer, shaders
├── pyrendering/      # pybind11 bindings → pyrendering Python module
├── examples/         # Python usage examples
└── cmake/            # Dependencies.cmake, Shaders.cmake
```
