# sigvis

GPU-accelerated signal visualization library. Renders eye diagrams, density histograms, and long 1-D waveforms using Vulkan mesh shaders with additive accumulation (phosphor-oscilloscope style). Headless by design; primary consumer is a Python extension (`pyrendering`).

## Directory Structure

```
sigvis/
├── CMakeLists.txt
├── cmake/
│   ├── Dependencies.cmake      # FetchContent: VMA, pybind11, GLM, fmt
│   └── Shaders.cmake           # compile_shaders() macro → embeds SPIR-V as C++ headers
├── vulkan_engine/              # vke:: — thin RAII Vulkan wrappers
│   ├── include/vke/
│   └── src/
├── plot_renderer/              # plot:: — rendering logic
│   ├── include/plot/
│   ├── src/
│   └── shaders/
├── pyrendering/                # Python extension (pybind11)
│   └── src/bindings.cpp
└── examples/
    ├── render_eye.py           # PAM4 eye diagram + batched sinusoids
    ├── render_waveform.py      # long 1-D waveform example
    └── sigutils.py             # DSP helpers (rcf, cconv, …)
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSIGVIS_PYTHON=ON
cmake --build build --target pyrendering
# output: build/pyrendering/pyrendering.cpython-*.so
```

Debug + validation layers:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DSIGVIS_VALIDATION=ON -DSIGVIS_PYTHON=ON
```

Shaders are compiled by `glslc` at CMake configure time and embedded as `spv::*` arrays in
`build/plot_renderer/generated/shaders/*_spv.hpp`. Adding a new shader requires adding it to
`plot_renderer/CMakeLists.txt`'s `compile_shaders()` call.

## Python API

```python
import sys; sys.path.insert(0, "build/pyrendering")
import pyrendering as pr

pr.init(validation_layers=False)   # must be called before creating any object
caps = pr.capabilities()           # dict with mesh shader limits, device name

# --- Histogram (accumulation renderer) ---
hist = pr.Histogram(width=3840, height=2160)
hist.clear()

# Eye/trace data: shape (n_traces, trace_length, 2) float32 [x, y]
hist.draw(samples, x_range=(-1.0, 1.0), y_range=(-2.0, 2.0), line_width=1e-3)

# 1-D waveform: shape (N,) float32 — x is uniform 0→1
hist.draw_waveform(signal, x_range=(0.0, 1.0), y_range=(-1.5, 1.5), line_width=3e-3)

data = hist.download()   # (H, W) float32 — blocks until GPU done
del hist

# --- High-level renderer (histogram + composite in one call) ---
renderer = pr.EyeDiagramRenderer(width=1920, height=1080)
renderer.set_samples(samples)   # shape (n_traces, trace_length, 2)
rgba = renderer.render(x_range=(-1,1), y_range=(-1,1), line_width=1.0, max_intensity=1.0)
# returns (H, W, 4) uint8

pr.shutdown()  # safe to call while Histogram/EyeDiagramRenderer objects still exist
```

`draw()` is **asynchronous**: memcpy to a persistent staging buffer returns immediately; the GPU
runs in the background. The next `draw()` call will stall if the previous GPU work hasn't finished
consuming the staging buffer. `download()` and `flush()` block until all GPU work is complete.

## Architecture

### `vulkan_engine` (`vke::`)

RAII wrappers — every Vulkan handle is owned by exactly one C++ object. Move-only except
`DescriptorSet` (copyable, non-owning handle) and `Sampler` (shared_ptr internals).

`vke::Context` is the sole factory for all GPU objects. It owns the VkInstance, VkDevice,
VmaAllocator, per-queue command pools, and a fence pool used by async submission.

**Submission model:**
- `ctx.submit_and_wait(cmd)` — blocking, simple
- `handle = ctx.submit(cmd)` — async; `SubmitHandle` owns the fence + command buffer
- `ctx.wait(handle)` — waits and **returns fence to pool** (must use this, not `handle.wait()`)

Always use `ctx.wait(handle)` for async handles — `handle.wait()` does not recycle the fence
and will cause fence leaks on device destruction.

### `plot_renderer` (`plot::`)

**`plot::Histogram`** — the core accumulator. Owns an `R32_SFLOAT` image that acts as the
accumulation target. Two draw paths share the same image:

- `draw()` — renders 2-D traces (vec2 x/y pairs) using `lines_renderer.mesh`
- `draw_waveform()` — renders a 1-D signal (float y values, implicit uniform x) using
  `waveform_renderer.mesh`

Both pipelines use the same fragment shader (`lines_renderer.frag`) and additive blending
(`VK_BLEND_OP_ADD`, `R` channel only). `LOAD_OP_LOAD` is used so accumulation is incremental.
Call `clear()` explicitly between renders.

**`plot::EyeDiagramRenderer`** — wraps `Histogram` with a second composite pass
(`histogram_sample.vert/frag`) that applies log scale + Turbo colormap → RGBA8 output.

### Shaders

| Shader | Purpose |
|---|---|
| `lines_renderer.mesh` | 2-D trace geometry: reads `Vertex[]` (vec2), computes miter joints |
| `waveform_renderer.mesh` | 1-D waveform geometry: reads `float[]`, derives x from index |
| `lines_renderer.frag` | Shared fragment: soft-edge intensity `1 - v^4` |
| `histogram_sample.vert/frag` | Fullscreen triangle, log scale + Turbo colormap |

Both mesh shaders use `WORKGROUP_SIZE=32` (32 segments per workgroup, 8 verts / 7 tris each).
The dispatch limit is 65535 workgroups in X. For large workloads, C++ batches the dispatch and
increments `trace_offset` / `first_segment` in the push constants between batches.

**Coordinate conventions:**
- `lines_renderer`: world space with `center`/`zoom`; isometric via `viewport_ratio`
- `waveform_renderer`: signal space with `x_range`/`y_range`; same isometric `viewport_ratio`
- `line_width` is in isometric units (equal pixel density on both axes regardless of aspect ratio)
  ≈ `line_width * height/2` pixels

### Descriptor Layout

Both pipelines share the same `DescriptorLayout`: binding 0 = `StorageBuffer`, mesh stage.
`Histogram::Impl` allocates two descriptor sets from this layout — one bound to
`sample_buffer` (vec2 traces), one bound to `waveform_device` (float waveform).

## Key Push Constant Structs (C++)

```cpp
// lines_renderer.mesh — sizeof == 40
struct HistogramPC {
    glm::vec2 center, zoom, viewport_ratio;
    float line_width; uint32_t n_samples, n_samples_per_trace, trace_offset;
};

// waveform_renderer.mesh — sizeof == 40
struct WaveformPC {
    glm::vec2 x_range, y_range, viewport_ratio;
    float line_width; uint32_t n_samples, first_segment, n_segments;
};
```

## DSP Helpers (`examples/sigutils.py`)

Key functions used by examples:
- `rcf(t, f, beta)` — raised cosine filter (time + freq domain)
- `cconv(a, b)` — circular convolution via FFT (`max(len(a), len(b))` point)

For batch PAM4 generation, precompute the filter with `make_rcf_filter(sps, beta, n_syms=8)` and
use `_overlap_save_circular(x, h)` (vectorized, no Python loop) instead of `sigutils.cconv`.
The overlap-save implementation wraps the first block's overlap from the signal tail so there
are no startup transients when reshaping into eye diagram traces.

## Gotchas

- **Fence lifecycle**: always `ctx.wait(handle)` not `handle.wait()` — the latter skips fence
  recycling and leaks fences visible as `vkDestroyDevice` validation warnings.
- **Shader limits**: `maxMeshWorkGroupCount[0]` = 65535. Check via `pr.capabilities()`.
  Both `draw()` and `draw_waveform()` batch automatically; no action needed from callers.
- **Validation layers**: Khronos validation layer (`libVkLayer_khronos_validation.so`) crashes
  with a segfault on very large workloads (observed at ~1M traces). Use
  `pr.init(validation_layers=False)` for production runs.
- **`draw()` is async**: the caller's numpy array is safe to modify immediately after `draw()`
  returns. Stalls occur only if the staging buffer is still in use — i.e., if `draw()` is called
  again before the GPU has consumed the previous staging upload.
- **Colormap**: `apply_colormap()` in the examples uses inferno on a log scale.
  `log_min` is an **absolute** histogram value (not relative). Values below `log_min` map to
  black. Typical values: 0.05–0.1.
