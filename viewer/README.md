# sigvis viewer

A realtime, GPU-accelerated phosphor-oscilloscope app built on `plot::` and `vke::`. It
renders a live sample stream two ways — a **trigger** (persistence) display and a **roll**
(strip-chart) display — both fed from one shared, gap-aware sample timeline. Sources include a
synthetic generator, live audio capture, and a **Red Pitaya** streaming ADC over the network.

For the architecture (the shared `SampleRing`, `StreamScope`, the trigger engine, the time
model), see [DESIGN.md](DESIGN.md). This file is the how-to-use.

## Build

The viewer is off by default. Enable it with `-DSIGVIS_VIEWER=ON`:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSIGVIS_VIEWER=ON
cmake --build build --target viewer -j
# binary: build/viewer/viewer
```

### Extra prerequisites (beyond the core build in the top-level README)

- A Vulkan 1.3 driver with **mesh shader support** (`VK_EXT_mesh_shader`).
- **GLFW 3** — `find_package(glfw3)` (Arch: `glfw`; Debian/Ubuntu: `libglfw3-dev`).
- **Dear ImGui**, vendored as source. CMake looks in `$HOME/rendering/imgui` by default;
  point it elsewhere with `-DIMGUI_DIR=/path/to/imgui` (must contain `imgui.cpp` and the
  `backends/` GLFW+Vulkan implementations).
- **PulseAudio simple API** for audio capture — `libpulse-simple` (Arch: `libpulse`;
  Debian/Ubuntu: `libpulse-dev`). Works transparently against a PipeWire pulse server.

## Run

```bash
./build/viewer/viewer            # opens a window; runs until closed
```

Flags:

- `--frames N` — present `N` frames then exit (handy for a quick smoke test).
- `--no-vsync` — uncap the present rate.

## Display modes

Pick **trigger**, **roll**, or **spectrum** at the top of the control panel. Frames are ingested
into the shared ring once per frame regardless of mode, so switching is seamless and all modes
see the same buffered history.

- **trigger** — a phosphor persistence display. New trigger events are detected (level + slope +
  hysteresis + holdoff), aligned (with sub-sample/equivalent-time precision), and overlaid into
  a decaying accumulator. The horizontal axis is time relative to the trigger.
- **roll** — a strip chart. The newest samples enter at the right and history scrolls left; each
  pixel column reduces its samples to an amplitude envelope + density. The horizontal axis is
  wall-clock time.
- **spectrum** — a density spectrum (dBFS vs. frequency). Each FFT block is windowed, transformed
  (configurable size, power of two), converted to dBFS (0 dBFS = full-scale sine), and drawn into
  the same phosphor accumulator, so the (frequency × dBFS) density builds up over time. Pan/zoom
  the frequency (x) and dBFS (y) axes; a cursor readout shows frequency + dBFS under the pointer.
  At sparse full-rate Red Pitaya, reduce the FFT size if no contiguous block of that length
  exists (the spectrum simply won't update otherwise).

Both apply a log + Turbo tonemap, so brightness encodes how often the signal passes through each
pixel (modulation, noise distribution, multi-level structure).

## Mouse & keyboard

| Input | Action |
|---|---|
| **Left-drag** (on the plot) | Pan |
| **Scroll** | Zoom X about the cursor |
| **Shift + Scroll** | Zoom Y about the cursor |
| Left-drag on a trigger handle | Move that handle (see below) — panning is suppressed |

Panning/zooming the view re-aligns the trigger display live.

## On-screen trigger controls (trigger mode)

Drawn over the waveform; their precision scales with zoom (unlike the panel sliders):

- **Level line** — full-width horizontal line; grab it anywhere to drag the trigger level. The
  right-edge handle shows the slope (`rise`/`fall`) and the numeric level.
- **Trigger marker** — full-height vertical line at the trigger position; drag it horizontally to
  move the trigger point within the window (the `trig pos` slider mirrors it). The crosshair at
  its intersection with the level line sits on the trigger convergence node.
- **Hysteresis band** — the shaded ±hysteresis dead zone; grab either left-edge handle to drag
  the band width.

Off-screen level/marker are clamped to the edge with an arrow so they stay grabbable.

## Control panel

**Trigger:** timebase (window seconds), trig level, slope, hysteresis, holdoff (0 = one window),
equivalent-time, trig pos, persistence (seconds), line width, max intensity, black level, box
floor. A readout shows the window length in samples and captures harvested per frame.

- **autoset** — analyzes a few recent buffers and sets the vertical scale, trigger level (50%),
  hysteresis (from the noise floor), and — when the signal is continuous — a timebase showing a
  few periods, then centers the trigger and resets the x pan. One-shot; a fast way onto an
  unknown signal.
- **auto** — auto trigger mode (default on): if no trigger occurs within ~100 ms, the waveform is
  drawn anyway (untriggered free-run) so the signal stays visible while you set the level. Turn
  it off for Normal mode (only triggered captures shown).

**Roll:** window (seconds), vertical line width, coverage floor, density (phosphor↔additive),
Lagrange interpolation factor, max intensity, black level.

**Spectrum:** FFT size (2ⁿ), window (Hann / Blackman-Harris / Flat-top / Rect), dBFS top/floor,
persistence (seconds), line width, max intensity, black level, box floor. A readout shows the FFT
size, resolution bandwidth, and span; another shows the cursor's frequency + dBFS.

## Sources

Select **synthetic**, **audio**, or **redpitaya**. Exactly one is active; the rate is synced to
the renderer automatically.

- **synthetic** — a continuous AM sine with adjustable frequency, amplitude, noise, phase
  jitter, and AM depth/rate. Useful for exercising both modes without hardware.
- **audio** — mono capture via PulseAudio/PipeWire (default source). To visualize playback,
  capture a sink's `.monitor` (e.g. `PULSE_SOURCE=…​.monitor ./build/viewer/viewer`).
- **redpitaya** — live streaming ADC via `plstreamd`. Enter the board's **host**, control port
  (default 7654), and your local **UDP port** (default 5000), choose **channel A/B**, and press
  **connect** (or **reconnect** after changing fields). The daemon streams to this host; the rate
  is derived from the stream's decimation once the first frame arrives. Connecting is bounded and
  cancellable, so a wrong address fails in a few seconds rather than hanging.

## Status readout

- **received / renderer drops** — frames handed to the renderer / discarded because the render
  thread fell behind (or while paused).
- **fabric drops** (Red Pitaya) — frames the FPGA dropped before transmit (from the frame
  header).
- **net/kernel drops** (Red Pitaya) — frames transmitted but lost in the network or the kernel
  UDP buffer.
- **timeline gaps** — total holes on the reassembled timeline (`= fabric + net/kernel +
  renderer`); these are the gaps actually drawn.

See the drop-stat notes in [DESIGN.md](DESIGN.md) for how to read these diagnostically.
