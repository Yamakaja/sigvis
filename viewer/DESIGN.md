# Live high-speed acquisition viewer — design

Integrating a Red Pitaya streaming ADC (`plstream`) as a live source for the
phosphor scope, and reworking the render path so a single architecture serves
both regimes the hardware produces and both display modes.

## The two acquisition regimes

The link is 1 Gbit/s. The Red Pitaya offers up to 125 MS/s × 2 ch × 16 bit ≈
4 Gbit/s, so the regime depends on decimation:

- **Full rate (decim 0, 125 MS/s, ~4 Gbit/s offered).** We receive at most ~25%.
  Each delivered packet is internally contiguous, but large *known* gaps fall
  between packets (every frame header carries `seq` and a 125 MHz `timestamp`, so
  we always know exactly where a packet sits and how many samples are missing).
- **Decimated (e.g. decim 4, 7.8 MS/s, ~250 Mbit/s).** Fits the link with margin:
  the full continuous sample stream arrives, packetized, with rare dropped frames.

## Core principle: transport ≠ display unit

The packet is a *transport* unit. The display units are roll *columns* and
triggered *traces*, defined by a timebase and a trigger — never by packet
boundaries. So the spine is one layer that reassembles packets into a single
**gap-aware sample timeline**, which both render modes consume. Because `seq`
gives every sample an exact absolute index, we reconstruct a positionally-correct
timeline with explicit holes; we never guess.

The single rule shared by every render path:

> Form a line segment only between two **adjacent, present** samples. Never bridge
> a hole. Let persistence (trigger) or per-column accumulation (roll) integrate
> coverage over time.

Decimated data simply has near-complete coverage per capture; full-rate is sparse
per capture but dense in aggregate. Same code.

## Architecture

```
ISource ──drain(Frame[])──► StreamScope.ingest()
                                 │  writes into…
                          ┌──────▼──────────────────────────┐
                          │  SampleRing (shared, GPU)        │
                          │  • float samples[C]              │
                          │  • int64 slot_seq[F]  (gen tags) │
                          └──────┬────────────────┬──────────┘
                          roll   │                │  trigger
                         (compute reduce,   (CPU scan + holdoff harvest,
                          tag-skip holes)    spc-keyed box/line draw,
                                             sub-sample align, clip to view)
                            strip image          persistence accumulator
                                 └──────── log + Turbo composite ───────┘
```

Frames are ingested **once**, mode-independently, so switching trigger↔roll is
seamless and both modes see the same buffered history.

### Frame — the source contract

Sources emit sequence-numbered, timestamped frames of one channel, not bare
floats, so the ring can reassemble:

```cpp
struct Frame {
    int64_t  seq;               // absolute frame index; gaps = dropped frames
    int64_t  timestamp;         // ticks at first sample (RP adc_clk; else synthesized)
    uint32_t samples_per_frame; // S, fixed within a session
    std::vector<float> samples; // one selected channel, normalized
};
```

`PlstreamSource` fills these from the acq frame header. Audio/synthetic sources
fill `seq` = a running counter with no gaps and a synthetic timestamp, so they
flow through the identical ring/trigger/roll path — trigger mode then works for
them too, which is useful for development without hardware.

### SampleRing — generation-tagged frame slots

Writing a sentinel into every dropped slot would cost 4× the received bandwidth
at full rate (75% holes), so validity is tracked at **frame granularity** (drops
are always whole frames):

- Ring = `F` frame-slots × `S` samples/frame, `C = F·S`. 100 ms @ 125 MS/s ≈
  12.5 M samples ≈ 50 MB.
- A received frame `seq` writes its `S` samples to slot `seq mod F` and stamps
  `slot_seq[seq mod F] = seq`. **Only received frames are uploaded.**
- CPU tracks `slot_seq[s]` = last frame seq written to physical slot `s`. Given the
  current head frame `H`, slot `s` is **present** iff it holds the frame the live
  window expects there (the unique `f ∈ [H−F, H)` with `f mod F == s`) — i.e. its
  successor wasn't dropped before overwriting it. A stale (pre-window) slot is
  detected without clearing.

To keep the GPU **int64-free** (per the time-representation rule), the CPU does all
the rebasing and uploads a small **`uint32 present[F]`** map indexed by physical
slot; the roll reduce shader just reads `present[offset / S]` — no 64-bit math on
the GPU. The trigger path needs no GPU presence map (the CPU builds run lists from
`slot_seq` directly). The sample ring is device-local with per-frame host staging +
recorded copies (matching `roll_scope`'s proven, barrier-ordered upload, which also
sidesteps writing slots that in-flight reads might still touch); `present[]` rides
the same staged upload.

A change in sample rate / frame size (decim change, source switch, channel
switch) resets the ring.

## Time representation

- **Master clock = int64 absolute sample index** (`seq · S + offset`). Exact,
  monotonic (~2340 yr at 125 MS/s), and *is* the ring address (`mod C`) and tag
  key (`/ S`). All global bookkeeping is integer.
- **Sub-sample = a separate `float frac ∈ [0,1)`**, stored only on trigger events
  that need it. Bounded magnitude ⇒ no growing-float precision loss. Not packed
  into the int64 — "which sample" and "where between" stay orthogonal.
- **Rebase before float.** Never hand an absolute int64 to float/GPU math. Diff
  in int64 first (`absolute_index − trigger.index` → small, fits int32), then cast
  to float and subtract `frac`. The GPU only ever sees small trigger-relative
  offsets (large-world origin-rebasing).
- **Real-time display.** Relative readouts = int64 diff ÷ `fs` (exact). Absolute
  "time since arm" = `double(index) / fs` (exact to ~834 days at full rate).
- The hardware `timestamp` is a *parallel* reference (corroborate drops, drive the
  equivalent-time fold), not the indexing clock. Sample index = identity;
  timestamp = physical time/phase.

## Zoom / LOD — one ratio governs everything

```
spc = samples per pixel column = fs · L / width_px      (L = timebase)
```

- **spc > 2 — zoomed out, dense.** Integrate within the column: per-column
  envelope + 1/dy density. Periods are sub-pixel, so triggering only deepens an
  envelope — sub-sample trigger accuracy is unnecessary here (`frac = 0`). Roll
  uses the compute reduce; trigger gets the same look from `waveform_renderer.mesh`
  box mode with many overlaid captures accumulating density.
- **spc ≤ 2 — zoomed in, sparse.** Draw actual line segments between the few
  visible samples (optional Lagrange interpolation for the inter-sample curve).
  Triggering is high-value (resolves cycles, fills fast, overlays neighbors).
  Only emit geometry for samples in the visible x-range — clip each capture's run
  list to `[t−pre+x_min·L, t−pre+x_max·L]` so zoomed-in cost is independent of L.

The crossover is automatic from `spc`, fixed at **spc = 2** with a short blend so
the transition doesn't pop. Both modes already exist inside the pipelines (box vs
line); the LOD switch just drives line width / mode and the view clip.

Trigger's dense regime uses **mesh box-mode + additive accumulation**, not roll's
reduce: roll reduces one contiguous run per column, whereas trigger overlays many
aligned, possibly overlapping captures at arbitrary ring offsets — exactly what
additive mesh accumulation (the eye-diagram path) is built for. Roll and trigger
share the *concept* (1/dy density, log + Turbo composite), not the pass.

## Trigger model

- **Not acquisition-limited.** Unlike a DSO with capture dead-time, the full
  stream lives in the ring, so each frame we harvest *every* qualifying edge in the
  buffered history at once and overlay them — persistence fills as fast as the data
  allows, at any zoom.
- **Holdoff = L by default**, the lever that makes trigger rate scale as 1/L:
  zoom in → shorter L → more triggers/s → more neighboring periods overlaid; zoom
  out → few non-redundant windows. Adjustable down toward one period (→ dense
  eye / cycle-overlay). Exposed in real time units.
- **Trigger is centered** (pre = L/2).
- **Cap** captures-per-frame (set high) so extreme zoom-out doesn't harvest
  thousands of near-identical full-L windows for an envelope.
- **Detection on the CPU** in ingest (a cheap 1-D scan at these rates): level +
  slope + hysteresis + holdoff → `{int64 index, float frac}` events. The GPU draws
  straight from the ring given each event's run list — sample data uploaded once,
  never copied per capture. Move to a compute prepass only if ever needed.
- **Equivalent-time** for full-rate zoom-out: sub-sample δ alignment is the core;
  the advanced `timestamp mod T` phase fold is the same path with an alternate
  x-map, added after the basics.

## Viewer / UI

- `Frame`-based `ISource`; the two existing sources adapt trivially.
- `StreamScope` owns the shared ring + both paths; `main.cpp` pushes frames once
  and calls `render(mode, …)`.
- ImGui: source = {synthetic, audio, **redpitaya**} with channel A/B selector and
  connection fields (host, ctrl port 7654, local UDP port, decim); trigger controls
  (level, slope, hysteresis, holdoff, timebase L, pre-trigger, equivalent-time).

## Build order

1. `Frame` + adapt synthetic/audio sources (keep existing rendering working).
2. `SampleRing` (tagged slots + ingest).
3. Roll on the shared ring (tag-skip holes).
4. Trigger engine (CPU scan + holdoff harvest → spc-keyed mesh draw, clipped).
5. `PlstreamSource` (plstreamc.h control + UDP receive/reassemble).
6. UI: source selector, channel, connection, trigger controls.
