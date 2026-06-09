#pragma once
#include <cstdint>
#include <memory>
#include <vke/vke.hpp>

#include <plot/sample_ring.hpp>
#include <plot/roll_scope.hpp>   // RollParams

namespace plot {

enum class TriggerSlope { Rising, Falling };

enum class SpectrumWindow { Hann, BlackmanHarris, Flattop, Rect };

// Per-frame parameters for the density spectrum. Each FFT block of `fft_size` samples
// is windowed, transformed, converted to dBFS (0 dBFS = a full-scale sine), and drawn
// as one trace into the shared phosphor accumulator — so overlaying many blocks over
// time builds a (frequency × dBFS) density display. No separate averaging; the
// persistence decay provides the temporal integration.
struct SpectrumParams {
    uint32_t       fft_size = 4096;          // power of two
    SpectrumWindow window   = SpectrumWindow::Hann;

    float x_min = 0.0f, x_max = 1.0f;        // visible fraction of [0, fs/2]
    float db_floor = -120.0f, db_top = 6.0f; // dBFS axis (maps to y)

    float line_width    = 2e-3f;
    float decay_alpha   = 0.9f;              // per-frame phosphor multiply [0,1]
    float max_intensity = 1.0f;
    float min_weight    = 0.0f;
    float black_level   = 0.05f;
};

// Per-frame parameters for the triggered persistence display.
struct TriggerParams {
    float y_min = -1.5f;          // amplitude window (signal units)
    float y_max =  1.5f;
    float window_seconds = 1e-3f; // timebase L (centered on the trigger)
    float x_min = 0.0f;           // visible window fraction [0,1] (pan/zoom in x)
    float x_max = 1.0f;

    float        level      = 0.0f;             // trigger level (signal units)
    TriggerSlope slope      = TriggerSlope::Rising;
    float        hysteresis = 0.02f;            // signal units, anti-noise band
    float        holdoff_seconds = 0.0f;        // 0 ⇒ one window (L); else min spacing
    bool         equivalent_time = true;        // sub-sample crossing alignment
    float        pre_frac   = 0.5f;             // trigger position in the window [0,1]
    uint32_t     max_captures = 4096;           // harvested-per-frame cap

    // Auto trigger mode: if no trigger occurs within auto_timeout_seconds, draw the
    // waveform anyway (untriggered free-run sweep) so the signal stays visible — e.g.
    // while setting up the level. Normal mode (false) shows only triggered captures.
    bool         auto_mode = true;
    float        auto_timeout_seconds = 0.1f;

    float line_width    = 2e-3f; // isometric units
    float decay_alpha   = 0.9f;  // per-frame phosphor multiply [0,1]
    float max_intensity = 1.0f;  // composite log scale
    float min_weight    = 0.0f;  // box-mode intensity floor [0,1]
    float black_level   = 0.05f; // composite black clip
};

// Unified live scope: owns the shared SampleRing and the render paths that read it.
// Frames are ingested once (mode-independent); the caller renders whichever mode is
// active each frame. The roll path lives here now; the trigger path lands next and
// will read the same ring.
//
// Both render paths leave `target` in COLOR_ATTACHMENT_OPTIMAL, ready for a LOAD pass
// (e.g. an ImGui overlay).
class StreamScope {
public:
    StreamScope(vke::Context& ctx, uint32_t width, uint32_t height,
                VkFormat target_format, float sample_rate_hz,
                float max_window_seconds = 0.1f, uint32_t frames_in_flight = 2);
    ~StreamScope();

    StreamScope(const StreamScope&)            = delete;
    StreamScope& operator=(const StreamScope&) = delete;
    StreamScope(StreamScope&&) noexcept;
    StreamScope& operator=(StreamScope&&) noexcept;

    void resize(uint32_t width, uint32_t height);

    // Change the time axis (Hz). Discards history (resets the ring).
    void  set_sample_rate(float sample_rate_hz);
    float sample_rate() const noexcept;

    // Drop all history (clears the strip / persistence on the next render).
    void reset();

    // Queue one frame onto the shared timeline (gap-aware via seq).
    void ingest(const RingFrame& f);

    // Roll render: upload the ring, reduce newly-available columns into the circular
    // strip, composite (log + Turbo) into `target`.
    void render_roll(vke::CommandBuffer& cmd, vke::Image& target, const RollParams& params);

    // Trigger render: upload the ring, harvest new trigger events (CPU scan), decay +
    // additively draw their captures into the persistence accumulator, composite into
    // `target`. Clears + re-harvests the buffered history when alignment params change.
    void render_trigger(vke::CommandBuffer& cmd, vke::Image& target, const TriggerParams& params);

    // Density spectrum: FFT newly-available contiguous blocks, draw each as a dBFS
    // trace into the persistence accumulator, composite into `target`. Frames with no
    // fft_size-contiguous block (e.g. sparse full-rate data with a large FFT) are
    // skipped — reduce the FFT size in that case.
    void render_spectrum(vke::CommandBuffer& cmd, vke::Image& target, const SpectrumParams& params);

    // Clear phosphor persistence and re-harvest from the buffered history next frame.
    void reset_persistence();
    size_t captures_last_frame() const noexcept;

    // Suggested trigger setup from analyzing recent buffered samples (Autoset). The
    // caller applies whichever fields are valid to its view/trigger controls.
    struct AutosetResult {
        bool  ok = false;            // false ⇒ not enough data; ignore the result
        bool  has_timebase = false;  // window_seconds valid (false when coverage is sparse)
        float y_min = 0.f, y_max = 0.f;
        float level = 0.f, hysteresis = 0.f;
        float window_seconds = 0.f;
    };
    // Analyze recent buffered samples → robust vertical extent, 50% level, noise-based
    // hysteresis, and (when coverage is dense) a timebase of a few periods. O(samples).
    AutosetResult autoset() const;

    // ---- UI / status ----
    uint32_t width()  const noexcept;
    uint32_t height() const noexcept;
    uint32_t samples_per_stripe() const noexcept; // K for the current window
    float    buffered_seconds()   const noexcept;
    float    capacity_seconds()   const noexcept;
    float    shown_seconds()      const noexcept;
    uint64_t dropped_frames()     const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace plot
