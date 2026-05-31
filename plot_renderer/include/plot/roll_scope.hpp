#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vke/vke.hpp>

namespace plot {

// Per-frame parameters for the rolling (strip-chart) scope.
struct RollParams {
    float y_min = -1.5f;          // amplitude window (signal units)
    float y_max =  1.5f;
    float window_seconds = 2.0f;  // visible time span (snapped to an integer samples/stripe)
    float line_width_px  = 1.5f;  // vertical line width in pixels (splats across amplitude bins)
    float max_intensity  = 1.0f;  // composite log scale
    float black_level    = 0.05f; // raw bin values <= this map to black
};

// Rolling high-speed waveform display ("strip chart"). New samples enter at the
// right edge and history scrolls left. The horizontal axis is wall-clock time; each
// pixel column ("stripe") is a fixed integer K samples, reduced on the GPU into a
// vertical amplitude histogram (density-graded, log + Turbo). Raw samples live in a
// circular GPU buffer (only new samples are uploaded each frame; old ones are
// overwritten). Columns are written once into a horizontally-circular strip image
// and scrolled by an integer offset — no per-frame copies, no subpixel jitter.
//
// Coexists with WaveformScope: shares the vke primitives and the log/Turbo look,
// owns its own pipelines and resources.
class RollScope {
public:
    RollScope(vke::Context& ctx, uint32_t width, uint32_t height,
              VkFormat target_format, float sample_rate_hz,
              float max_window_seconds = 10.0f, uint32_t frames_in_flight = 2);
    ~RollScope();

    RollScope(const RollScope&)            = delete;
    RollScope& operator=(const RollScope&) = delete;
    RollScope(RollScope&&) noexcept;
    RollScope& operator=(RollScope&&) noexcept;

    // Recreate the strip image at a new size; triggers a replay from the sample ring.
    // Caller must ensure no in-flight work references the old strip (vkDeviceWaitIdle).
    void resize(uint32_t width, uint32_t height);

    // Queue samples for upload into the ring on the next render().
    void push_chunk(std::span<const float> samples);
    size_t pending() const;

    // Record upload + compute reduction + composite into `cmd`, compositing into
    // `target`. Leaves `target` in COLOR_ATTACHMENT_OPTIMAL (ready for an overlay).
    void render(vke::CommandBuffer& cmd, vke::Image& target, const RollParams& params);

    // Effective samples-per-stripe (K) for the current window — exposed for the UI
    // so it can show the exact (quantized) window length.
    uint32_t samples_per_stripe() const noexcept;

    uint32_t width()  const noexcept;
    uint32_t height() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace plot
