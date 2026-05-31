#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vke/vke.hpp>

namespace plot {

// Per-frame parameters for the realtime phosphor scope.
struct ScopeParams {
    float x_min = 0.0f;   // visible record window in normalized sample index [0,1]
    float x_max = 1.0f;
    float y_min = -1.5f;  // amplitude window (signal units)
    float y_max = 1.5f;

    float line_width    = 2e-3f; // isometric units (~ line_width * height/2 px)
    float decay_alpha   = 0.92f; // per-frame accumulator multiply in [0,1] (1 = no fade)
    float max_intensity = 1.0f;  // composite log scale
    float min_weight    = 0.0f;  // box-mode intensity floor [0,1]
    float black_level   = 0.05f; // raw values <= this map to black (~exp(-3): single trace clears by ~3 tau)
};

// Free-running phosphor oscilloscope renderer. Owns a persistent R32_SFLOAT
// accumulator that survives between frames; each render() multiplies it by
// decay_alpha (phosphor fade), additively draws any pushed chunks (traces), and
// composites the result (log + Turbo) into a caller-supplied target image (e.g. a
// swapchain image). All traces overlay across the full width (free-run trigger).
class WaveformScope {
public:
    WaveformScope(vke::Context& ctx, uint32_t width, uint32_t height,
                  VkFormat target_format, uint32_t frames_in_flight = 2);
    ~WaveformScope();

    WaveformScope(const WaveformScope&)            = delete;
    WaveformScope& operator=(const WaveformScope&) = delete;
    WaveformScope(WaveformScope&&) noexcept;
    WaveformScope& operator=(WaveformScope&&) noexcept;

    // Recreate the accumulator at a new size and reset persistence. Caller must
    // ensure no in-flight work references the old accumulator (vkDeviceWaitIdle).
    void resize(uint32_t width, uint32_t height);

    // Reset phosphor persistence (clears the accumulator on the next render()).
    void reset_persistence();

    // Queue a trace for the next render(). All chunks must share one length.
    // Excess chunks beyond the per-frame cap drop the oldest.
    void push_chunk(std::span<const float> samples);
    size_t pending() const;

    // Record decay + traces + composite into `cmd`, compositing into `target`.
    // Adds its own barriers; leaves `target` in COLOR_ATTACHMENT_OPTIMAL (ready for
    // a subsequent LOAD pass such as an ImGui overlay).
    void render(vke::CommandBuffer& cmd, vke::Image& target, const ScopeParams& params);

    uint32_t width()  const noexcept;
    uint32_t height() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace plot
