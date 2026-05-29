#pragma once
#include <memory>
#include <span>
#include <vke/vke.hpp>
#include "types.hpp"

namespace plot {

class Histogram {
public:
    Histogram() noexcept;
    explicit Histogram(vke::Context& ctx, uint32_t width, uint32_t height);
    ~Histogram();

    Histogram(const Histogram&)            = delete;
    Histogram& operator=(const Histogram&) = delete;
    Histogram(Histogram&&) noexcept;
    Histogram& operator=(Histogram&&) noexcept;

    // Wait for any in-flight draw() to complete. Called implicitly by download().
    void flush();

    // Zero the histogram image on the GPU.
    void clear();

    // Accumulate samples into the histogram. May be called repeatedly to build
    // up the histogram incrementally. Does NOT clear first — call clear() explicitly.
    void draw(std::span<const Sample> samples, uint32_t trace_length,
              const RenderParams& params);

    // Accumulate a single 1-D waveform (float32 amplitude values, uniform x spacing)
    // into the histogram. x_range selects which portion of the signal to render and
    // reduces the number of GPU segments accordingly.
    void draw_waveform(std::span<const float> samples, const WaveformParams& params);

    // Accumulate individual datapoints rendered as soft circular dots into the
    // histogram. Each dot uses a (1 - r^2)^2 falloff. By default a dot integrates
    // to 1 (energy-conserving, useful for density analysis); sub-pixel dots are
    // clamped to a 1-pixel radius so they light one pixel at ~intensity 1 rather
    // than scaling without bound. Shares the accumulation image with draw() /
    // draw_waveform(), so dots and traces may be composited together.
    void draw_points(std::span<const Sample> points, const PointParams& params);

    // Free the sample/waveform staging and device buffers to recover GPU memory.
    // Blocks until any in-flight GPU work is complete. Safe to call between renders;
    // draw() and draw_waveform() will reallocate on next use.
    void release_buffers();

    // R32_SFLOAT image in COLOR_ATTACHMENT_OPTIMAL. Valid until next call or destruction.
    const vke::Image& image() const noexcept;

    uint32_t width()  const noexcept;
    uint32_t height() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace plot
