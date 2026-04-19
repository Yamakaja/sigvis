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

    // Zero the histogram image on the GPU.
    void clear();

    // Accumulate samples into the histogram. May be called repeatedly to build
    // up the histogram incrementally. Does NOT clear first — call clear() explicitly.
    void draw(std::span<const Sample> samples, uint32_t trace_length,
              const RenderParams& params);

    // R32_SFLOAT image in COLOR_ATTACHMENT_OPTIMAL. Valid until next call or destruction.
    const vke::Image& image() const noexcept;

    uint32_t width()  const noexcept;
    uint32_t height() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace plot
