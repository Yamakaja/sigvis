#pragma once
#include <memory>
#include <span>
#include <vke/vke.hpp>
#include "types.hpp"
#include <span>

namespace plot {

class EyeDiagramRenderer {
public:
    EyeDiagramRenderer() noexcept; // null state; safe to move-assign over
    explicit EyeDiagramRenderer(vke::Context& ctx,
                                 uint32_t width  = 1920,
                                 uint32_t height = 1080);
    ~EyeDiagramRenderer();

    EyeDiagramRenderer(const EyeDiagramRenderer&) = delete;
    EyeDiagramRenderer& operator=(const EyeDiagramRenderer&) = delete;
    EyeDiagramRenderer(EyeDiagramRenderer&&) noexcept;
    EyeDiagramRenderer& operator=(EyeDiagramRenderer&&) noexcept;

    // Upload sample data to GPU. samples is (n_traces * trace_length) flat.
    void set_samples(std::span<const Sample> samples, uint32_t trace_length);

    // Render histogram only (R32_SFLOAT). Reference valid until next render call.
    const vke::Image& render_histogram(const RenderParams& params);

    // Render full RGBA composite. Reference valid until next render call.
    const vke::Image& render(const RenderParams& params);

    uint32_t width()  const noexcept;
    uint32_t height() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace plot
