#pragma once
#include <cstdint>
#include <utility>
#include <glm/glm.hpp>

namespace plot {

struct Sample {
    glm::vec2 position;
};
static_assert(sizeof(Sample) == 8);

struct RenderParams {
    glm::vec2 center        = {0.f, 0.f};
    glm::vec2 zoom          = {1.f, 1.f};
    float     line_width    = 1.0f;
    uint32_t  width         = 1920;
    uint32_t  height        = 1080;
    float     max_intensity = 1.0f;
};

struct WaveformParams {
    std::pair<float, float> x_range   = {0.0f, 1.0f}; // normalised [0,1] range of samples to display
    std::pair<float, float> y_range   = {-1.0f, 1.0f};
    float                   line_width = 1e-3f;
};

} // namespace plot
