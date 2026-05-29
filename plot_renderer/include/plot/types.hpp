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
    std::pair<float, float> x_range    = {0.0f, 1.0f}; // normalised [0,1] range of samples to display
    std::pair<float, float> y_range    = {-1.0f, 1.0f};
    float                   line_width = 1e-3f;
    float                   min_weight = 0.0f;          // floor for box-mode intensity weight [0, 1]
};

struct PointParams {
    glm::vec2 center     = {0.f, 0.f};
    glm::vec2 zoom       = {1.f, 1.f};
    float     point_size = 1e-2f; // dot radius in isometric units (≈ point_size * height/2 px)
    bool      normalize  = true;  // true: each dot integrates to 1; false: peak intensity 1
};

} // namespace plot
