#pragma once
#include <cstdint>
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

} // namespace plot
