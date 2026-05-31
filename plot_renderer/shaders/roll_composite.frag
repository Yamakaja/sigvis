#version 460

// Roll-mode composite. Reads a horizontally-circular strip image so the newest
// column sits at the right edge and history scrolls left, applies the same log +
// Turbo tonemap and black-level clipping as the phosphor composite.
//
// The strip is read as a STORAGE image (imageLoad), not sampled, so it can stay in
// VK_IMAGE_LAYOUT_GENERAL permanently. That avoids per-frame whole-image layout
// transitions (GENERAL<->SHADER_READ_ONLY) which, across frames in flight, would
// retile the entire image while it is being read — visible as global luminance
// flicker even though only a few columns change per frame.
// Strip resolution matches the render target (1:1), so we index by pixel.

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    float max_intensity;
    float black_level;   // raw values <= this map to pure black (alpha 0)
    uint  head_col;       // (columns_written) mod strip_width
    uint  strip_width;    // W
} pc;

layout(r32f, binding = 0) readonly uniform image2D strip;

// Turbo colormap polynomial approximation.
// Copyright 2019 Google LLC. SPDX-License-Identifier: Apache-2.0
// Authors: Anton Mikhailov (colormap), Ruofei Du (GLSL)
vec3 turbo(float x) {
    const vec4 r4 = vec4( 0.13572138,   4.61539260, -42.66032258,  132.13108234);
    const vec4 g4 = vec4( 0.09140261,   2.19418839,   4.84296658,  -14.18503333);
    const vec4 b4 = vec4( 0.10667330,  12.64194608, -60.58204836,  110.36276771);
    const vec2 r2 = vec2(-152.94239396, 59.28637943);
    const vec2 g2 = vec2(   4.27729857,  2.82956604);
    const vec2 b2 = vec2( -89.90310912, 27.34824973);
    x = clamp(x, 0.0, 1.0);
    vec4 v4 = vec4(1.0, x, x*x, x*x*x);
    vec2 v2 = v4.zw * v4.z;
    return vec3(dot(v4, r4) + dot(v2, r2),
                dot(v4, g4) + dot(v2, g2),
                dot(v4, b4) + dot(v2, b2));
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    int phys_x = int((pc.head_col + uint(px.x)) % pc.strip_width);
    float raw = imageLoad(strip, ivec2(phys_x, px.y)).r;

    bool  lit   = raw > pc.black_level;
    float luma  = lit ? log(1.0 + raw) / (log(10.0) * pc.max_intensity) : 0.0;
    float alpha = lit ? 1.0 : 0.0;
    outColor = vec4(turbo(luma) * alpha, alpha);
}
