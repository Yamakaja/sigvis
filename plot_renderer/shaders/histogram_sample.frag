#version 460

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    float max_intensity;
} pc;

layout(binding = 0) uniform sampler2D histogram_sampler;

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
    vec2  uv    = gl_FragCoord.xy / vec2(textureSize(histogram_sampler, 0));
    float raw   = texture(histogram_sampler, uv).r;

    float luma  = (raw > 0.0) ? log(1.0 + raw) / (log(10.0) * pc.max_intensity) : 0.0;
    float alpha = (raw > 0.0) ? 1.0 : 0.0;

    outColor = vec4(turbo(luma) * alpha, alpha);
}
