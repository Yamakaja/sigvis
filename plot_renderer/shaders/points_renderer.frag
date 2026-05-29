#version 460

layout(push_constant) uniform PC {
    vec2  center;
    vec2  zoom;
    vec2  viewport_ratio;
    float point_size;
    float amplitude;      // peak intensity; profile integrates to 1 when normalize is on
    uint  n_points;
    uint  first_point;
} pc;

layout(location = 0) in struct Varying {
    vec2 uv;
} fs_in;

layout(location = 0) out float outIntensity;

void main() {
    // Cheap Gaussian-like bump: (1 - r^2)^2, compactly supported, no exp.
    float r2 = dot(fs_in.uv, fs_in.uv);
    if (r2 > 1.0)
        discard;
    float f = 1.0 - r2;
    outIntensity = pc.amplitude * f * f;
}
