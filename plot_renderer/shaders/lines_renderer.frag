#version 460

layout(location = 0) in struct Varying {
    vec2 texCoord;
} fs_in;

layout(location = 0) out float outIntensity;

void main() {
    float v  = fs_in.texCoord.y;
    outIntensity = clamp(fs_in.texCoord.x * (1.0 - v * v * v * v), 0.0, 1.0);
}
