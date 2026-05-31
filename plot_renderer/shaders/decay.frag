#version 460

// Phosphor decay pass. Paired with the fullscreen triangle in histogram_sample.vert
// and a blend state of (src = ZERO, dst = CONSTANT_COLOR, op = ADD) so that the
// accumulation target is multiplied in place by the dynamic blend constant (alpha):
//
//     hist = 0 * src + alpha * hist
//
// The fragment output itself is irrelevant (multiplied by ZERO); we only need a
// fragment to be produced for every texel so the blend runs everywhere.
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(0.0);
}
