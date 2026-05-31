#pragma once

// Signal-space view window for the scope. x is the normalized position within a
// record [0,1] (time axis); y is amplitude in signal units. Pan/zoom mutate this
// window; the viewer resets phosphor persistence whenever it changes.
struct ViewState {
    float x_min = 0.0f;
    float x_max = 1.0f;
    float y_min = -1.5f;
    float y_max = 1.5f;

    float x_span() const { return x_max - x_min; }
    float y_span() const { return y_max - y_min; }

    // Pan by a fraction of the current span (e.g. from a pixel delta / viewport size).
    void pan(float dx_frac, float dy_frac) {
        float dx = dx_frac * x_span();
        float dy = dy_frac * y_span();
        x_min += dx; x_max += dx;
        y_min += dy; y_max += dy;
    }

    // Zoom about a focus point given in [0,1] fractions of the current window.
    // factor < 1 zooms in, > 1 zooms out.
    void zoom(float factor, float fx, float fy, bool zoom_x, bool zoom_y) {
        if (zoom_x) {
            float cx = x_min + fx * x_span();
            float nx = x_span() * factor;
            x_min = cx - fx * nx;
            x_max = cx + (1.0f - fx) * nx;
        }
        if (zoom_y) {
            float cy = y_min + fy * y_span();
            float ny = y_span() * factor;
            y_min = cy - fy * ny;
            y_max = cy + (1.0f - fy) * ny;
        }
    }
};
