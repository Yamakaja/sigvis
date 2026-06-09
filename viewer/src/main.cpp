// Realtime phosphor waveform viewer.
//
//   GLFW window + Vulkan swapchain (vke::Swapchain)
//   plot::StreamScope   : one shared gap-aware SampleRing feeding both render modes —
//                         roll (per-column envelope/density strip chart) and trigger
//                         (CPU-scanned, holdoff-gated captures drawn into a phosphor
//                         accumulator; spc-keyed box/line LOD; log+Turbo composite).
//   ISource             : background acquisition threads producing sequence-numbered
//                         frames (synthetic / live audio).
//   Dear ImGui overlay  : FPS, view, mode + trigger/roll controls, source, stats.

#include <vke/vke.hpp>
#include <plot/stream_scope.hpp>

#include "data_source.hpp"
#include "audio_source.hpp"
#include "plstream_source.hpp"
#include "view_state.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr uint32_t FRAMES_IN_FLIGHT = 2;

struct FrameSync {
    VkSemaphore        image_available = VK_NULL_HANDLE;
    VkFence            in_flight       = VK_NULL_HANDLE;
    vke::CommandBuffer cmd;  // kept alive until its fence is next waited on
};

// Shared input state (mutated by GLFW callbacks).
struct InputState {
    double scroll_accum = 0.0;
};
InputState g_input;
bool       g_framebuffer_resized = false;

void framebuffer_size_cb(GLFWwindow*, int, int) { g_framebuffer_resized = true; }
void scroll_cb(GLFWwindow*, double, double yoff) { g_input.scroll_accum += yoff; }

struct ChunkStats { float min = 0, max = 0, pp = 0, rms = 0; };

ChunkStats compute_stats(const std::vector<float>& c) {
    ChunkStats s;
    if (c.empty()) return s;
    s.min = c[0]; s.max = c[0];
    double sumsq = 0.0;
    for (float v : c) {
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
        sumsq += static_cast<double>(v) * v;
    }
    s.pp  = s.max - s.min;
    s.rms = static_cast<float>(std::sqrt(sumsq / c.size()));
    return s;
}

} // namespace

int main(int argc, char** argv) {
    int  max_frames = 0; // 0 = run until window closed
    bool vsync = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) max_frames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--no-vsync") == 0)          vsync = false;
    }

    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1600, 900, "sigvis scope", nullptr, nullptr);
    if (!window) { std::fprintf(stderr, "glfwCreateWindow failed\n"); glfwTerminate(); return 1; }
    glfwSetFramebufferSizeCallback(window, framebuffer_size_cb);
    // Install our scroll callback BEFORE ImGui so ImGui chains to it.
    glfwSetScrollCallback(window, scroll_cb);

    uint32_t glfw_ext_count = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
    std::vector<const char*> instance_exts(glfw_exts, glfw_exts + glfw_ext_count);

    vke::ContextCreateInfo ci{
        .require_mesh_shaders      = true,
        .extra_instance_extensions = instance_exts,
        .surface_factory = [window](VkInstance instance) -> VkSurfaceKHR {
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
                return VK_NULL_HANDLE;
            return surface;
        },
    };

    try {
        vke::Context ctx(ci);
        std::printf("Device: %s\n", ctx.physical_device_info().device_name.c_str());
        VkDevice device = ctx.native_device();

        int fbw = 0, fbh = 0;
        glfwGetFramebufferSize(window, &fbw, &fbh);
        vke::Swapchain swapchain(ctx, static_cast<uint32_t>(fbw),
                                 static_cast<uint32_t>(fbh), vsync);

        const float SAMPLE_RATE = 1.0e6f; // synthetic 1 MS/s (CHUNK * chunks_per_sec)

        // Unified scope: one shared gap-aware ring feeding both render modes. Frames are
        // ingested once per frame regardless of mode (see the drain loop below).
        plot::StreamScope stream(ctx, swapchain.extent().width, swapchain.extent().height,
                                 swapchain.format(), SAMPLE_RATE,
                                 /*max_window_seconds*/ 2.0f, FRAMES_IN_FLIGHT);

        // ---- Frame sync ----
        FrameSync frames[FRAMES_IN_FLIGHT];
        for (auto& f : frames) {
            VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            VkFenceCreateInfo     fci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                       .flags = VK_FENCE_CREATE_SIGNALED_BIT };
            vkCreateSemaphore(device, &sci, nullptr, &f.image_available);
            vkCreateFence(device, &fci, nullptr, &f.in_flight);
        }
        std::vector<VkSemaphore> render_finished;
        auto rebuild_render_finished = [&] {
            for (VkSemaphore s : render_finished) vkDestroySemaphore(device, s, nullptr);
            render_finished.assign(swapchain.image_count(), VK_NULL_HANDLE);
            for (auto& s : render_finished) {
                VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
                vkCreateSemaphore(device, &sci, nullptr, &s);
            }
        };
        rebuild_render_finished();

        // ---- ImGui ----
        VkDescriptorPool imgui_pool = VK_NULL_HANDLE;
        {
            VkDescriptorPoolSize pool_sizes[] = {{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64 }};
            VkDescriptorPoolCreateInfo dpci{
                .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                .maxSets       = 64, .poolSizeCount = 1, .pPoolSizes = pool_sizes,
            };
            vkCreateDescriptorPool(device, &dpci, nullptr, &imgui_pool);
        }
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForVulkan(window, true);

        VkFormat color_format = swapchain.format();
        ImGui_ImplVulkan_InitInfo imgui_init{};
        imgui_init.ApiVersion     = VK_API_VERSION_1_3;
        imgui_init.Instance       = ctx.native_instance();
        imgui_init.PhysicalDevice = ctx.native_physical_device();
        imgui_init.Device         = device;
        imgui_init.QueueFamily    = ctx.graphics_family();
        imgui_init.Queue          = ctx.graphics_queue();
        imgui_init.DescriptorPool = imgui_pool;
        imgui_init.MinImageCount  = 2;
        imgui_init.ImageCount     = swapchain.image_count();
        imgui_init.UseDynamicRendering = true;
        imgui_init.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        imgui_init.PipelineInfoMain.PipelineRenderingCreateInfo = VkPipelineRenderingCreateInfo{
            .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount    = 1,
            .pColorAttachmentFormats = &color_format,
        };
        ImGui_ImplVulkan_Init(&imgui_init);

        // ---- Sample sources ----
        // Synthetic: 1 MS/s in 1k-sample records at 1 kHz. Audio: 48 kHz mono capture.
        // RedPitaya: live streaming ADC over the network. Created lazily on selection;
        // exactly one is active at a time.
        const uint32_t CHUNK = 1000;
        DataSource synth(CHUNK, 1000);
        std::unique_ptr<AudioSource>    audio;       // lazily started
        std::string                     audio_device; // empty = server default
        std::unique_ptr<PlstreamSource> rp;          // lazily started
        char        rp_host[64]   = "192.168.1.100";
        int         rp_ctrl_port  = 7654;
        int         rp_udp_port   = 5000;
        int         rp_channel    = 0;               // 0 = A, 1 = B
        ISource* active = &synth;
        synth.start();
        stream.set_sample_rate(active->sample_rate());

        enum class SourceKind { Synthetic, Audio, RedPitaya };
        SourceKind source_kind = SourceKind::Synthetic;

        auto switch_source = [&](SourceKind kind) {
            ISource* next = active;
            if (kind == SourceKind::Synthetic) {
                next = &synth;
            } else if (kind == SourceKind::Audio) {
                if (!audio) audio = std::make_unique<AudioSource>(48000, 256, audio_device);
                if (!audio->ok() && audio->produced() == 0) audio->start();
                next = audio.get();
            } else {
                // RedPitaya connects only via the explicit "connect" button (avoid
                // auto-connecting to a possibly-wrong default host, whose blocking
                // connect would later freeze the UI when joined). Until connected, keep
                // the current source running.
                next = rp ? static_cast<ISource*>(rp.get()) : active;
            }
            if (next != active) {
                active = next;
                stream.set_sample_rate(active->sample_rate()); // resets ring + persistence
            }
            source_kind = kind;
        };

        // (Re)connect the Red Pitaya with the current connection fields.
        auto rp_reconnect = [&] {
            if (rp) rp->stop();
            rp = std::make_unique<PlstreamSource>(
                rp_host, static_cast<uint16_t>(rp_ctrl_port),
                static_cast<uint16_t>(rp_udp_port),
                rp_channel == 0 ? PlstreamSource::Channel::A : PlstreamSource::Channel::B);
            rp->start();
            active = rp.get();
            source_kind = SourceKind::RedPitaya;
            stream.set_sample_rate(active->sample_rate());
            stream.reset();
        };

        auto recreate = [&] {
            int w = 0, h = 0;
            glfwGetFramebufferSize(window, &w, &h);
            while (w == 0 || h == 0) { glfwGetFramebufferSize(window, &w, &h); glfwWaitEvents(); }
            vkDeviceWaitIdle(device);
            swapchain.recreate(static_cast<uint32_t>(w), static_cast<uint32_t>(h));
            rebuild_render_finished();
            stream.resize(swapchain.extent().width, swapchain.extent().height);
            g_framebuffer_resized = false;
        };

        // ---- View / controls ----
        enum class Mode { Trigger, Roll, Spectrum };
        Mode  mode          = Mode::Trigger;
        ViewState view;               // x in [0,1] = window fraction (trigger); y = amplitude
        ViewState spec_view{0.0f, 1.0f, -120.0f, 6.0f}; // x = fraction of Nyquist; y = dBFS
        bool  paused        = false;
        DataSource::Signal sig{};     // live synthetic-signal parameters

        // Trigger-mode controls.
        float trig_window_s     = 1e-3f;  // timebase L (seconds, centered on trigger)
        float trig_level        = 0.0f;
        int   trig_slope        = 0;      // 0 = rising, 1 = falling
        float trig_hysteresis   = 0.05f;
        float trig_holdoff_s    = 0.0f;   // 0 = one window
        bool  trig_eqt          = true;   // sub-sample equivalent-time alignment
        bool  trig_auto         = true;   // auto mode: free-run if no trigger within ~100ms
        float trig_prefrac      = 0.5f;   // trigger position in the window [0,1]
        float trig_tau_s        = 0.3f;   // persistence time constant
        int   trig_grab         = 0;      // on-screen handle drag: 0 none, 1 level, 2 marker
        bool  prev_lmb          = false;  // previous left-mouse state (for click/release edges)
        float trig_line_width   = 2.5e-3f;
        float trig_max_intensity = 1.0f;
        float trig_min_weight   = 0.0f;
        float trig_black_level  = 0.05f;

        // Spectrum-mode controls (density spectrum; dBFS range lives in spec_view.y).
        int   spec_fft_log2     = 12;     // FFT size = 1 << spec_fft_log2
        int   spec_window       = 0;      // 0 Hann, 1 Blackman-Harris, 2 Flat-top, 3 Rect
        float spec_line_width   = 2.5e-3f;
        float spec_tau          = 0.3f;   // persistence time constant
        float spec_max_intensity = 1.0f;
        float spec_min_weight   = 0.0f;
        float spec_black_level  = 0.05f;

        // Roll-mode controls.
        float roll_window_s     = 2.0f;
        float roll_line_width   = 1.5f;  // vertical line width in pixels
        float roll_min_weight   = 0.2f;  // coverage floor (connectivity)
        float roll_density      = 1.0f;  // phosphor (0) .. additive density (1)
        int   roll_interp       = 1;     // Lagrange upsample factor (1=off, up to 8)
        float roll_max_intensity = 1.0f;
        float roll_black_level  = 0.05f;

        std::vector<Frame> drained;
        std::vector<float> last_chunk;
        ChunkStats         stats;
        double last_mx = 0, last_my = 0;
        glfwGetCursorPos(window, &last_mx, &last_my);

        uint32_t frame_idx = 0;
        int      presented = 0;
        double   last_time = glfwGetTime();
        float    fps_smoothed = 0.0f;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            double now = glfwGetTime();
            float dt = static_cast<float>(now - last_time);
            last_time = now;
            if (dt > 0.0f) {
                float inst = 1.0f / dt;
                fps_smoothed = (fps_smoothed == 0.0f) ? inst : fps_smoothed * 0.9f + inst * 0.1f;
            }

            ImGuiIO& io = ImGui::GetIO();
            // Note: view (x/y window) changes are picked up by the renderers via their
            // params each frame — trigger persistence auto-clears on the change — so no
            // explicit "view changed" flag is needed here.

            // --- Mouse / window geometry ---
            double mx = 0, my = 0;
            glfwGetCursorPos(window, &mx, &my);
            double dmx = mx - last_mx, dmy = my - last_my;
            last_mx = mx; last_my = my;
            int ww = 0, wh = 0;
            glfwGetWindowSize(window, &ww, &wh);
            const bool lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

            // --- On-screen trigger handles (level line + trigger marker) ---
            // Screen positions in window-logical pixels (top = y_max, left = x_min), the
            // same space as the cursor and the ImGui overlay below.
            float trig_level_y = 0.f, trig_marker_x = 0.f;
            bool  over_level = false, over_marker = false, over_hyst = false, over_handle = false;
            if (mode == Mode::Trigger && ww > 0 && wh > 0) {
                auto y_of = [&](float a) { return (1.f - (a - view.y_min) / view.y_span()) * wh; };
                trig_level_y  = y_of(trig_level);
                trig_marker_x = (trig_prefrac - view.x_min) / view.x_span() * ww;
                float hy0 = y_of(trig_level + trig_hysteresis);   // upper band edge
                float hy1 = y_of(trig_level - trig_hysteresis);   // lower band edge
                over_level  = std::abs(static_cast<float>(my) - trig_level_y) < 6.f;
                over_marker = std::abs(static_cast<float>(mx) - trig_marker_x) < 6.f;
                over_hyst   = (mx < 44.0) && (std::abs(static_cast<float>(my) - hy0) < 6.f ||
                                              std::abs(static_cast<float>(my) - hy1) < 6.f);
                over_handle = (over_level || over_marker || over_hyst) && !io.WantCaptureMouse;

                if (lmb && !prev_lmb && over_handle)            // grab on click edge
                    trig_grab = over_marker ? 2 : (over_hyst ? 3 : 1); // marker > hyst > level
                if (trig_grab != 0 && lmb) {
                    float amp = view.y_max - static_cast<float>(my) / wh * view.y_span();
                    if      (trig_grab == 1) trig_level = amp;
                    else if (trig_grab == 2) trig_prefrac = std::clamp(
                        view.x_min + static_cast<float>(mx) / ww * view.x_span(), 0.02f, 0.98f);
                    else if (trig_grab == 3) trig_hysteresis = std::clamp(
                        std::abs(amp - trig_level), 0.f, view.y_span());
                }
            }
            if (!lmb) trig_grab = 0;
            prev_lmb = lmb;

            // The active view: spectrum has its own (x = frequency, y = dBFS).
            ViewState& av = (mode == Mode::Spectrum) ? spec_view : view;

            // --- Pan (left-drag) ---  (suppressed while over/dragging a trigger handle)
            if (!io.WantCaptureMouse && !over_handle && trig_grab == 0 && ww > 0 && wh > 0 && lmb) {
                av.pan(static_cast<float>(-dmx / ww), static_cast<float>(dmy / wh));
            }

            // --- Zoom (scroll) ---
            if (!io.WantCaptureMouse && g_input.scroll_accum != 0.0 && ww > 0 && wh > 0) {
                float factor = std::exp(static_cast<float>(-g_input.scroll_accum) * 0.15f);
                float fx = static_cast<float>(mx / ww);
                float fy = 1.0f - static_cast<float>(my / wh); // screen top = y_max
                bool zoom_y = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                               glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
                bool zoom_x = !zoom_y; // shift = vertical zoom, otherwise horizontal
                av.zoom(factor, fx, fy, zoom_x, zoom_y);
            }
            g_input.scroll_accum = 0.0;

            // --- Ingest new records onto the shared timeline ---
            synth.set_signal(sig);
            // Sync the time axis: a no-op unless the rate changed (e.g. the Red Pitaya's
            // rate becomes known on its first frame), which resets the ring.
            stream.set_sample_rate(active->sample_rate());
            if (!paused) {
                active->drain(drained);
                for (auto& f : drained) {
                    plot::RingFrame rf{ .seq = f.seq, .timestamp = f.timestamp,
                                        .samples_per_frame = f.samples_per_frame,
                                        .samples = f.samples };
                    stream.ingest(rf); // mode-independent: both paths read the same ring
                }
                if (!drained.empty()) { last_chunk = drained.back().samples; stats = compute_stats(last_chunk); }
            } else {
                drained.clear();
            }

            // --- UI ---
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            {
                ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_FirstUseEver);
                ImGui::Begin("scope");
                ImGui::Text("%.1f FPS  (%.2f ms)", fps_smoothed, dt * 1000.0f);
                ImGui::TextDisabled("%s", ctx.physical_device_info().device_name.c_str());
                ImGui::Separator();
                int mode_i = static_cast<int>(mode);
                ImGui::TextUnformatted("mode");
                ImGui::RadioButton("trigger", &mode_i, 0); ImGui::SameLine();
                ImGui::RadioButton("roll", &mode_i, 1); ImGui::SameLine();
                ImGui::RadioButton("spectrum", &mode_i, 2);
                Mode prev_mode = mode;
                mode = static_cast<Mode>(mode_i);
                if (mode != prev_mode) stream.reset_persistence(); // shared accumulator

                if (mode == Mode::Trigger) {
                    ImGui::SeparatorText("display (trigger)");
                    ImGui::Text("x window frac: [%.3f, %.3f]", view.x_min, view.x_max);
                    ImGui::Text("y: [%.3f, %.3f]", view.y_min, view.y_max);
                    if (ImGui::Button("reset view")) view = ViewState{};
                    ImGui::SameLine();
                    if (ImGui::Button("autoset")) {
                        auto a = stream.autoset();
                        if (a.ok) {
                            view.y_min = a.y_min; view.y_max = a.y_max;
                            view.x_min = 0.0f;    view.x_max = 1.0f; // reset x pan
                            trig_level = a.level; trig_hysteresis = a.hysteresis;
                            trig_prefrac = 0.5f;
                            if (a.has_timebase)
                                trig_window_s = std::clamp(a.window_seconds, 1e-5f, 1.0f);
                        }
                    }
                    ImGui::SameLine();
                    ImGui::Checkbox("auto", &trig_auto);
                    ImGui::SliderFloat("timebase (s)", &trig_window_s, 1e-5f, 1.0f, "%.5f",
                                       ImGuiSliderFlags_Logarithmic);
                    ImGui::SliderFloat("trig level", &trig_level, -2.0f, 2.0f, "%.3f");
                    ImGui::RadioButton("rising", &trig_slope, 0); ImGui::SameLine();
                    ImGui::RadioButton("falling", &trig_slope, 1);
                    ImGui::SliderFloat("hysteresis", &trig_hysteresis, 0.0f, 0.5f, "%.3f");
                    ImGui::SliderFloat("holdoff (s, 0=win)", &trig_holdoff_s, 0.0f, 0.05f, "%.5f",
                                       ImGuiSliderFlags_Logarithmic);
                    ImGui::Checkbox("equivalent-time", &trig_eqt);
                    ImGui::SliderFloat("trig pos", &trig_prefrac, 0.02f, 0.98f, "%.2f");
                    ImGui::SliderFloat("persistence (s)", &trig_tau_s, 0.0f, 3.0f, "%.2f");
                    ImGui::SliderFloat("line width", &trig_line_width, 5e-4f, 1e-2f, "%.4f");
                    ImGui::SliderFloat("max intensity", &trig_max_intensity, 0.1f, 8.0f, "%.2f");
                    ImGui::SliderFloat("black level", &trig_black_level, 0.0f, 0.5f, "%.3f");
                    ImGui::SliderFloat("box floor", &trig_min_weight, 0.0f, 1.0f, "%.2f");
                    uint32_t Lsamp = static_cast<uint32_t>(std::lround(trig_window_s * stream.sample_rate()));
                    ImGui::Text("L = %u samp  captures/frame %zu", Lsamp, stream.captures_last_frame());
                } else if (mode == Mode::Roll) {
                    ImGui::SeparatorText("display (roll)");
                    ImGui::SliderFloat("window (s)", &roll_window_s, 0.2f, 10.0f, "%.2f");
                    uint32_t K = stream.samples_per_stripe();
                    if (K > 0 && stream.sample_rate() > 0.0f) {
                        float exact = static_cast<float>(K) * stream.width() / stream.sample_rate();
                        ImGui::Text("%u samp/col -> %.3f s window", K, exact);
                        ImGui::Text("history: %.3f s shown / %.3f s buffered",
                                    stream.shown_seconds(), stream.buffered_seconds());
                        ImGui::Text("ring capacity: %.2f s", stream.capacity_seconds());
                    }
                    ImGui::SliderFloat("v line width (px)", &roll_line_width, 0.5f, 8.0f, "%.1f");
                    ImGui::SliderFloat("coverage floor", &roll_min_weight, 0.0f, 1.0f, "%.2f");
                    ImGui::SliderFloat("density", &roll_density, 0.0f, 1.0f, "%.2f");
                    ImGui::SliderInt("interp (Lagrange)", &roll_interp, 1, 8);
                    ImGui::SliderFloat("max intensity##roll", &roll_max_intensity, 0.1f, 8.0f, "%.2f");
                    ImGui::SliderFloat("black level##roll", &roll_black_level, 0.0f, 0.5f, "%.3f");
                } else { // Spectrum
                    ImGui::SeparatorText("display (spectrum)");
                    if (ImGui::Button("reset view")) spec_view = ViewState{0.0f, 1.0f, -120.0f, 6.0f};
                    ImGui::SliderInt("fft size (2^n)", &spec_fft_log2, 8, 16);
                    uint32_t Nfft = 1u << spec_fft_log2;
                    const char* windows[] = { "Hann", "Blackman-Harris", "Flat-top", "Rect" };
                    ImGui::Combo("window", &spec_window, windows, IM_ARRAYSIZE(windows));
                    ImGui::SliderFloat("dBFS top", &spec_view.y_max, -60.0f, 12.0f, "%.0f");
                    ImGui::SliderFloat("dBFS floor", &spec_view.y_min, -180.0f, -20.0f, "%.0f");
                    ImGui::SliderFloat("persistence (s)", &spec_tau, 0.0f, 3.0f, "%.2f");
                    ImGui::SliderFloat("line width", &spec_line_width, 5e-4f, 1e-2f, "%.4f");
                    ImGui::SliderFloat("max intensity##spec", &spec_max_intensity, 0.1f, 8.0f, "%.2f");
                    ImGui::SliderFloat("black level##spec", &spec_black_level, 0.0f, 0.5f, "%.3f");
                    ImGui::SliderFloat("box floor##spec", &spec_min_weight, 0.0f, 1.0f, "%.2f");
                    float fs = stream.sample_rate();
                    float rbw = (fs > 0.0f) ? fs / static_cast<float>(Nfft) : 0.0f;
                    ImGui::Text("%u-pt FFT  RBW %.1f Hz  span 0..%.0f Hz", Nfft, rbw, 0.5f * fs);
                    // Cursor readout: frequency + dBFS at the pointer.
                    if (ww > 0 && wh > 0 && !io.WantCaptureMouse) {
                        float fcur = (spec_view.x_min + static_cast<float>(mx) / ww * spec_view.x_span())
                                     * 0.5f * fs;
                        float dcur = spec_view.y_max - static_cast<float>(my) / wh * spec_view.y_span();
                        ImGui::Text("cursor: %.1f Hz   %.1f dBFS", fcur, dcur);
                    } else {
                        ImGui::TextDisabled("cursor: (move over plot)");
                    }
                }
                ImGui::Checkbox("pause", &paused);

                ImGui::SeparatorText("source");
                int src_i = static_cast<int>(source_kind);
                if (ImGui::RadioButton("synthetic", &src_i, 0)) switch_source(SourceKind::Synthetic);
                ImGui::SameLine();
                if (ImGui::RadioButton("audio", &src_i, 1)) switch_source(SourceKind::Audio);
                ImGui::SameLine();
                if (ImGui::RadioButton("redpitaya", &src_i, 2)) switch_source(SourceKind::RedPitaya);
                ImGui::Text("rate: %.0f Hz", active->sample_rate());

                if (source_kind == SourceKind::Synthetic) {
                    ImGui::SeparatorText("signal");
                    ImGui::SliderFloat("frequency (cyc/rec)", &sig.cycles, 0.5f, 60.0f, "%.2f");
                    ImGui::SliderFloat("amplitude", &sig.amplitude, 0.0f, 2.0f, "%.2f");
                    ImGui::SliderFloat("noise std", &sig.noise_std, 0.0f, 0.5f, "%.3f");
                    ImGui::SliderFloat("phase jitter", &sig.jitter, 0.0f, 3.1416f, "%.2f");
                    ImGui::SliderFloat("AM depth", &sig.am_depth, 0.0f, 1.0f, "%.2f");
                    ImGui::SliderFloat("AM rate", &sig.am_rate, 0.0f, 0.5f, "%.3f");
                } else if (source_kind == SourceKind::Audio) {
                    ImGui::SeparatorText("audio");
                    if (audio && audio->ok()) {
                        ImGui::TextUnformatted("capturing (default source)");
                        ImGui::TextDisabled("for playback, capture a sink .monitor via PULSE_SOURCE");
                    } else if (audio) {
                        ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "audio error:");
                        ImGui::TextWrapped("%s", audio->error().c_str());
                    }
                } else {
                    ImGui::SeparatorText("redpitaya");
                    ImGui::InputText("host", rp_host, sizeof(rp_host));
                    ImGui::InputInt("ctrl port", &rp_ctrl_port);
                    ImGui::InputInt("udp port", &rp_udp_port);
                    bool ch_changed = false;
                    if (ImGui::RadioButton("ch A", &rp_channel, 0)) ch_changed = true;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("ch B", &rp_channel, 1)) ch_changed = true;
                    if (ch_changed && rp) {
                        rp->set_channel(rp_channel == 0 ? PlstreamSource::Channel::A
                                                        : PlstreamSource::Channel::B);
                        stream.reset(); // don't mix A/B on the timeline
                    }
                    if (ImGui::Button(rp ? "reconnect" : "connect")) rp_reconnect();
                    if (rp && rp->ok()) {
                        ImGui::TextColored(ImVec4(0.4f, 1, 0.5f, 1), "streaming @ %.0f Hz",
                                           rp->sample_rate());
                    } else if (rp) {
                        ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "error:");
                        ImGui::TextWrapped("%s", rp->error().c_str());
                    } else {
                        ImGui::TextDisabled("not connected");
                    }
                }

                ImGui::SeparatorText("status");
                ImGui::Text("last record  min %.3f  max %.3f", stats.min, stats.max);
                ImGui::Text("             pp  %.3f  rms %.3f", stats.pp, stats.rms);
                ImGui::Text("received %llu  renderer drops %llu",
                            (unsigned long long)active->produced(),
                            (unsigned long long)active->dropped());
                if (source_kind == SourceKind::RedPitaya && rp) {
                    ImGui::Text("fabric drops:     %llu",
                                (unsigned long long)rp->fabric_drops());
                    ImGui::Text("net/kernel drops: %llu",
                                (unsigned long long)rp->network_drops());
                }
                ImGui::Text("timeline gaps: %llu",
                            (unsigned long long)stream.dropped_frames());
                ImGui::End();
            }

            // --- On-screen trigger overlay: hysteresis band, level line, trigger marker ---
            if (mode == Mode::Trigger && ww > 0 && wh > 0) {
                ImDrawList* dl = ImGui::GetForegroundDrawList();
                const float W = static_cast<float>(ww), H = static_cast<float>(wh);
                auto y_of = [&](float amp) { return (1.f - (amp - view.y_min) / view.y_span()) * H; };

                const ImU32 c_level    = IM_COL32(255, 180,  40, 210);
                const ImU32 c_level_hi = IM_COL32(255, 215, 100, 255);
                const ImU32 c_band     = IM_COL32(255, 180,  40,  36);
                const ImU32 c_marker   = IM_COL32( 90, 220, 170, 190);
                const ImU32 c_marker_hi = IM_COL32(150, 245, 205, 255);

                // Hysteresis band (symmetric ±hyst dead zone) + grabbable edge handles.
                float yb0 = y_of(trig_level + trig_hysteresis);
                float yb1 = y_of(trig_level - trig_hysteresis);
                const bool hy_hi = over_hyst || trig_grab == 3;
                dl->AddRectFilled(ImVec2(0.f, std::min(yb0, yb1)), ImVec2(W, std::max(yb0, yb1)), c_band);
                dl->AddLine(ImVec2(0.f, yb0), ImVec2(W, yb0), hy_hi ? c_level_hi : c_level, 1.f);
                dl->AddLine(ImVec2(0.f, yb1), ImVec2(W, yb1), hy_hi ? c_level_hi : c_level, 1.f);
                dl->AddTriangleFilled(ImVec2(0.f, yb0-5.f), ImVec2(0.f, yb0+5.f), ImVec2(11.f, yb0),
                                      hy_hi ? c_level_hi : c_level);
                dl->AddTriangleFilled(ImVec2(0.f, yb1-5.f), ImVec2(0.f, yb1+5.f), ImVec2(11.f, yb1),
                                      hy_hi ? c_level_hi : c_level);

                const bool lvl_hi = over_level  || trig_grab == 1;
                const bool mk_hi  = over_marker || trig_grab == 2;

                // Trigger marker (vertical) + "T" cap.
                float mxp = std::clamp(trig_marker_x, 0.f, W);
                dl->AddLine(ImVec2(mxp, 0.f), ImVec2(mxp, H), mk_hi ? c_marker_hi : c_marker,
                            mk_hi ? 2.f : 1.5f);
                dl->AddText(ImVec2(mxp + 4.f, 3.f), mk_hi ? c_marker_hi : c_marker, "T");
                if (trig_marker_x < 0.f)      dl->AddTriangleFilled(
                    ImVec2(8, H*0.5f), ImVec2(20, H*0.5f-6), ImVec2(20, H*0.5f+6), c_marker_hi);
                else if (trig_marker_x > W)   dl->AddTriangleFilled(
                    ImVec2(W-8, H*0.5f), ImVec2(W-20, H*0.5f-6), ImVec2(W-20, H*0.5f+6), c_marker_hi);

                // Level line (full width) + right-edge handle, readout, slope glyph.
                float ly = std::clamp(trig_level_y, 0.f, H);
                dl->AddLine(ImVec2(0.f, ly), ImVec2(W, ly), lvl_hi ? c_level_hi : c_level,
                            lvl_hi ? 2.f : 1.5f);
                dl->AddTriangleFilled(ImVec2(W, ly - 6.f), ImVec2(W, ly + 6.f),
                                      ImVec2(W - 10.f, ly), lvl_hi ? c_level_hi : c_level);
                char buf[48];
                std::snprintf(buf, sizeof(buf), "%s %.3f", trig_slope == 0 ? "rise" : "fall", trig_level);
                dl->AddText(ImVec2(W - 96.f, ly - 16.f), lvl_hi ? c_level_hi : c_level, buf);
                if (trig_level_y < 0.f)       dl->AddTriangleFilled(
                    ImVec2(W*0.5f, 8), ImVec2(W*0.5f-6, 20), ImVec2(W*0.5f+6, 20), c_level_hi);
                else if (trig_level_y > H)    dl->AddTriangleFilled(
                    ImVec2(W*0.5f, H-8), ImVec2(W*0.5f-6, H-20), ImVec2(W*0.5f+6, H-20), c_level_hi);

                // Trigger point crosshair at the intersection (the convergence node).
                dl->AddCircle(ImVec2(mxp, ly), 5.f, c_level_hi, 12, 1.5f);
            }
            ImGui::Render();

            // (Trigger persistence auto-clears inside StreamScope when alignment params
            // — including the x/y view window — change, so panning re-harvests cleanly.)

            // --- Render ---
            FrameSync& frame = frames[frame_idx];
            vkWaitForFences(device, 1, &frame.in_flight, VK_TRUE, UINT64_MAX);

            auto image_index = swapchain.acquire(frame.image_available);
            if (!image_index) { recreate(); continue; }
            vkResetFences(device, 1, &frame.in_flight);

            vke::Image& target = swapchain.image(*image_index);

            frame.cmd = ctx.create_command_buffer();
            frame.cmd.begin();

            if (mode == Mode::Trigger) {
                float alpha = (trig_tau_s <= 1e-4f) ? 0.0f
                    : std::clamp(std::exp(-dt / trig_tau_s), 0.0f, 1.0f);
                plot::TriggerParams params{
                    .y_min = view.y_min, .y_max = view.y_max,
                    .window_seconds = trig_window_s,
                    .x_min = view.x_min, .x_max = view.x_max,
                    .level = trig_level,
                    .slope = trig_slope == 0 ? plot::TriggerSlope::Rising
                                             : plot::TriggerSlope::Falling,
                    .hysteresis = trig_hysteresis,
                    .holdoff_seconds = trig_holdoff_s,
                    .equivalent_time = trig_eqt,
                    .pre_frac = trig_prefrac,
                    .max_captures = 4096,
                    .auto_mode = trig_auto,
                    .auto_timeout_seconds = 0.1f,
                    .line_width = trig_line_width,
                    .decay_alpha = alpha,
                    .max_intensity = trig_max_intensity,
                    .min_weight = trig_min_weight,
                    .black_level = trig_black_level,
                };
                stream.render_trigger(frame.cmd, target, params);
            } else if (mode == Mode::Roll) {
                plot::RollParams params{
                    .y_min = view.y_min, .y_max = view.y_max,
                    .window_seconds = roll_window_s,
                    .line_width_px = roll_line_width,
                    .min_weight = roll_min_weight,
                    .density = roll_density,
                    .interp = static_cast<uint32_t>(roll_interp),
                    .max_intensity = roll_max_intensity,
                    .black_level = roll_black_level,
                };
                stream.render_roll(frame.cmd, target, params);
            } else { // Spectrum
                float alpha = (spec_tau <= 1e-4f) ? 0.0f
                    : std::clamp(std::exp(-dt / spec_tau), 0.0f, 1.0f);
                plot::SpectrumParams params{
                    .fft_size = 1u << spec_fft_log2,
                    .window = static_cast<plot::SpectrumWindow>(spec_window),
                    .x_min = spec_view.x_min, .x_max = spec_view.x_max,
                    .db_floor = spec_view.y_min, .db_top = spec_view.y_max,
                    .line_width = spec_line_width,
                    .decay_alpha = alpha,
                    .max_intensity = spec_max_intensity,
                    .min_weight = spec_min_weight,
                    .black_level = spec_black_level,
                };
                stream.render_spectrum(frame.cmd, target, params);
            }

            // ImGui overlay on top of the composited scope image.
            vke::CommandBuffer::ColorAttachmentInfo ui_color[] = {{
                .image    = &target,
                .load_op  = VK_ATTACHMENT_LOAD_OP_LOAD,
                .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            }};
            frame.cmd.begin_rendering({
                .color_attachments = ui_color, .auto_layout_transitions = false });
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), frame.cmd.native_handle());
            frame.cmd.end_rendering();

            frame.cmd.image_barrier({
                .image           = target,
                .new_layout      = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                .src_stage_mask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .dst_stage_mask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                .dst_access_mask = VK_ACCESS_2_NONE,
            });
            frame.cmd.end();

            VkSemaphore           waits[]   = { frame.image_available };
            VkPipelineStageFlags2 stages[]  = { VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT };
            VkSemaphore           signals[] = { render_finished[*image_index] };
            ctx.submit(frame.cmd, {
                .wait_semaphores   = waits,
                .wait_stage_masks  = stages,
                .signal_semaphores = signals,
                .fence             = frame.in_flight,
            });

            bool need_recreate = swapchain.present(*image_index, render_finished[*image_index]);
            if (need_recreate || g_framebuffer_resized) recreate();

            frame_idx = (frame_idx + 1) % FRAMES_IN_FLIGHT;
            if (max_frames > 0 && ++presented >= max_frames) break;
        }

        synth.stop();
        if (audio) audio->stop();
        if (rp) rp->stop();
        vkDeviceWaitIdle(device);

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        vkDestroyDescriptorPool(device, imgui_pool, nullptr);

        for (VkSemaphore s : render_finished) vkDestroySemaphore(device, s, nullptr);
        for (auto& f : frames) {
            vkDestroySemaphore(device, f.image_available, nullptr);
            vkDestroyFence(device, f.in_flight, nullptr);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
