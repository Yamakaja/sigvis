// Realtime phosphor waveform viewer.
//
//   GLFW window + Vulkan swapchain (vke::Swapchain)
//   plot::WaveformScope : persistent R32F accumulator, time-based phosphor decay,
//                         additive trace draws, log+Turbo composite
//   DataSource          : background thread producing 1 MS/s-style 1k records
//   Dear ImGui overlay  : FPS, view, waveform stats, persistence/line controls
//
// Free-run trigger: every record overlays across the full width; persistence fades
// at a configurable time constant (tau). Pan/zoom remaps new traces and resets
// persistence.

#include <vke/vke.hpp>
#include <plot/waveform_scope.hpp>
#include <plot/roll_scope.hpp>

#include "data_source.hpp"
#include "audio_source.hpp"
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

        plot::WaveformScope scope(ctx, swapchain.extent().width, swapchain.extent().height,
                                  swapchain.format(), FRAMES_IN_FLIGHT);
        plot::RollScope roll(ctx, swapchain.extent().width, swapchain.extent().height,
                             swapchain.format(), SAMPLE_RATE,
                             /*max_window_seconds*/ 10.0f, FRAMES_IN_FLIGHT);

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
        // Synthetic: 1 MS/s in 1k-sample records at 1 kHz. Audio: 48 kHz mono capture
        // (created lazily on first selection). Exactly one is active at a time.
        const uint32_t CHUNK = 1000;
        DataSource synth(CHUNK, 1000);
        std::unique_ptr<AudioSource> audio;        // lazily started
        std::string audio_device;                  // empty = server default
        ISource* active = &synth;
        synth.start();
        roll.set_sample_rate(active->sample_rate());

        enum class SourceKind { Synthetic, Audio };
        SourceKind source_kind = SourceKind::Synthetic;

        auto switch_source = [&](SourceKind kind) {
            ISource* next = active;
            if (kind == SourceKind::Synthetic) {
                next = &synth;
            } else {
                if (!audio) audio = std::make_unique<AudioSource>(48000, 256, audio_device);
                if (!audio->ok() && audio->produced() == 0) audio->start();
                next = audio.get();
            }
            if (next != active) {
                active = next;
                roll.set_sample_rate(active->sample_rate());
                roll.reset();
                scope.reset_persistence();
            }
            source_kind = kind;
        };

        auto recreate = [&] {
            int w = 0, h = 0;
            glfwGetFramebufferSize(window, &w, &h);
            while (w == 0 || h == 0) { glfwGetFramebufferSize(window, &w, &h); glfwWaitEvents(); }
            vkDeviceWaitIdle(device);
            swapchain.recreate(static_cast<uint32_t>(w), static_cast<uint32_t>(h));
            rebuild_render_finished();
            scope.resize(swapchain.extent().width, swapchain.extent().height);
            roll.resize(swapchain.extent().width, swapchain.extent().height);
            g_framebuffer_resized = false;
        };

        // ---- View / controls ----
        enum class Mode { Trigger, Roll };
        Mode  mode          = Mode::Trigger;
        ViewState view;
        float tau_seconds   = 0.3f;   // persistence time constant
        float line_width    = 2.5e-3f;
        float max_intensity = 1.0f;
        float min_weight    = 0.0f;
        float black_level   = 0.05f;  // ~exp(-3): single trace clears by ~3 tau
        bool  paused        = false;
        DataSource::Signal sig{};     // live synthetic-signal parameters

        // Roll-mode controls.
        float roll_window_s     = 2.0f;
        float roll_line_width   = 1.5f;  // vertical line width in pixels
        float roll_min_weight   = 0.2f;  // coverage floor (connectivity)
        float roll_density      = 1.0f;  // phosphor (0) .. additive density (1)
        int   roll_interp       = 1;     // Lagrange upsample factor (1=off, up to 8)
        float roll_max_intensity = 1.0f;
        float roll_black_level  = 0.05f;

        std::vector<std::vector<float>> drained;
        std::vector<float>              last_chunk;
        ChunkStats                      stats;
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
            bool view_changed = false;

            // --- Pan (left-drag) ---
            double mx = 0, my = 0;
            glfwGetCursorPos(window, &mx, &my);
            double dmx = mx - last_mx, dmy = my - last_my;
            last_mx = mx; last_my = my;
            int ww = 0, wh = 0;
            glfwGetWindowSize(window, &ww, &wh);
            if (!io.WantCaptureMouse && ww > 0 && wh > 0 &&
                glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                view.pan(static_cast<float>(-dmx / ww), static_cast<float>(dmy / wh));
                if (dmx != 0.0 || dmy != 0.0) view_changed = true;
            }

            // --- Zoom (scroll) ---
            if (!io.WantCaptureMouse && g_input.scroll_accum != 0.0 && ww > 0 && wh > 0) {
                float factor = std::exp(static_cast<float>(-g_input.scroll_accum) * 0.15f);
                float fx = static_cast<float>(mx / ww);
                float fy = 1.0f - static_cast<float>(my / wh); // screen top = y_max
                bool zoom_y = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                               glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
                bool zoom_x = !zoom_y; // shift = vertical zoom, otherwise horizontal
                view.zoom(factor, fx, fy, zoom_x, zoom_y);
                view_changed = true;
            }
            g_input.scroll_accum = 0.0;

            // --- Ingest new records (route to the active mode) ---
            synth.set_signal(sig);
            if (!paused) {
                active->drain(drained);
                for (auto& c : drained) {
                    if (mode == Mode::Trigger) scope.push_chunk(c);
                    else                       roll.push_chunk(c);
                }
                if (!drained.empty()) { last_chunk = drained.back(); stats = compute_stats(last_chunk); }
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
                ImGui::RadioButton("roll", &mode_i, 1);
                mode = static_cast<Mode>(mode_i);

                if (mode == Mode::Trigger) {
                    ImGui::SeparatorText("display (trigger)");
                    ImGui::Text("x: [%.4f, %.4f]", view.x_min, view.x_max);
                    ImGui::Text("y: [%.3f, %.3f]", view.y_min, view.y_max);
                    if (ImGui::Button("reset view")) { view = ViewState{}; view_changed = true; }
                    ImGui::SliderFloat("persistence (s)", &tau_seconds, 0.0f, 3.0f, "%.2f");
                    ImGui::SliderFloat("line width", &line_width, 5e-4f, 1e-2f, "%.4f");
                    ImGui::SliderFloat("max intensity", &max_intensity, 0.1f, 8.0f, "%.2f");
                    ImGui::SliderFloat("black level", &black_level, 0.0f, 0.5f, "%.3f");
                    ImGui::SliderFloat("box floor", &min_weight, 0.0f, 1.0f, "%.2f");
                } else {
                    ImGui::SeparatorText("display (roll)");
                    ImGui::SliderFloat("window (s)", &roll_window_s, 0.2f, 10.0f, "%.2f");
                    uint32_t K = roll.samples_per_stripe();
                    if (K > 0 && roll.sample_rate() > 0.0f) {
                        float exact = static_cast<float>(K) * roll.width() / roll.sample_rate();
                        ImGui::Text("%u samp/col -> %.3f s window", K, exact);
                        ImGui::Text("history: %.3f s shown / %.3f s buffered",
                                    roll.shown_seconds(), roll.buffered_seconds());
                        ImGui::Text("ring capacity: %.2f s", roll.capacity_seconds());
                    }
                    ImGui::SliderFloat("v line width (px)", &roll_line_width, 0.5f, 8.0f, "%.1f");
                    ImGui::SliderFloat("coverage floor", &roll_min_weight, 0.0f, 1.0f, "%.2f");
                    ImGui::SliderFloat("density", &roll_density, 0.0f, 1.0f, "%.2f");
                    ImGui::SliderInt("interp (Lagrange)", &roll_interp, 1, 8);
                    ImGui::SliderFloat("max intensity##roll", &roll_max_intensity, 0.1f, 8.0f, "%.2f");
                    ImGui::SliderFloat("black level##roll", &roll_black_level, 0.0f, 0.5f, "%.3f");
                }
                ImGui::Checkbox("pause", &paused);

                ImGui::SeparatorText("source");
                int src_i = static_cast<int>(source_kind);
                if (ImGui::RadioButton("synthetic", &src_i, 0)) switch_source(SourceKind::Synthetic);
                ImGui::SameLine();
                if (ImGui::RadioButton("audio", &src_i, 1)) switch_source(SourceKind::Audio);
                ImGui::Text("rate: %.0f Hz", active->sample_rate());

                if (source_kind == SourceKind::Synthetic) {
                    ImGui::SeparatorText("signal");
                    ImGui::SliderFloat("frequency (cyc/rec)", &sig.cycles, 0.5f, 60.0f, "%.2f");
                    ImGui::SliderFloat("amplitude", &sig.amplitude, 0.0f, 2.0f, "%.2f");
                    ImGui::SliderFloat("noise std", &sig.noise_std, 0.0f, 0.5f, "%.3f");
                    ImGui::SliderFloat("phase jitter", &sig.jitter, 0.0f, 3.1416f, "%.2f");
                    ImGui::SliderFloat("AM depth", &sig.am_depth, 0.0f, 1.0f, "%.2f");
                    ImGui::SliderFloat("AM rate", &sig.am_rate, 0.0f, 0.5f, "%.3f");
                } else {
                    ImGui::SeparatorText("audio");
                    if (audio && audio->ok()) {
                        ImGui::TextUnformatted("capturing (default source)");
                        ImGui::TextDisabled("for playback, capture a sink .monitor via PULSE_SOURCE");
                    } else if (audio) {
                        ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "audio error:");
                        ImGui::TextWrapped("%s", audio->error().c_str());
                    }
                }

                ImGui::SeparatorText("status");
                ImGui::Text("last record  min %.3f  max %.3f", stats.min, stats.max);
                ImGui::Text("             pp  %.3f  rms %.3f", stats.pp, stats.rms);
                ImGui::Text("produced %llu  dropped %llu",
                            (unsigned long long)active->produced(),
                            (unsigned long long)active->dropped());
                ImGui::Text("pending this frame: %zu",
                            mode == Mode::Trigger ? scope.pending() : roll.pending());
                ImGui::End();
            }
            ImGui::Render();

            if (view_changed && mode == Mode::Trigger) scope.reset_persistence();

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
                float alpha = (tau_seconds <= 1e-4f) ? 0.0f
                    : std::clamp(std::exp(-dt / tau_seconds), 0.0f, 1.0f);
                plot::ScopeParams params{
                    .x_min = view.x_min, .x_max = view.x_max,
                    .y_min = view.y_min, .y_max = view.y_max,
                    .line_width = line_width, .decay_alpha = alpha,
                    .max_intensity = max_intensity, .min_weight = min_weight,
                    .black_level = black_level,
                };
                scope.render(frame.cmd, target, params);
            } else {
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
                roll.render(frame.cmd, target, params);
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
