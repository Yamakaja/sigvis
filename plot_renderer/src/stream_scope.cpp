#include <plot/stream_scope.hpp>
#include <vke/vke.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

#include "shaders/stream_reduce_comp_spv.hpp"
#include "shaders/stream_trigger_mesh_spv.hpp"
#include "shaders/waveform_renderer_mesh_spv.hpp"
#include "shaders/lines_renderer_frag_spv.hpp"
#include "shaders/histogram_sample_vert_spv.hpp"
#include "shaders/histogram_sample_frag_spv.hpp"
#include "shaders/decay_frag_spv.hpp"
#include "shaders/roll_composite_frag_spv.hpp"

namespace plot {

// Matches stream_reduce.comp (8 * u32 + 6 * f32 == 56 bytes).
struct StreamReducePC {
    uint32_t ring_capacity;
    uint32_t strip_width;
    uint32_t height;
    uint32_t samples_per_stripe;
    uint32_t ring_start_phys;
    uint32_t col_start_phys;
    float    y_min;
    float    y_max;
    float    line_width_px;
    float    weight_scale;
    float    min_weight;
    float    density;
    uint32_t interp;
    uint32_t samples_per_frame;
};
static_assert(sizeof(StreamReducePC) == 56);

// Matches roll_composite.frag.
struct StreamCompositePC {
    float    max_intensity;
    float    black_level;
    uint32_t head_col;
    uint32_t strip_width;
};
static_assert(sizeof(StreamCompositePC) == 16);

// Matches stream_trigger.mesh (sizeof == 56).
struct TriggerPC {
    glm::vec2 x_range;
    glm::vec2 y_range;
    glm::vec2 viewport_ratio;
    float     line_width;
    uint32_t  ring_capacity;
    uint32_t  run_phys_start;
    uint32_t  n_segments;
    uint32_t  viewport_width;
    float     min_weight;
    float     window_len;
    float     window_pos0;
};
static_assert(sizeof(TriggerPC) == 56);

// Matches histogram_sample.frag.
struct TrigCompositePC {
    float max_intensity;
    float black_level;
};

// Matches waveform_renderer.mesh (sizeof == 48) — reused to draw spectrum traces.
struct WaveformPC {
    glm::vec2 x_range;
    glm::vec2 y_range;
    glm::vec2 viewport_ratio;
    float     line_width;
    uint32_t  n_samples;
    uint32_t  first_segment;
    uint32_t  n_segments;
    uint32_t  viewport_width;
    float     min_weight;
};
static_assert(sizeof(WaveformPC) == 48);

// CPU radix-2 FFT + windowing → single-sided dBFS (0 dBFS = full-scale sine).
class SpectrumFFT {
public:
    void configure(uint32_t n, SpectrumWindow w) {
        if (n == n_ && w == window_) return;
        n_ = n; window_ = w;
        re_.assign(n_, 0.f); im_.assign(n_, 0.f); win_.resize(n_);
        twr_.resize(n_ / 2); twi_.resize(n_ / 2);
        rev_.resize(n_);

        const double TWO_PI = 2.0 * std::numbers::pi;
        double sum = 0.0;
        for (uint32_t i = 0; i < n_; ++i) {
            double x = static_cast<double>(i) / (n_ - 1), wv;
            switch (w) {
            case SpectrumWindow::Hann:    wv = 0.5 - 0.5 * std::cos(TWO_PI * x); break;
            case SpectrumWindow::Flattop: wv = 0.21557895 - 0.41663158 * std::cos(TWO_PI * x)
                + 0.277263158 * std::cos(2 * TWO_PI * x) - 0.083578947 * std::cos(3 * TWO_PI * x)
                + 0.006947368 * std::cos(4 * TWO_PI * x); break;
            case SpectrumWindow::BlackmanHarris: wv = 0.35875 - 0.48829 * std::cos(TWO_PI * x)
                + 0.14128 * std::cos(2 * TWO_PI * x) - 0.01168 * std::cos(3 * TWO_PI * x); break;
            default: wv = 1.0; break; // Rect
            }
            win_[i] = static_cast<float>(wv); sum += wv;
        }
        cg_ = static_cast<float>(sum / n_);                 // coherent gain (mean window)

        for (uint32_t k = 0; k < n_ / 2; ++k) {             // forward twiddles e^{-2πi k/N}
            double a = -TWO_PI * k / n_;
            twr_[k] = static_cast<float>(std::cos(a));
            twi_[k] = static_cast<float>(std::sin(a));
        }
        uint32_t bits = 0; while ((1u << bits) < n_) ++bits;
        for (uint32_t i = 0; i < n_; ++i) {                 // bit-reversal table
            uint32_t r = 0;
            for (uint32_t b = 0; b < bits; ++b) if (i & (1u << b)) r |= 1u << (bits - 1 - b);
            rev_[i] = r;
        }
    }

    uint32_t size() const { return n_; }
    uint32_t bins() const { return n_ / 2 + 1; }

    // Window + transform a real block; write bins() dBFS values to `out`.
    void process(std::span<const float> x, std::span<float> out) {
        for (uint32_t i = 0; i < n_; ++i) {
            uint32_t r = rev_[i];
            re_[r] = x[i] * win_[i];
            im_[r] = 0.f;
        }
        for (uint32_t len = 2; len <= n_; len <<= 1) {
            uint32_t half = len >> 1, step = n_ / len;
            for (uint32_t i = 0; i < n_; i += len)
                for (uint32_t k = 0; k < half; ++k) {
                    uint32_t t = k * step;
                    float wr = twr_[t], wi = twi_[t];
                    float vr = re_[i + k + half] * wr - im_[i + k + half] * wi;
                    float vi = re_[i + k + half] * wi + im_[i + k + half] * wr;
                    float ur = re_[i + k], ui = im_[i + k];
                    re_[i + k] = ur + vr; im_[i + k] = ui + vi;
                    re_[i + k + half] = ur - vr; im_[i + k + half] = ui - vi;
                }
        }
        const uint32_t nb = n_ / 2;
        const float    inv = 1.0f / (cg_ * n_);
        for (uint32_t k = 0; k <= nb; ++k) {
            float mag = std::sqrt(re_[k] * re_[k] + im_[k] * im_[k]);
            float amp = mag * inv * ((k == 0 || k == nb) ? 1.0f : 2.0f); // single-sided
            out[k] = 20.0f * std::log10(std::max(amp, 1e-12f));
        }
    }

private:
    uint32_t              n_ = 0;
    SpectrumWindow        window_ = SpectrumWindow::Hann;
    float                 cg_ = 1.0f;
    std::vector<float>    re_, im_, win_, twr_, twi_;
    std::vector<uint32_t> rev_;
};

namespace {
constexpr float    WEIGHT_SCALE     = 1024.0f;
constexpr uint32_t MAX_RUNS_PER_FRAME = 1u << 16; // total draw cap (runs)
// Per-frame work bounds, so a re-harvest (param change / pan / zoom) can never spike
// the CPU scan or the GPU draw — which previously stalled the render thread (dropping
// source frames → holes in the newest data) and could trip the GPU watchdog.
constexpr int64_t  MAX_SCAN_SAMPLES = 8 << 20;    // cap the trigger scan span (CPU)
constexpr uint64_t MAX_SEGMENTS     = 2u << 20;   // cap drawn segments / frame (GPU)
constexpr int64_t  ANALYSIS_SAMPLES = 1 << 17;    // recent span autoset() examines
constexpr double   AUTOSET_PERIODS  = 2.5;        // periods to show after autoset
constexpr uint32_t SPEC_MAX_BLOCKS  = 8;          // FFT blocks drawn per frame (CPU bound)
constexpr uint32_t MAX_SPEC_FFT     = 1u << 16;   // largest supported FFT size
constexpr uint32_t MAX_SPEC_BINS    = MAX_SPEC_FFT / 2 + 1;

VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) {
    return a == 0 ? v : ((v + a - 1) / a) * a;
}

VkPipelineColorBlendAttachmentState additive_blend() {
    return VkPipelineColorBlendAttachmentState{
        .blendEnable         = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .colorBlendOp        = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp        = VK_BLEND_OP_ADD,
        .colorWriteMask      = VK_COLOR_COMPONENT_R_BIT,
    };
}
// hist = 0*src + alpha*dst, alpha = dynamic blend constant.
VkPipelineColorBlendAttachmentState decay_blend() {
    return VkPipelineColorBlendAttachmentState{
        .blendEnable         = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR,
        .colorBlendOp        = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp        = VK_BLEND_OP_ADD,
        .colorWriteMask      = VK_COLOR_COMPONENT_R_BIT,
    };
}
} // namespace

struct StreamScope::Impl {
    vke::Context& ctx;
    uint32_t      width;
    uint32_t      height;
    VkFormat      target_format;
    uint32_t      frames_in_flight;

    SampleRing ring;
    uint64_t   ring_gen = ~0ull; // last generation the reduce set was written for

    // Strip image (R32_SFLOAT), horizontally circular.
    vke::Image strip;

    // Compute reduction.
    vke::DescriptorLayout reduce_layout;
    vke::DescriptorSet    reduce_set;
    vke::Pipeline         reduce_pipeline;

    // Composite (roll strip -> target).
    vke::DescriptorLayout composite_layout;
    vke::DescriptorSet    composite_set;
    vke::Pipeline         composite_pipeline;

    // ---- Trigger path ----
    vke::Image accumulator;        // persistent R32_SFLOAT phosphor buffer
    bool       needs_clear_persist = true;

    vke::DescriptorLayout trace_layout;   // ring samples, mesh stage
    vke::DescriptorSet    trace_set;
    vke::Pipeline         trace_pipeline;
    uint64_t              trace_gen = ~0ull;

    vke::Pipeline decay_pipeline;

    vke::Sampler          sampler;
    vke::DescriptorLayout tcomposite_layout;
    vke::DescriptorSet    tcomposite_set;
    vke::Pipeline         tcomposite_pipeline;

    // Trigger scan state (carried across frames for contiguous hysteresis).
    int64_t  last_scanned   = std::numeric_limits<int64_t>::min();
    bool     armed          = false;
    bool     have_prev      = false;
    int64_t  prev_idx       = 0;
    float    prev_s         = 0.0f;
    int64_t  next_allowed   = std::numeric_limits<int64_t>::min();
    int64_t  last_trigger_index = std::numeric_limits<int64_t>::min(); // for auto free-run
    size_t   captures_last  = 0;

    // Snapshot of alignment-affecting params; a change forces clear + re-harvest.
    bool  have_prev_params = false;
    float p_level = 0, p_hyst = 0, p_win = 0, p_xmin = 0, p_xmax = 0, p_ymin = 0, p_ymax = 0,
          p_holdoff = 0, p_prefrac = 0;
    int   p_slope = 0; bool p_eqt = false;

    struct Run { uint32_t phys_start; uint32_t n_segments; float window_pos0; };
    std::vector<Run> runs;

    // ---- Spectrum path (density spectrum into the same accumulator) ----
    VkDeviceSize ssbo_align = 0;
    SpectrumFFT  fft;
    vke::DescriptorLayout spec_layout;   // dynamic SSBO of dBFS bins, mesh stage
    vke::Pipeline         spec_pipeline; // waveform mesh + additive blend
    struct SpecSlot { vke::Buffer host; vke::Buffer device; vke::DescriptorSet set; };
    std::vector<SpecSlot> spec_slots;
    uint32_t     spec_slot_index = 0;
    uint32_t     spec_bins   = 0;        // bins of the configured FFT (N/2+1)
    VkDeviceSize spec_stride = 0;        // per-block byte stride (aligned)
    std::vector<float> spec_scratch;     // this frame's dBFS blocks, flattened
    uint32_t     spec_pending = 0;       // blocks gathered this frame
    int64_t      spec_scanned = std::numeric_limits<int64_t>::min(); // consumed up to
    // Snapshot of spectrum params that force a clear + rescan when changed.
    bool     have_spec_params = false;
    uint32_t sp_fft = 0; int sp_win = -1;
    float    sp_xmin = 0, sp_xmax = 0, sp_dbf = 0, sp_dbt = 0;

    // Column bookkeeping (absolute K-stripe counters off the shared ring).
    uint64_t cols_emitted_ = 0;     // columns turned into strip columns
    uint32_t cur_K         = 0;
    bool     needs_replay  = true;

    Impl(vke::Context& c, uint32_t w, uint32_t h, VkFormat fmt, float fs,
         float max_window_s, uint32_t fif)
        : ctx(c), width(w), height(h), target_format(fmt),
          frames_in_flight(std::max(1u, fif)),
          ring(c, max_window_s, std::max(1u, fif))
    {
        ssbo_align = ctx.physical_device_info().limits.minStorageBufferOffsetAlignment;
        ring.set_sample_rate(fs);
        create_strip();
        create_accumulator();
        create_pipelines();
        create_trigger_pipelines();
        create_spectrum_pipeline();
        clear_strip_immediate();
    }

    void create_strip() {
        strip = ctx.create_image({
            .width = width, .height = height, .format = VK_FORMAT_R32_SFLOAT,
            .usage = vke::ImageUsage::ColorAttachment | vke::ImageUsage::Storage |
                     vke::ImageUsage::Sampled | vke::ImageUsage::TransferSrc,
            .debug_name = "stream_strip" });
        needs_replay = true;
    }

    void create_accumulator() {
        accumulator = ctx.create_image({
            .width = width, .height = height, .format = VK_FORMAT_R32_SFLOAT,
            .usage = vke::ImageUsage::ColorAttachment | vke::ImageUsage::Sampled |
                     vke::ImageUsage::TransferSrc,
            .debug_name = "stream_accumulator" });
        needs_clear_persist = true;
    }

    void create_pipelines() {
        vke::DescriptorBinding rb[] = {
            { .binding = 0, .type = vke::DescriptorType::StorageBuffer,
              .stages = VK_SHADER_STAGE_COMPUTE_BIT },
            { .binding = 1, .type = vke::DescriptorType::StorageImage,
              .stages = VK_SHADER_STAGE_COMPUTE_BIT },
            { .binding = 2, .type = vke::DescriptorType::StorageBuffer,
              .stages = VK_SHADER_STAGE_COMPUTE_BIT },
        };
        reduce_layout = ctx.create_descriptor_layout({ .bindings = rb,
            .debug_name = "stream_reduce_layout" });
        reduce_set = reduce_layout.allocate_set("stream_reduce_set");

        auto comp = ctx.create_shader({
            .stage = vke::ShaderStage::Compute, .spirv_code = spv::stream_reduce_comp });
        vke::PushConstantRange rpc{ .stages = VK_SHADER_STAGE_COMPUTE_BIT,
                                    .size = sizeof(StreamReducePC) };
        const vke::DescriptorLayout* rl[] = { &reduce_layout };
        reduce_pipeline = ctx.create_pipeline(vke::ComputePipelineCreateInfo{
            .compute_shader = &comp, .descriptor_layouts = rl,
            .push_constant_ranges = { &rpc, 1 }, .debug_name = "stream_reduce_pipeline" });

        vke::DescriptorBinding cb{ .binding = 0,
            .type = vke::DescriptorType::StorageImage,
            .stages = VK_SHADER_STAGE_FRAGMENT_BIT };
        composite_layout = ctx.create_descriptor_layout({ .bindings = { &cb, 1 },
            .debug_name = "stream_composite_layout" });
        composite_set = composite_layout.allocate_set("stream_composite_set");

        auto fs_vert = ctx.create_shader({
            .stage = vke::ShaderStage::Vertex, .spirv_code = spv::histogram_sample_vert });
        auto comp_frag = ctx.create_shader({
            .stage = vke::ShaderStage::Fragment, .spirv_code = spv::roll_composite_frag });
        vke::PushConstantRange cpc{ .stages = VK_SHADER_STAGE_FRAGMENT_BIT,
                                    .size = sizeof(StreamCompositePC) };
        const vke::DescriptorLayout* cl[] = { &composite_layout };
        composite_pipeline = ctx.create_pipeline(vke::GraphicsPipelineCreateInfo{
            .vertex_shader = &fs_vert, .fragment_shader = &comp_frag,
            .descriptor_layouts = cl, .push_constant_ranges = { &cpc, 1 },
            .color_attachment_formats = { &target_format, 1 },
            .cull_mode = VK_CULL_MODE_NONE, .debug_name = "stream_composite_pipeline" });

        write_composite_descriptor();
    }

    // Reduce set points at the ring's device buffers; rewrite when they change (the
    // SampleRing recreates them on reconfigure — caller must be idle then).
    void write_reduce_descriptor() {
        reduce_set.write()
            .bind_storage_buffer(0, ring.samples_buffer())
            .bind_storage_image(1, strip, VK_IMAGE_LAYOUT_GENERAL)
            .bind_storage_buffer(2, ring.present_buffer())
            .commit();
        ring_gen = ring.generation();
    }
    void write_composite_descriptor() {
        composite_set.write()
            .bind_storage_image(0, strip, VK_IMAGE_LAYOUT_GENERAL)
            .commit();
    }

    void create_trigger_pipelines() {
        // ---- Trace pass: stream_trigger.mesh reads the shared ring, additive blend ----
        vke::DescriptorBinding tb{ .binding = 0,
            .type = vke::DescriptorType::StorageBuffer,
            .stages = VK_SHADER_STAGE_MESH_BIT_EXT };
        trace_layout = ctx.create_descriptor_layout({ .bindings = { &tb, 1 },
            .debug_name = "stream_trace_layout" });
        trace_set = trace_layout.allocate_set("stream_trace_set");

        auto trig_mesh = ctx.create_shader({
            .stage = vke::ShaderStage::Mesh, .spirv_code = spv::stream_trigger_mesh });
        auto lines_frag = ctx.create_shader({
            .stage = vke::ShaderStage::Fragment, .spirv_code = spv::lines_renderer_frag });
        vke::PushConstantRange tpc{ .stages = VK_SHADER_STAGE_MESH_BIT_EXT,
                                    .size = sizeof(TriggerPC) };
        auto add_blend = additive_blend();
        VkFormat r32 = VK_FORMAT_R32_SFLOAT;
        const vke::DescriptorLayout* tl[] = { &trace_layout };
        trace_pipeline = ctx.create_pipeline(vke::MeshPipelineCreateInfo{
            .mesh_shader = &trig_mesh, .fragment_shader = &lines_frag,
            .descriptor_layouts = tl, .push_constant_ranges = { &tpc, 1 },
            .color_attachment_formats = { &r32, 1 }, .cull_mode = VK_CULL_MODE_NONE,
            .blend_attachments = { &add_blend, 1 }, .debug_name = "stream_trace_pipeline" });

        // ---- Decay pass (fullscreen multiply by blend constant) ----
        auto fs_vert = ctx.create_shader({
            .stage = vke::ShaderStage::Vertex, .spirv_code = spv::histogram_sample_vert });
        auto decay_frag = ctx.create_shader({
            .stage = vke::ShaderStage::Fragment, .spirv_code = spv::decay_frag });
        auto dec_blend = decay_blend();
        decay_pipeline = ctx.create_pipeline(vke::GraphicsPipelineCreateInfo{
            .vertex_shader = &fs_vert, .fragment_shader = &decay_frag,
            .color_attachment_formats = { &r32, 1 }, .cull_mode = VK_CULL_MODE_NONE,
            .blend_attachments = { &dec_blend, 1 }, .dynamic_blend_constants = true,
            .debug_name = "stream_decay_pipeline" });

        // ---- Composite pass (accumulator -> target, log + Turbo) ----
        sampler = ctx.create_sampler({
            .mag_filter = VK_FILTER_NEAREST, .min_filter = VK_FILTER_NEAREST,
            .address_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .address_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE });
        vke::DescriptorBinding cb{ .binding = 0,
            .type = vke::DescriptorType::CombinedImageSampler,
            .stages = VK_SHADER_STAGE_FRAGMENT_BIT };
        tcomposite_layout = ctx.create_descriptor_layout({ .bindings = { &cb, 1 },
            .debug_name = "stream_tcomposite_layout" });
        tcomposite_set = tcomposite_layout.allocate_set("stream_tcomposite_set");

        auto comp_frag = ctx.create_shader({
            .stage = vke::ShaderStage::Fragment, .spirv_code = spv::histogram_sample_frag });
        vke::PushConstantRange cpc{ .stages = VK_SHADER_STAGE_FRAGMENT_BIT,
                                    .size = sizeof(TrigCompositePC) };
        const vke::DescriptorLayout* cl[] = { &tcomposite_layout };
        tcomposite_pipeline = ctx.create_pipeline(vke::GraphicsPipelineCreateInfo{
            .vertex_shader = &fs_vert, .fragment_shader = &comp_frag,
            .descriptor_layouts = cl, .push_constant_ranges = { &cpc, 1 },
            .color_attachment_formats = { &target_format, 1 },
            .cull_mode = VK_CULL_MODE_NONE, .debug_name = "stream_tcomposite_pipeline" });

        write_tcomposite_descriptor();
    }

    void write_trace_descriptor() {
        trace_set.write().bind_storage_buffer(0, ring.samples_buffer()).commit();
        trace_gen = ring.generation();
    }
    void write_tcomposite_descriptor() {
        tcomposite_set.write()
            .bind_combined_image_sampler(0, accumulator, sampler,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .commit();
    }

    void create_spectrum_pipeline() {
        // Spectrum traces are drawn from a per-block uploaded dBFS array (dynamic SSBO)
        // by the waveform mesh, additively into the shared accumulator.
        vke::DescriptorBinding b{ .binding = 0,
            .type = vke::DescriptorType::StorageBufferDynamic,
            .stages = VK_SHADER_STAGE_MESH_BIT_EXT };
        spec_layout = ctx.create_descriptor_layout({ .bindings = { &b, 1 },
            .max_sets = frames_in_flight + 1, .debug_name = "stream_spec_layout" });

        auto wf_mesh = ctx.create_shader({
            .stage = vke::ShaderStage::Mesh, .spirv_code = spv::waveform_renderer_mesh });
        auto lines_frag = ctx.create_shader({
            .stage = vke::ShaderStage::Fragment, .spirv_code = spv::lines_renderer_frag });
        vke::PushConstantRange pc{ .stages = VK_SHADER_STAGE_MESH_BIT_EXT, .size = sizeof(WaveformPC) };
        auto add_blend = additive_blend();
        VkFormat r32 = VK_FORMAT_R32_SFLOAT;
        const vke::DescriptorLayout* sl[] = { &spec_layout };
        spec_pipeline = ctx.create_pipeline(vke::MeshPipelineCreateInfo{
            .mesh_shader = &wf_mesh, .fragment_shader = &lines_frag,
            .descriptor_layouts = sl, .push_constant_ranges = { &pc, 1 },
            .color_attachment_formats = { &r32, 1 }, .cull_mode = VK_CULL_MODE_NONE,
            .blend_attachments = { &add_blend, 1 }, .debug_name = "stream_spec_pipeline" });

        // Allocate the upload slots ONCE, sized for the maximum FFT. The FFT itself is
        // CPU-side; only the bin count changes with size, so we never reallocate buffers
        // or descriptor sets (which previously churned — and exhausted — the pool).
        spec_stride = align_up(static_cast<VkDeviceSize>(MAX_SPEC_BINS) * sizeof(float), ssbo_align);
        VkDeviceSize total = spec_stride * SPEC_MAX_BLOCKS;
        spec_slots.resize(frames_in_flight);
        for (auto& s : spec_slots) {
            s.host = ctx.create_buffer({ .size = total, .usage = vke::BufferUsage::TransferSrc,
                .domain = vke::MemoryDomain::Host, .debug_name = "stream_spec_host" });
            s.device = ctx.create_buffer({ .size = total,
                .usage = vke::BufferUsage::Storage | vke::BufferUsage::TransferDst,
                .domain = vke::MemoryDomain::Device, .debug_name = "stream_spec_device" });
            s.set = spec_layout.allocate_set("stream_spec_set");
            s.set.write().bind_storage_buffer_dynamic(0, s.device, 0,
                static_cast<VkDeviceSize>(MAX_SPEC_BINS) * sizeof(float)).commit();
        }
        spec_slot_index = 0;
    }

    void barrier_accumulator_to_attachment(vke::CommandBuffer& cmd) {
        cmd.image_barrier({
            .image = accumulator, .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .src_stage_mask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                              VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .src_access_mask = VK_ACCESS_2_SHADER_READ_BIT |
                               VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dst_stage_mask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dst_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT });
    }

    // Composite the phosphor accumulator into `target` (log + Turbo). Shared by the
    // trigger and spectrum paths.
    void composite_accumulator(vke::CommandBuffer& cmd, vke::Image& target,
                               float max_intensity, float black_level) {
        cmd.image_barrier({
            .image = accumulator, .new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .src_stage_mask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .src_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dst_stage_mask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT });
        cmd.image_barrier({
            .image = target, .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .src_stage_mask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
            .src_access_mask = VK_ACCESS_2_NONE,
            .dst_stage_mask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dst_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT });
        vke::CommandBuffer::ColorAttachmentInfo tc[] = {{
            .image = &target, .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear_value = {.float32 = {0.f, 0.f, 0.f, 1.f}} }};
        cmd.begin_rendering({ .width = target.width(), .height = target.height(),
            .color_attachments = tc, .auto_layout_transitions = false });
        cmd.bind_pipeline(tcomposite_pipeline);
        cmd.set_viewport(0.f, 0.f, static_cast<float>(target.width()),
                         static_cast<float>(target.height()));
        cmd.set_scissor(0, 0, target.width(), target.height());
        vke::DescriptorSet csets[] = { tcomposite_set };
        cmd.bind_descriptor_sets(tcomposite_pipeline, csets);
        TrigCompositePC cpc{ .max_intensity = max_intensity, .black_level = black_level };
        cmd.push_constants(tcomposite_pipeline, VK_SHADER_STAGE_FRAGMENT_BIT, cpc);
        cmd.draw(3);
        cmd.end_rendering();
    }

    void reset_trigger_scan() {
        last_scanned = std::numeric_limits<int64_t>::min();
        armed = false; have_prev = false;
        next_allowed = std::numeric_limits<int64_t>::min();
        last_trigger_index = std::numeric_limits<int64_t>::min();
    }

    // Append the runs for one capture (trigger at sample `t`, sub-sample `frac`) into
    // `runs`, clipped to the visible x-range and to present spans. Returns false when a
    // per-frame budget is hit (caller should stop adding captures). Shared by the
    // trigger harvest and the auto free-run sweep.
    bool add_capture(int64_t t, float frac, const TriggerParams& p, int64_t L,
                     int64_t pre, int64_t head, uint32_t C, uint64_t& seg_total) {
        double cross = static_cast<double>(t - 1) + frac;
        int64_t vis_lo = static_cast<int64_t>(
            std::floor(cross - pre + static_cast<double>(p.x_min) * (L - 1))) - 1;
        int64_t vis_hi = static_cast<int64_t>(
            std::ceil (cross - pre + static_cast<double>(p.x_max) * (L - 1))) + 2;
        vis_lo = std::max(vis_lo, head - static_cast<int64_t>(C));
        vis_hi = std::min(vis_hi, head);
        int64_t a = vis_lo;
        while (a < vis_hi) {
            if (!ring.present_at(a)) { ++a; continue; }
            int64_t b = a;
            while (b < vis_hi && ring.present_at(b)) ++b;
            if (b - a >= 2) {
                uint64_t segs = static_cast<uint64_t>(b - a - 1);
                if (seg_total + segs > MAX_SEGMENTS || runs.size() >= MAX_RUNS_PER_FRAME)
                    return false;
                runs.push_back(Run{
                    .phys_start = static_cast<uint32_t>(((a % C) + C) % C),
                    .n_segments = static_cast<uint32_t>(segs),
                    .window_pos0 = static_cast<float>(static_cast<double>(a) - cross + pre) });
                seg_total += segs;
            }
            a = b;
        }
        return true;
    }

    // CPU trigger scan: harvest new events in the buffered window into `runs` (one run
    // per contiguous present span of each capture, clipped to the visible x-range).
    void harvest(const TriggerParams& p, int64_t L, int64_t pre, int64_t post,
                 int64_t holdoff, int64_t head, uint32_t C) {
        runs.clear();
        captures_last = 0;
        if (C == 0) return;

        const int64_t lo_avail = head - static_cast<int64_t>(C);
        if (last_scanned < lo_avail) reset_trigger_scan(); // fell behind → lost continuity

        int64_t t_hi = head - post;                         // exclusive
        int64_t t_lo = std::max(last_scanned, lo_avail + pre + 1);
        // Bound the scan span so a re-harvest never sweeps the whole ring (it ends at
        // t_hi, so the newest harvestable triggers are always covered regardless of L).
        t_lo = std::max(t_lo, t_hi - MAX_SCAN_SAMPLES);
        const bool  rising = (p.slope == TriggerSlope::Rising);
        const float level  = p.level, hyst = p.hysteresis;

        std::vector<std::pair<int64_t, float>> events;
        for (int64_t i = t_lo; i < t_hi; ++i) {
            if (!ring.present_at(i)) { armed = false; have_prev = false; continue; }
            float s = ring.sample_at(i);
            if (have_prev && prev_idx == i - 1 && i >= next_allowed) {
                bool cross = rising ? (armed && prev_s < level && s >= level)
                                    : (armed && prev_s > level && s <= level);
                if (cross) {
                    float frac = (p.equivalent_time && s != prev_s)
                        ? (level - prev_s) / (s - prev_s) : 0.0f;
                    events.emplace_back(i, frac);
                    armed = false;
                    next_allowed = i + holdoff;
                }
            }
            if (rising) { if (s < level - hyst) armed = true; }
            else        { if (s > level + hyst) armed = true; }
            prev_s = s; prev_idx = i; have_prev = true;
        }
        last_scanned = std::max(last_scanned, t_hi);

        if (events.size() > p.max_captures)
            events.erase(events.begin(), events.end() - p.max_captures);
        captures_last = events.size();
        if (!events.empty()) last_trigger_index = events.back().first; // newest

        // Build runs newest-capture-first and stop at the segment budget, so the most
        // recent captures are always drawn within a bounded per-frame cost.
        uint64_t seg_total = 0;
        for (auto it = events.rbegin(); it != events.rend(); ++it)
            if (!add_capture(it->first, it->second, p, L, pre, head, C, seg_total))
                break;
    }

    void clear_strip_immediate() {
        auto cmd = ctx.create_command_buffer();
        cmd.begin();
        vke::CommandBuffer::ColorAttachmentInfo c[] = {{
            .image = &strip, .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear_value = {.float32 = {0.f, 0.f, 0.f, 0.f}} }};
        cmd.begin_rendering({ .width = width, .height = height,
            .color_attachments = c, .auto_layout_transitions = true });
        cmd.end_rendering();
        cmd.image_barrier({
            .image = strip, .new_layout = VK_IMAGE_LAYOUT_GENERAL,
            .src_stage_mask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .src_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dst_stage_mask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dst_access_mask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT });
        cmd.end();
        ctx.submit_and_wait(std::move(cmd));
    }

    void clear_strip(vke::CommandBuffer& cmd) {
        cmd.image_barrier({
            .image = strip, .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .src_stage_mask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .src_access_mask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
            .dst_stage_mask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dst_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT });
        vke::CommandBuffer::ColorAttachmentInfo c[] = {{
            .image = &strip, .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .store_op = VK_ATTACHMENT_STORE_OP_STORE,
            .clear_value = {.float32 = {0.f, 0.f, 0.f, 0.f}} }};
        cmd.begin_rendering({ .width = width, .height = height,
            .color_attachments = c, .auto_layout_transitions = false });
        cmd.end_rendering();
        cmd.image_barrier({
            .image = strip, .new_layout = VK_IMAGE_LAYOUT_GENERAL,
            .src_stage_mask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .src_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dst_stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dst_access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                               VK_ACCESS_2_SHADER_STORAGE_READ_BIT });
    }
};

StreamScope::StreamScope(vke::Context& ctx, uint32_t width, uint32_t height,
                         VkFormat target_format, float sample_rate_hz,
                         float max_window_seconds, uint32_t frames_in_flight)
    : impl_(std::make_unique<Impl>(ctx, width, height, target_format, sample_rate_hz,
                                   max_window_seconds, frames_in_flight)) {}

StreamScope::~StreamScope()                                = default;
StreamScope::StreamScope(StreamScope&&) noexcept           = default;
StreamScope& StreamScope::operator=(StreamScope&&) noexcept = default;

uint32_t StreamScope::width()  const noexcept { return impl_->width; }
uint32_t StreamScope::height() const noexcept { return impl_->height; }
uint32_t StreamScope::samples_per_stripe() const noexcept { return impl_->cur_K; }
float    StreamScope::sample_rate() const noexcept { return impl_->ring.sample_rate(); }
uint64_t StreamScope::dropped_frames() const noexcept { return impl_->ring.dropped_frames(); }
size_t   StreamScope::captures_last_frame() const noexcept { return impl_->captures_last; }

StreamScope::AutosetResult StreamScope::autoset() const {
    auto& im = *impl_;
    AutosetResult r;
    const float   fs   = im.ring.sample_rate();
    const int64_t head = im.ring.head_index();
    const uint32_t C   = im.ring.capacity();
    if (fs <= 0.0f || head <= 0 || C == 0) return r;

    const int64_t M = std::min<int64_t>(im.ring.buffered_samples(), ANALYSIS_SAMPLES);
    if (M < 64) return r;
    const int64_t start = head - M;

    // Recent present samples (skip holes).
    std::vector<float> vals;
    vals.reserve(static_cast<size_t>(M));
    for (int64_t a = start; a < head; ++a)
        if (im.ring.present_at(a)) vals.push_back(im.ring.sample_at(a));
    if (vals.size() < 32) return r;
    const double coverage = static_cast<double>(vals.size()) / static_cast<double>(M);

    // Robust vertical extent (0.5–99.5 percentile) and 50% level.
    auto pct = [](std::vector<float>& v, double q) {
        size_t k = static_cast<size_t>(std::clamp(q * v.size(), 0.0, v.size() - 1.0));
        std::nth_element(v.begin(), v.begin() + k, v.end());
        return v[k];
    };
    std::vector<float> tmp = vals;
    float lo = pct(tmp, 0.005), hi = pct(tmp, 0.995);
    if (hi <= lo) { lo = *std::min_element(vals.begin(), vals.end());
                    hi = *std::max_element(vals.begin(), vals.end()); }
    const float pp = std::max(hi - lo, 1e-6f);
    const float level = 0.5f * (lo + hi);

    // Noise via second-difference MAD (cancels slope, robust to transients).
    std::vector<float> dd;
    dd.reserve(vals.size());
    for (size_t i = 1; i + 1 < vals.size(); ++i)
        dd.push_back(std::fabs(vals[i + 1] - 2.0f * vals[i] + vals[i - 1]));
    float noise = 0.0f;
    if (dd.size() >= 8) {
        size_t k = dd.size() / 2;
        std::nth_element(dd.begin(), dd.begin() + k, dd.end());
        noise = 1.4826f * dd[k] / std::sqrt(6.0f);
    }
    const float hyst = std::clamp(4.0f * noise, 0.01f * pp, 0.25f * pp);

    const float margin = 0.1f * pp;
    r.y_min = lo - margin; r.y_max = hi + margin;
    r.level = level; r.hysteresis = hyst;
    r.ok = true;

    // Timebase from the median rising-crossing interval (only when coverage is dense).
    if (coverage > 0.8) {
        std::vector<int64_t> crossings;
        bool armed = false, havep = false; float ps = 0.0f; int64_t pidx = 0;
        for (int64_t a = start; a < head; ++a) {
            if (!im.ring.present_at(a)) { armed = false; havep = false; continue; }
            float s = im.ring.sample_at(a);
            if (havep && pidx == a - 1 && armed && ps < level && s >= level) {
                crossings.push_back(a); armed = false;
            }
            if (s < level - hyst) armed = true;
            ps = s; pidx = a; havep = true;
        }
        if (crossings.size() >= 3) {
            std::vector<int64_t> iv;
            for (size_t i = 1; i < crossings.size(); ++i)
                iv.push_back(crossings[i] - crossings[i - 1]);
            size_t k = iv.size() / 2;
            std::nth_element(iv.begin(), iv.begin() + k, iv.end());
            if (iv[k] > 0) {
                r.window_seconds = static_cast<float>(AUTOSET_PERIODS *
                    static_cast<double>(iv[k]) / fs);
                r.has_timebase = true;
            }
        }
    }
    return r;
}

float StreamScope::buffered_seconds() const noexcept {
    auto& im = *impl_;
    float fs = im.ring.sample_rate();
    if (fs <= 0.0f) return 0.0f;
    return static_cast<float>(im.ring.buffered_samples()) / fs;
}
float StreamScope::capacity_seconds() const noexcept {
    auto& im = *impl_;
    float fs = im.ring.sample_rate();
    if (fs <= 0.0f) return 0.0f;
    return static_cast<float>(im.ring.capacity()) / fs;
}
float StreamScope::shown_seconds() const noexcept {
    auto& im = *impl_;
    float fs = im.ring.sample_rate();
    if (fs <= 0.0f || im.cur_K == 0) return 0.0f;
    uint64_t avail_cols = static_cast<uint64_t>(im.ring.head_index()) / im.cur_K;
    uint64_t shown_cols = std::min<uint64_t>(avail_cols, im.width);
    return static_cast<float>(shown_cols * im.cur_K) / fs;
}

void StreamScope::reset() {
    auto& im = *impl_;
    im.ring.reset();
    im.cols_emitted_ = 0;
    im.needs_replay  = true;
    im.reset_trigger_scan();
    im.spec_scanned = std::numeric_limits<int64_t>::min();
    im.needs_clear_persist = true;
}

void StreamScope::reset_persistence() {
    impl_->reset_trigger_scan();
    impl_->spec_scanned = std::numeric_limits<int64_t>::min();
    impl_->needs_clear_persist = true;
}

void StreamScope::set_sample_rate(float hz) {
    auto& im = *impl_;
    if (hz == im.ring.sample_rate() || hz <= 0.0f) return;
    im.ring.set_sample_rate(hz);
    im.cols_emitted_ = 0;
    im.needs_replay  = true;
    im.reset_trigger_scan();
    im.spec_scanned = std::numeric_limits<int64_t>::min();
    im.needs_clear_persist = true;
}

void StreamScope::resize(uint32_t width, uint32_t height) {
    auto& im = *impl_;
    if (width == im.width && height == im.height) return;
    im.width = width;
    im.height = height;
    im.create_strip();
    im.create_accumulator();
    im.write_composite_descriptor();   // roll strip composite (storage image)
    im.write_tcomposite_descriptor();  // trigger composite (accumulator)
    im.ring_gen  = ~0ull;              // force reduce rebind (picks up new strip)
    im.trace_gen = ~0ull;
    im.needs_replay = true;
    im.clear_strip_immediate();
}

void StreamScope::ingest(const RingFrame& f) { impl_->ring.ingest(f); }

void StreamScope::render_roll(vke::CommandBuffer& cmd, vke::Image& target,
                              const RollParams& params) {
    auto& im = *impl_;
    const uint32_t W = im.width;
    const uint32_t H = im.height;

    // Upload newly-ingested samples + presence into the ring.
    im.ring.record_upload(cmd);

    const uint32_t C  = im.ring.capacity();
    const uint32_t S  = im.ring.samples_per_frame();
    const float    fs = im.ring.sample_rate();
    const int64_t  head = im.ring.head_index();

    // The ring may have (re)created its buffers (reconfigure); rebind once configured.
    if (C != 0 && im.ring.generation() != im.ring_gen) {
        im.write_reduce_descriptor();
        im.needs_replay = true;
    }

    // ---- Samples-per-stripe (K) from the window; replay if it changed ----
    uint32_t K = (C == 0) ? 1u : static_cast<uint32_t>(std::lround(
        static_cast<double>(params.window_seconds) * fs / std::max(1u, W)));
    K = std::max(1u, K);
    if (K != im.cur_K) { im.cur_K = K; im.needs_replay = true; }

    // Columns available off the absolute timeline (column c needs c*K+K+1 samples,
    // the +1 being the bridge sample to the next column).
    uint64_t cols_available = (head >= static_cast<int64_t>(K) + 1)
        ? static_cast<uint64_t>(head - 1) / K : 0;

    // ---- Replay: clear the strip and rewind to re-render recent history ----
    if (im.needs_replay && C != 0) {
        im.clear_strip(cmd);
        uint64_t want = std::min<uint64_t>({ static_cast<uint64_t>(W),
                                             C / K, cols_available });
        im.cols_emitted_ = cols_available - want;
        im.needs_replay  = false;
    }

    // ---- New whole columns available, bounded by ring + strip width ----
    uint64_t n_new64 = (cols_available > im.cols_emitted_)
        ? cols_available - im.cols_emitted_ : 0;
    uint64_t max_safe = (K == 0) ? 0 : C / K;
    if (n_new64 > max_safe) { im.cols_emitted_ += n_new64 - max_safe; n_new64 = max_safe; }
    if (n_new64 > W)        { im.cols_emitted_ += n_new64 - W;        n_new64 = W; }
    uint32_t n_new = static_cast<uint32_t>(n_new64);

    // ---- Reduce: write new columns into the circular strip ----
    if (n_new > 0 && C != 0 && S != 0) {
        cmd.image_barrier({
            .image = im.strip, .new_layout = VK_IMAGE_LAYOUT_GENERAL,
            .src_stage_mask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .src_access_mask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            .dst_stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dst_access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT });

        StreamReducePC pc{
            .ring_capacity = C, .strip_width = W, .height = H, .samples_per_stripe = K,
            .ring_start_phys = static_cast<uint32_t>((im.cols_emitted_ * K) % C),
            .col_start_phys  = static_cast<uint32_t>(im.cols_emitted_ % W),
            .y_min = params.y_min, .y_max = params.y_max,
            .line_width_px = params.line_width_px, .weight_scale = WEIGHT_SCALE,
            .min_weight = params.min_weight, .density = params.density,
            .interp = std::max(1u, params.interp), .samples_per_frame = S };

        cmd.bind_pipeline(im.reduce_pipeline);
        vke::DescriptorSet sets[] = { im.reduce_set };
        cmd.bind_descriptor_sets(im.reduce_pipeline, sets);
        cmd.push_constants(im.reduce_pipeline, VK_SHADER_STAGE_COMPUTE_BIT, pc);
        cmd.dispatch(n_new, 1, 1);

        im.cols_emitted_ += n_new;
    }

    // ---- Composite the circular strip into the target ----
    cmd.image_barrier({
        .image = im.strip, .new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .src_stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .src_access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .dst_stage_mask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT });
    cmd.image_barrier({
        .image = target, .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .src_stage_mask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
        .src_access_mask = VK_ACCESS_2_NONE,
        .dst_stage_mask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dst_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT });

    vke::CommandBuffer::ColorAttachmentInfo tc[] = {{
        .image = &target, .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .store_op = VK_ATTACHMENT_STORE_OP_STORE,
        .clear_value = {.float32 = {0.f, 0.f, 0.f, 1.f}} }};
    cmd.begin_rendering({ .width = target.width(), .height = target.height(),
        .color_attachments = tc, .auto_layout_transitions = false });
    cmd.bind_pipeline(im.composite_pipeline);
    cmd.set_viewport(0.f, 0.f, static_cast<float>(target.width()),
                     static_cast<float>(target.height()));
    cmd.set_scissor(0, 0, target.width(), target.height());
    vke::DescriptorSet csets[] = { im.composite_set };
    cmd.bind_descriptor_sets(im.composite_pipeline, csets);
    StreamCompositePC cpc{ .max_intensity = params.max_intensity,
                           .black_level = params.black_level,
                           .head_col = static_cast<uint32_t>(im.cols_emitted_ % W),
                           .strip_width = W };
    cmd.push_constants(im.composite_pipeline, VK_SHADER_STAGE_FRAGMENT_BIT, cpc);
    cmd.draw(3);
    cmd.end_rendering();
}

void StreamScope::render_trigger(vke::CommandBuffer& cmd, vke::Image& target,
                                 const TriggerParams& params) {
    auto& im = *impl_;
    const uint32_t W = im.width, H = im.height;
    const float    aspect = static_cast<float>(H) / static_cast<float>(W);

    im.ring.record_upload(cmd);

    const uint32_t C    = im.ring.capacity();
    const float    fs   = im.ring.sample_rate();
    const int64_t  head = im.ring.head_index();

    // Rebind the ring buffer if it was recreated (reconfigure).
    if (C != 0 && im.ring.generation() != im.trace_gen) {
        im.write_trace_descriptor();
        im.needs_clear_persist = true;
        im.reset_trigger_scan();
    }

    // Alignment-affecting param change → clear persistence and re-harvest history.
    const bool changed = !im.have_prev_params ||
        im.p_level != params.level || im.p_hyst != params.hysteresis ||
        im.p_win != params.window_seconds || im.p_xmin != params.x_min ||
        im.p_xmax != params.x_max || im.p_ymin != params.y_min ||
        im.p_ymax != params.y_max || im.p_holdoff != params.holdoff_seconds ||
        im.p_slope != static_cast<int>(params.slope) || im.p_eqt != params.equivalent_time ||
        im.p_prefrac != params.pre_frac;
    if (changed) {
        im.needs_clear_persist = true;
        im.reset_trigger_scan();
        im.have_prev_params = true;
        im.p_level = params.level; im.p_hyst = params.hysteresis;
        im.p_win = params.window_seconds; im.p_xmin = params.x_min; im.p_xmax = params.x_max;
        im.p_ymin = params.y_min; im.p_ymax = params.y_max;
        im.p_holdoff = params.holdoff_seconds; im.p_prefrac = params.pre_frac;
        im.p_slope = static_cast<int>(params.slope); im.p_eqt = params.equivalent_time;
    }

    // Timebase geometry (trigger at pre_frac of the window).
    const int64_t L   = (fs > 0.0f)
        ? std::max<int64_t>(2, std::lround(static_cast<double>(params.window_seconds) * fs)) : 2;
    const int64_t pre = std::clamp<int64_t>(
        std::lround(static_cast<double>(params.pre_frac) * (L - 1)), 1, L - 1);
    const int64_t post = L - pre;
    const int64_t holdoff = (params.holdoff_seconds > 0.0f)
        ? std::max<int64_t>(1, std::lround(static_cast<double>(params.holdoff_seconds) * fs)) : L;

    // ---- Harvest new trigger captures into im.runs (CPU) ----
    im.runs.clear();
    im.captures_last = 0;
    if (C != 0 && fs > 0.0f && head > L)
        im.harvest(params, L, pre, post, holdoff, head, C);

    // Auto mode: no trigger within the timeout → draw an untriggered free-run sweep at
    // the head so the waveform stays visible (e.g. while placing the level).
    if (params.auto_mode && im.runs.empty() && C != 0 && fs > 0.0f && head > L) {
        int64_t timeout = std::max<int64_t>(1,
            std::lround(static_cast<double>(params.auto_timeout_seconds) * fs));
        bool stale = im.last_trigger_index == std::numeric_limits<int64_t>::min()
                  || head - im.last_trigger_index > timeout;
        if (stale) {
            uint64_t seg = 0;
            im.add_capture(head - post, 0.0f, params, L, pre, head, C, seg);
        }
    }

    // ---- Accumulator pass: decay (or clear) then additively draw the captures ----
    im.barrier_accumulator_to_attachment(cmd);

    vke::CommandBuffer::ColorAttachmentInfo acc_color[] = {{
        .image = &im.accumulator,
        .load_op = im.needs_clear_persist ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                          : VK_ATTACHMENT_LOAD_OP_LOAD,
        .store_op = VK_ATTACHMENT_STORE_OP_STORE,
        .clear_value = {.float32 = {0.f, 0.f, 0.f, 0.f}} }};
    cmd.begin_rendering({ .width = W, .height = H,
        .color_attachments = acc_color, .auto_layout_transitions = false });
    cmd.set_viewport(0.f, 0.f, static_cast<float>(W), static_cast<float>(H));
    cmd.set_scissor(0, 0, W, H);

    if (!im.needs_clear_persist) {
        cmd.bind_pipeline(im.decay_pipeline);
        cmd.set_blend_constants(params.decay_alpha, params.decay_alpha,
                                params.decay_alpha, params.decay_alpha);
        cmd.draw(3);
    }
    im.needs_clear_persist = false;

    if (!im.runs.empty()) {
        cmd.bind_pipeline(im.trace_pipeline);
        vke::DescriptorSet sets[] = { im.trace_set };
        cmd.bind_descriptor_sets(im.trace_pipeline, sets);
        constexpr uint32_t WG = 32;
        for (const auto& r : im.runs) {
            TriggerPC pc{
                .x_range = { params.x_min, params.x_max },
                .y_range = { params.y_min, params.y_max },
                .viewport_ratio = { aspect, 1.0f / aspect },
                .line_width = params.line_width,
                .ring_capacity = C, .run_phys_start = r.phys_start,
                .n_segments = r.n_segments, .viewport_width = W,
                .min_weight = params.min_weight,
                .window_len = static_cast<float>(L), .window_pos0 = r.window_pos0 };
            cmd.push_constants(im.trace_pipeline, VK_SHADER_STAGE_MESH_BIT_EXT, pc);
            cmd.draw_mesh_tasks((r.n_segments + WG - 1) / WG, 1, 1);
        }
    }
    cmd.end_rendering();

    // ---- Composite accumulator -> target (log + Turbo) ----
    im.composite_accumulator(cmd, target, params.max_intensity, params.black_level);
}

void StreamScope::render_spectrum(vke::CommandBuffer& cmd, vke::Image& target,
                                  const SpectrumParams& params) {
    auto& im = *impl_;
    const uint32_t W = im.width, H = im.height;
    const float    aspect = static_cast<float>(H) / static_cast<float>(W);
    const float    fs   = im.ring.sample_rate();
    const int64_t  head = im.ring.head_index();
    const uint32_t C    = im.ring.capacity();
    const uint32_t N    = std::clamp(params.fft_size, 2u, MAX_SPEC_FFT);

    // (Re)configure the FFT when size/window changes. The upload slots are fixed-size
    // (allocated once for MAX_SPEC_FFT), so only the CPU FFT + bin count change here.
    const bool cfg_changed = im.sp_fft != N || im.sp_win != static_cast<int>(params.window);
    if (cfg_changed) { im.fft.configure(N, params.window); im.spec_bins = im.fft.bins(); }

    // Any alignment-affecting change → clear persistence and rescan.
    const bool changed = !im.have_spec_params || cfg_changed ||
        im.sp_xmin != params.x_min || im.sp_xmax != params.x_max ||
        im.sp_dbf != params.db_floor || im.sp_dbt != params.db_top;
    if (changed) {
        im.needs_clear_persist = true;
        im.spec_scanned = std::numeric_limits<int64_t>::min();
        im.have_spec_params = true;
        im.sp_fft = N; im.sp_win = static_cast<int>(params.window);
        im.sp_xmin = params.x_min; im.sp_xmax = params.x_max;
        im.sp_dbf = params.db_floor; im.sp_dbt = params.db_top;
    }

    // ---- Harvest FFT blocks from contiguous present data (CPU) ----
    const uint32_t bins = im.fft.bins();
    im.spec_pending = 0;
    im.spec_scratch.clear();
    if (C >= N && fs > 0.0f && head >= static_cast<int64_t>(N)) {
        if (im.spec_scanned == std::numeric_limits<int64_t>::min() ||
            im.spec_scanned < head - static_cast<int64_t>(C))
            im.spec_scanned = head - static_cast<int64_t>(N);   // most recent block
        std::vector<float> block(N);
        while (im.spec_pending < SPEC_MAX_BLOCKS &&
               im.spec_scanned + static_cast<int64_t>(N) <= head) {
            int64_t p = im.spec_scanned, hole = -1;
            for (int64_t a = p; a < p + static_cast<int64_t>(N); ++a)
                if (!im.ring.present_at(a)) { hole = a; break; }
            if (hole < 0) {
                for (uint32_t i = 0; i < N; ++i) block[i] = im.ring.sample_at(p + i);
                size_t off = im.spec_scratch.size();
                im.spec_scratch.resize(off + bins);
                im.fft.process(block, std::span<float>(im.spec_scratch.data() + off, bins));
                ++im.spec_pending;
                im.spec_scanned = p + static_cast<int64_t>(N);  // non-overlapping
            } else {
                im.spec_scanned = hole + 1;                     // skip past the gap
            }
        }
    }

    // ---- Upload this frame's blocks ----
    Impl::SpecSlot* slot = nullptr;
    if (im.spec_pending > 0) {
        slot = &im.spec_slots[im.spec_slot_index];
        im.spec_slot_index = (im.spec_slot_index + 1) % im.frames_in_flight;
        auto host = slot->host.template mapped_as<std::byte>();
        for (uint32_t k = 0; k < im.spec_pending; ++k)
            std::memcpy(host.data() + k * im.spec_stride,
                        im.spec_scratch.data() + static_cast<size_t>(k) * bins,
                        static_cast<size_t>(bins) * sizeof(float));
        cmd.copy_buffer(slot->host, slot->device, 0, 0, im.spec_stride * im.spec_pending);
        cmd.buffer_barrier({ .buffer = slot->device,
            .src_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .src_access_mask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dst_stage_mask = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            .dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT });
    }

    // ---- Accumulator pass: decay (or clear) then additively draw the spectra ----
    im.barrier_accumulator_to_attachment(cmd);
    vke::CommandBuffer::ColorAttachmentInfo acc[] = {{
        .image = &im.accumulator,
        .load_op = im.needs_clear_persist ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                          : VK_ATTACHMENT_LOAD_OP_LOAD,
        .store_op = VK_ATTACHMENT_STORE_OP_STORE,
        .clear_value = {.float32 = {0.f, 0.f, 0.f, 0.f}} }};
    cmd.begin_rendering({ .width = W, .height = H,
        .color_attachments = acc, .auto_layout_transitions = false });
    cmd.set_viewport(0.f, 0.f, static_cast<float>(W), static_cast<float>(H));
    cmd.set_scissor(0, 0, W, H);
    if (!im.needs_clear_persist) {
        cmd.bind_pipeline(im.decay_pipeline);
        cmd.set_blend_constants(params.decay_alpha, params.decay_alpha,
                                params.decay_alpha, params.decay_alpha);
        cmd.draw(3);
    }
    im.needs_clear_persist = false;
    if (slot && bins >= 2) {
        cmd.bind_pipeline(im.spec_pipeline);
        WaveformPC pc{
            .x_range = { params.x_min, params.x_max },
            .y_range = { params.db_floor, params.db_top },
            .viewport_ratio = { aspect, 1.0f / aspect },
            .line_width = params.line_width,
            .n_samples = bins, .first_segment = 0, .n_segments = bins - 1,
            .viewport_width = W, .min_weight = params.min_weight };
        uint32_t groups = (bins - 1 + 31) / 32;
        for (uint32_t k = 0; k < im.spec_pending; ++k) {
            uint32_t dyn = static_cast<uint32_t>(k * im.spec_stride);
            vke::DescriptorSet sets[] = { slot->set };
            std::span<const uint32_t> offs(&dyn, 1);
            cmd.bind_descriptor_sets(im.spec_pipeline, sets, offs);
            cmd.push_constants(im.spec_pipeline, VK_SHADER_STAGE_MESH_BIT_EXT, pc);
            cmd.draw_mesh_tasks(groups, 1, 1);
        }
    }
    cmd.end_rendering();

    // ---- Composite accumulator -> target (log + Turbo) ----
    im.composite_accumulator(cmd, target, params.max_intensity, params.black_level);
}

} // namespace plot
