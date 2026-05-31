#include <plot/waveform_scope.hpp>
#include <vke/vke.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <cstring>
#include <vector>

#include "shaders/waveform_renderer_mesh_spv.hpp"
#include "shaders/lines_renderer_frag_spv.hpp"
#include "shaders/histogram_sample_vert_spv.hpp"
#include "shaders/histogram_sample_frag_spv.hpp"
#include "shaders/decay_frag_spv.hpp"

namespace plot {

// Matches waveform_renderer.mesh (sizeof == 48).
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

// Matches histogram_sample.frag.
struct CompositePC {
    float max_intensity;
    float black_level;
};

namespace {

constexpr uint32_t MAX_CHUNKS_PER_FRAME = 256;

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

// hist = 0*src + alpha*dst, where alpha is the dynamic blend constant.
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

VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) {
    return a == 0 ? v : ((v + a - 1) / a) * a;
}

} // namespace

struct WaveformScope::Impl {
    vke::Context& ctx;
    uint32_t      width;
    uint32_t      height;
    VkFormat      target_format;
    uint32_t      frames_in_flight;
    VkDeviceSize  ssbo_align;

    vke::Image accumulator;  // persistent R32_SFLOAT phosphor buffer
    bool       needs_clear = true;

    // Trace pass (waveform mesh, additive) — dynamic storage buffer so multiple
    // chunks share one buffer and are selected by a per-draw dynamic offset.
    vke::DescriptorLayout trace_layout;
    vke::Pipeline         trace_pipeline;

    // Decay pass (fullscreen multiply via blend constant).
    vke::Pipeline decay_pipeline;

    // Composite pass (accumulator -> target, log + Turbo).
    vke::Sampler          sampler;
    vke::DescriptorLayout composite_layout;
    vke::DescriptorSet    composite_set;
    vke::Pipeline         composite_pipeline;

    // Per-frame-in-flight upload slot.
    struct Slot {
        vke::Buffer        host;    // host-visible, persistently mapped
        vke::Buffer        device;  // device-local storage
        vke::DescriptorSet set;     // bound (dynamic) to `device`
    };
    std::vector<Slot> slots;
    uint32_t          slot_index = 0;

    // Geometry of the current chunk set.
    uint32_t     chunk_len = 0;
    VkDeviceSize stride    = 0;     // per-chunk byte stride (aligned)

    // CPU-side pending traces (flattened, chunk_len floats each).
    std::vector<float> staging;
    uint32_t           pending_count = 0;

    Impl(vke::Context& c, uint32_t w, uint32_t h, VkFormat fmt, uint32_t fif)
        : ctx(c), width(w), height(h), target_format(fmt),
          frames_in_flight(std::max(1u, fif))
    {
        ssbo_align = ctx.physical_device_info().limits.minStorageBufferOffsetAlignment;
        create_accumulator();
        create_pipelines();
        slots.resize(frames_in_flight);
    }

    void create_accumulator() {
        accumulator = ctx.create_image({
            .width      = width,
            .height     = height,
            .format     = VK_FORMAT_R32_SFLOAT,
            .usage      = vke::ImageUsage::ColorAttachment | vke::ImageUsage::Sampled |
                          vke::ImageUsage::TransferSrc,
            .debug_name = "scope_accumulator",
        });
        needs_clear = true;
    }

    void create_pipelines() {
        // ---- Trace pipeline ----
        vke::DescriptorBinding trace_binding{
            .binding = 0,
            .type    = vke::DescriptorType::StorageBufferDynamic,
            .stages  = VK_SHADER_STAGE_MESH_BIT_EXT,
        };
        trace_layout = ctx.create_descriptor_layout({
            .bindings   = { &trace_binding, 1 },
            .max_sets   = frames_in_flight + 1,
            .debug_name = "scope_trace_layout",
        });

        auto wf_mesh = ctx.create_shader({
            .stage = vke::ShaderStage::Mesh, .spirv_code = spv::waveform_renderer_mesh });
        auto lines_frag = ctx.create_shader({
            .stage = vke::ShaderStage::Fragment, .spirv_code = spv::lines_renderer_frag });

        vke::PushConstantRange wf_pc{ .stages = VK_SHADER_STAGE_MESH_BIT_EXT,
                                      .size = sizeof(WaveformPC) };
        auto add_blend = additive_blend();
        VkFormat r32 = VK_FORMAT_R32_SFLOAT;
        const vke::DescriptorLayout* trace_layouts[] = { &trace_layout };
        trace_pipeline = ctx.create_pipeline(vke::MeshPipelineCreateInfo{
            .mesh_shader              = &wf_mesh,
            .fragment_shader          = &lines_frag,
            .descriptor_layouts       = trace_layouts,
            .push_constant_ranges     = { &wf_pc, 1 },
            .color_attachment_formats = { &r32, 1 },
            .cull_mode                = VK_CULL_MODE_NONE,
            .blend_attachments        = { &add_blend, 1 },
            .debug_name               = "scope_trace_pipeline",
        });

        // ---- Decay pipeline (fullscreen multiply) ----
        auto fs_vert = ctx.create_shader({
            .stage = vke::ShaderStage::Vertex, .spirv_code = spv::histogram_sample_vert });
        auto decay_frag = ctx.create_shader({
            .stage = vke::ShaderStage::Fragment, .spirv_code = spv::decay_frag });
        auto dec_blend = decay_blend();
        decay_pipeline = ctx.create_pipeline(vke::GraphicsPipelineCreateInfo{
            .vertex_shader            = &fs_vert,
            .fragment_shader          = &decay_frag,
            .color_attachment_formats = { &r32, 1 },
            .cull_mode                = VK_CULL_MODE_NONE,
            .blend_attachments        = { &dec_blend, 1 },
            .dynamic_blend_constants  = true,
            .debug_name               = "scope_decay_pipeline",
        });

        // ---- Composite pipeline (accumulator -> target) ----
        sampler = ctx.create_sampler({
            .mag_filter = VK_FILTER_NEAREST,
            .min_filter = VK_FILTER_NEAREST,
            .address_u  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .address_v  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        });
        vke::DescriptorBinding comp_binding{
            .binding = 0,
            .type    = vke::DescriptorType::CombinedImageSampler,
            .stages  = VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        composite_layout = ctx.create_descriptor_layout({
            .bindings = { &comp_binding, 1 }, .debug_name = "scope_composite_layout" });
        composite_set = composite_layout.allocate_set("scope_composite_set");

        auto comp_frag = ctx.create_shader({
            .stage = vke::ShaderStage::Fragment, .spirv_code = spv::histogram_sample_frag });
        vke::PushConstantRange comp_pc{ .stages = VK_SHADER_STAGE_FRAGMENT_BIT,
                                        .size = sizeof(CompositePC) };
        const vke::DescriptorLayout* comp_layouts[] = { &composite_layout };
        composite_pipeline = ctx.create_pipeline(vke::GraphicsPipelineCreateInfo{
            .vertex_shader            = &fs_vert,
            .fragment_shader          = &comp_frag,
            .descriptor_layouts       = comp_layouts,
            .push_constant_ranges     = { &comp_pc, 1 },
            .color_attachment_formats = { &target_format, 1 },
            .cull_mode                = VK_CULL_MODE_NONE,
            .debug_name               = "scope_composite_pipeline",
        });

        write_composite_descriptor();
    }

    void write_composite_descriptor() {
        composite_set.write()
            .bind_combined_image_sampler(0, accumulator, sampler,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .commit();
    }

    // (Re)allocate the per-slot upload buffers once the chunk length is known.
    void ensure_slot_buffers(uint32_t len) {
        if (len == chunk_len && stride != 0) return;
        chunk_len = len;
        stride    = align_up(static_cast<VkDeviceSize>(len) * sizeof(float), ssbo_align);
        VkDeviceSize total = stride * MAX_CHUNKS_PER_FRAME;

        for (auto& slot : slots) {
            slot.host = ctx.create_buffer({
                .size = total, .usage = vke::BufferUsage::TransferSrc,
                .domain = vke::MemoryDomain::Host, .debug_name = "scope_upload_host" });
            slot.device = ctx.create_buffer({
                .size = total,
                .usage = vke::BufferUsage::Storage | vke::BufferUsage::TransferDst,
                .domain = vke::MemoryDomain::Device, .debug_name = "scope_upload_device" });
            slot.set = trace_layout.allocate_set("scope_trace_set");
            slot.set.write()
                .bind_storage_buffer_dynamic(0, slot.device, 0,
                                             static_cast<VkDeviceSize>(len) * sizeof(float))
                .commit();
        }
    }

    void barrier_accumulator_to_attachment(vke::CommandBuffer& cmd) {
        cmd.image_barrier({
            .image           = accumulator,
            .new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .src_stage_mask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .src_access_mask = VK_ACCESS_2_SHADER_READ_BIT |
                               VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dst_stage_mask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dst_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        });
    }
};

WaveformScope::WaveformScope(vke::Context& ctx, uint32_t width, uint32_t height,
                             VkFormat target_format, uint32_t frames_in_flight)
    : impl_(std::make_unique<Impl>(ctx, width, height, target_format, frames_in_flight)) {}

WaveformScope::~WaveformScope()                                  = default;
WaveformScope::WaveformScope(WaveformScope&&) noexcept           = default;
WaveformScope& WaveformScope::operator=(WaveformScope&&) noexcept = default;

uint32_t WaveformScope::width()  const noexcept { return impl_->width; }
uint32_t WaveformScope::height() const noexcept { return impl_->height; }

void WaveformScope::resize(uint32_t width, uint32_t height) {
    auto& im = *impl_;
    if (width == im.width && height == im.height) return;
    im.width = width;
    im.height = height;
    im.create_accumulator();
    im.write_composite_descriptor();  // descriptor points at the new accumulator
}

void WaveformScope::reset_persistence() {
    impl_->needs_clear = true;
}

void WaveformScope::push_chunk(std::span<const float> samples) {
    auto& im = *impl_;
    if (samples.empty()) return;
    uint32_t len = static_cast<uint32_t>(samples.size());

    if (im.chunk_len != 0 && len != im.chunk_len)
        return; // ignore mismatched-length chunks (single record length per session)

    if (im.pending_count >= MAX_CHUNKS_PER_FRAME) {
        // Drop the oldest pending trace to bound work per frame.
        im.staging.erase(im.staging.begin(), im.staging.begin() + len);
        --im.pending_count;
    }
    im.staging.insert(im.staging.end(), samples.begin(), samples.end());
    ++im.pending_count;
}

size_t WaveformScope::pending() const { return impl_->pending_count; }

void WaveformScope::render(vke::CommandBuffer& cmd, vke::Image& target,
                          const ScopeParams& params) {
    auto& im = *impl_;
    const float aspect = static_cast<float>(im.height) / static_cast<float>(im.width);

    // ---- 1. Upload pending traces (outside any render pass) ----
    Impl::Slot* slot = nullptr;
    uint32_t draw_count = 0;
    if (im.pending_count > 0) {
        uint32_t len = static_cast<uint32_t>(im.staging.size()) / im.pending_count;
        im.ensure_slot_buffers(len);

        slot       = &im.slots[im.slot_index];
        im.slot_index = (im.slot_index + 1) % im.frames_in_flight;
        draw_count = im.pending_count;

        auto host = slot->host.mapped_as<std::byte>();
        for (uint32_t k = 0; k < draw_count; ++k) {
            std::memcpy(host.data() + k * im.stride,
                        im.staging.data() + static_cast<size_t>(k) * len,
                        static_cast<size_t>(len) * sizeof(float));
        }
        cmd.copy_buffer(slot->host, slot->device, 0, 0, im.stride * draw_count);
        cmd.buffer_barrier({
            .buffer          = slot->device,
            .src_stage_mask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .src_access_mask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dst_stage_mask  = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            .dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT,
        });
    }

    // ---- 2. Accumulator pass: decay (or clear) then add new traces ----
    im.barrier_accumulator_to_attachment(cmd);

    vke::CommandBuffer::ColorAttachmentInfo acc_color[] = {{
        .image       = &im.accumulator,
        .load_op     = im.needs_clear ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                      : VK_ATTACHMENT_LOAD_OP_LOAD,
        .store_op    = VK_ATTACHMENT_STORE_OP_STORE,
        .clear_value = {.float32 = {0.f, 0.f, 0.f, 0.f}},
    }};
    cmd.begin_rendering({
        .width = im.width, .height = im.height,
        .color_attachments = acc_color, .auto_layout_transitions = false,
    });
    cmd.set_viewport(0.f, 0.f, static_cast<float>(im.width), static_cast<float>(im.height));
    cmd.set_scissor(0, 0, im.width, im.height);

    if (!im.needs_clear) {
        // Multiply the whole accumulator by the decay factor.
        cmd.bind_pipeline(im.decay_pipeline);
        cmd.set_blend_constants(params.decay_alpha, params.decay_alpha,
                                params.decay_alpha, params.decay_alpha);
        cmd.draw(3);
    }
    im.needs_clear = false;

    if (draw_count > 0) {
        cmd.bind_pipeline(im.trace_pipeline);
        WaveformPC pc{
            .x_range        = { params.x_min, params.x_max },
            .y_range        = { params.y_min, params.y_max },
            .viewport_ratio = { aspect, 1.0f / aspect },
            .line_width     = params.line_width,
            .n_samples      = im.chunk_len,
            .first_segment  = 0,
            .n_segments     = im.chunk_len - 1,
            .viewport_width = im.width,
            .min_weight     = params.min_weight,
        };
        constexpr uint32_t WG = 32;
        uint32_t groups_x = (im.chunk_len - 1 + WG - 1) / WG;
        for (uint32_t k = 0; k < draw_count; ++k) {
            uint32_t dyn = static_cast<uint32_t>(k * im.stride);
            vke::DescriptorSet sets[] = { slot->set };
            std::span<const uint32_t> offsets(&dyn, 1);
            cmd.bind_descriptor_sets(im.trace_pipeline, sets, offsets);
            cmd.push_constants(im.trace_pipeline, VK_SHADER_STAGE_MESH_BIT_EXT, pc);
            cmd.draw_mesh_tasks(groups_x, 1, 1);
        }
    }
    cmd.end_rendering();

    im.staging.clear();
    im.pending_count = 0;

    // ---- 3. Composite accumulator -> target (log + Turbo) ----
    cmd.image_barrier({
        .image           = im.accumulator,
        .new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .src_stage_mask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .src_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .dst_stage_mask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT,
    });
    cmd.image_barrier({
        .image           = target,
        .new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .src_stage_mask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
        .src_access_mask = VK_ACCESS_2_NONE,
        .dst_stage_mask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dst_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
    });

    vke::CommandBuffer::ColorAttachmentInfo tgt_color[] = {{
        .image       = &target,
        .load_op     = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .store_op    = VK_ATTACHMENT_STORE_OP_STORE,
        .clear_value = {.float32 = {0.f, 0.f, 0.f, 1.f}},
    }};
    cmd.begin_rendering({
        .width = target.width(), .height = target.height(),
        .color_attachments = tgt_color, .auto_layout_transitions = false,
    });
    cmd.bind_pipeline(im.composite_pipeline);
    cmd.set_viewport(0.f, 0.f, static_cast<float>(target.width()),
                     static_cast<float>(target.height()));
    cmd.set_scissor(0, 0, target.width(), target.height());
    vke::DescriptorSet comp_sets[] = { im.composite_set };
    cmd.bind_descriptor_sets(im.composite_pipeline, comp_sets);
    CompositePC comp_pc{ .max_intensity = params.max_intensity,
                         .black_level   = params.black_level };
    cmd.push_constants(im.composite_pipeline, VK_SHADER_STAGE_FRAGMENT_BIT, comp_pc);
    cmd.draw(3);
    cmd.end_rendering();
}

} // namespace plot
