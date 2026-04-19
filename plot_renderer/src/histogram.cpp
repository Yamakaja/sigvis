#include <plot/histogram.hpp>
#include <vke/vke.hpp>
#include <algorithm>
#include <cstring>

#include "shaders/lines_renderer_mesh_spv.hpp"
#include "shaders/lines_renderer_frag_spv.hpp"

namespace plot {

struct HistogramPC {
    glm::vec2 center;
    glm::vec2 zoom;
    glm::vec2 viewport_ratio;
    float     line_width;
    uint32_t  n_samples;
    uint32_t  n_samples_per_trace;
    uint32_t  trace_offset;
};
static_assert(sizeof(HistogramPC) == 40);

static VkPipelineColorBlendAttachmentState additive_blend() {
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

struct Histogram::Impl {
    vke::Context& ctx;
    uint32_t      width;
    uint32_t      height;

    vke::Image            image;
    vke::Buffer           staging_buffer;   // host-visible, persistently mapped
    vke::Buffer           sample_buffer;    // device-local
    vke::SubmitHandle     pending;          // in-flight fence + owns the command buffer
    vke::DescriptorLayout desc_layout;
    vke::DescriptorSet    desc_set;
    vke::Pipeline         pipeline;

    explicit Impl(vke::Context& ctx, uint32_t w, uint32_t h)
        : ctx(ctx), width(w), height(h)
    {
        image = ctx.create_image({
            .width      = w,
            .height     = h,
            .format     = VK_FORMAT_R32_SFLOAT,
            .usage      = vke::ImageUsage::ColorAttachment |
                          vke::ImageUsage::Sampled         |
                          vke::ImageUsage::TransferSrc,
            .debug_name = "histogram",
        });

        vke::DescriptorBinding binding{
            .binding = 0,
            .type    = vke::DescriptorType::StorageBuffer,
            .stages  = VK_SHADER_STAGE_MESH_BIT_EXT,
        };
        desc_layout = ctx.create_descriptor_layout({ .bindings = { &binding, 1 } });
        desc_set    = desc_layout.allocate_set("histogram_set");

        auto mesh_shader = ctx.create_shader({
            .stage      = vke::ShaderStage::Mesh,
            .spirv_code = spv::lines_renderer_mesh,
        });
        auto frag_shader = ctx.create_shader({
            .stage      = vke::ShaderStage::Fragment,
            .spirv_code = spv::lines_renderer_frag,
        });

        vke::PushConstantRange pc_range{
            .stages = VK_SHADER_STAGE_MESH_BIT_EXT,
            .size   = sizeof(HistogramPC),
        };
        auto blend    = additive_blend();
        VkFormat fmt  = VK_FORMAT_R32_SFLOAT;
        const vke::DescriptorLayout* layouts[] = { &desc_layout };

        pipeline = ctx.create_pipeline(vke::MeshPipelineCreateInfo{
            .mesh_shader              = &mesh_shader,
            .fragment_shader          = &frag_shader,
            .descriptor_layouts       = layouts,
            .push_constant_ranges     = { &pc_range, 1 },
            .color_attachment_formats = { &fmt, 1 },
            .cull_mode                = VK_CULL_MODE_NONE,
            .blend_attachments        = { &blend, 1 },
            .debug_name               = "histogram_pipeline",
        });
    }

    // Block until any in-flight command buffer has completed.
    void wait_pending() {
        if (pending.is_valid())
            ctx.wait(pending);
    }

    void update_desc_set() {
        desc_set.write().bind_storage_buffer(0, sample_buffer).commit();
    }

    void record_batch(vke::CommandBuffer& cmd, const HistogramPC& pc,
                      uint32_t batch_traces, uint32_t groups_y) {
        cmd.image_barrier({
            .image           = image,
            .new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .src_stage_mask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .src_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_2_SHADER_READ_BIT,
            .dst_stage_mask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dst_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        });

        vke::CommandBuffer::ColorAttachmentInfo color[] = {{
            .image       = &image,
            .load_op     = VK_ATTACHMENT_LOAD_OP_LOAD,
            .store_op    = VK_ATTACHMENT_STORE_OP_STORE,
            .clear_value = {.float32 = {0.f, 0.f, 0.f, 0.f}},
        }};
        cmd.begin_rendering({
            .width                   = width,
            .height                  = height,
            .color_attachments       = color,
            .auto_layout_transitions = false,
        });
        cmd.bind_pipeline(pipeline);
        cmd.set_viewport(0.f, 0.f, static_cast<float>(width), static_cast<float>(height));
        cmd.set_scissor(0, 0, width, height);
        vke::DescriptorSet sets[] = { desc_set };
        cmd.bind_descriptor_sets(pipeline, sets);
        cmd.push_constants(pipeline, VK_SHADER_STAGE_MESH_BIT_EXT, pc);
        cmd.draw_mesh_tasks(batch_traces, groups_y, 1);
        cmd.end_rendering();
    }
};

Histogram::Histogram() noexcept = default;

Histogram::Histogram(vke::Context& ctx, uint32_t width, uint32_t height)
    : impl_(std::make_unique<Impl>(ctx, width, height))
{}

Histogram::~Histogram()                               = default;
Histogram::Histogram(Histogram&&) noexcept            = default;
Histogram& Histogram::operator=(Histogram&&) noexcept = default;

uint32_t Histogram::width()  const noexcept { return impl_->width; }
uint32_t Histogram::height() const noexcept { return impl_->height; }

const vke::Image& Histogram::image() const noexcept { return impl_->image; }

void Histogram::flush() {
    impl_->wait_pending();
}

void Histogram::clear() {
    auto& im = *impl_;
    im.wait_pending();

    auto cmd = im.ctx.create_command_buffer();
    cmd.begin();
    cmd.image_barrier({
        .image           = im.image,
        .new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .src_stage_mask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .src_access_mask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
        .dst_stage_mask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dst_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
    });
    vke::CommandBuffer::ColorAttachmentInfo color[] = {{
        .image       = &im.image,
        .load_op     = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .store_op    = VK_ATTACHMENT_STORE_OP_STORE,
        .clear_value = {.float32 = {0.f, 0.f, 0.f, 0.f}},
    }};
    cmd.begin_rendering({
        .width                   = im.width,
        .height                  = im.height,
        .color_attachments       = color,
        .auto_layout_transitions = false,
    });
    cmd.end_rendering();
    cmd.end();
    im.ctx.submit_and_wait(std::move(cmd));
}

void Histogram::draw(std::span<const Sample> samples, uint32_t trace_length,
                     const RenderParams& params) {
    auto& im = *impl_;

    VkDeviceSize needed = sizeof(Sample) * samples.size();

    // Block only if the staging buffer is still in use by an in-flight transfer.
    // This is the only point where we stall; after this the caller's data is safe
    // to modify as soon as we return.
    im.wait_pending();

    // Grow buffers if needed (safe now that pending work is done).
    if (!im.staging_buffer || im.staging_buffer.size() < needed) {
        im.staging_buffer = im.ctx.create_buffer({
            .size       = needed,
            .usage      = vke::BufferUsage::TransferSrc,
            .domain     = vke::MemoryDomain::Host,
            .debug_name = "histogram_staging",
        });
    }
    if (!im.sample_buffer || im.sample_buffer.size() < needed) {
        im.sample_buffer = im.ctx.create_buffer({
            .size       = needed,
            .usage      = vke::BufferUsage::Storage | vke::BufferUsage::TransferDst,
            .domain     = vke::MemoryDomain::Device,
            .debug_name = "histogram_samples",
        });
        im.update_desc_set();
    }

    // Copy into staging — after this line the caller's buffer is safe to reuse.
    auto dst = im.staging_buffer.mapped_as<Sample>();
    std::memcpy(dst.data(), samples.data(), needed);

    uint32_t n_samples = static_cast<uint32_t>(samples.size());
    uint32_t spt       = trace_length > 0 ? trace_length : n_samples;
    uint32_t n_traces  = trace_length > 0 ? n_samples / trace_length : 1;
    uint32_t groups_y  = (spt > 2) ? ((spt - 2 + 31) / 32) : 1;
    float    aspect    = static_cast<float>(im.height) / static_cast<float>(im.width);

    HistogramPC base_pc{
        .center              = params.center,
        .zoom                = params.zoom,
        .viewport_ratio      = { aspect, 1.0f / aspect },
        .line_width          = params.line_width,
        .n_samples           = n_samples,
        .n_samples_per_trace = spt,
        .trace_offset        = 0,
    };

    // Build one command buffer: staging→device transfer, then all render batches.
    auto cmd = im.ctx.create_command_buffer();
    cmd.begin();

    cmd.copy_buffer(im.staging_buffer, im.sample_buffer, 0, 0, needed);
    cmd.buffer_barrier({
        .buffer          = im.sample_buffer,
        .src_stage_mask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .src_access_mask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dst_stage_mask  = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
        .dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT,
    });

    constexpr uint32_t MAX_PER_BATCH = 65535;
    for (uint32_t offset = 0; offset < n_traces; offset += MAX_PER_BATCH) {
        HistogramPC pc  = base_pc;
        pc.trace_offset = offset;
        im.record_batch(cmd, pc, std::min(n_traces - offset, MAX_PER_BATCH), groups_y);
    }

    cmd.end();

    // Submit async — SubmitHandle owns the command buffer until wait() is called.
    im.pending = im.ctx.submit(std::move(cmd));
}

} // namespace plot
