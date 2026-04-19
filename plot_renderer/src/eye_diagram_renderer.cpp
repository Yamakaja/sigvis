#include <plot/eye_diagram_renderer.hpp>
#include <vke/vke.hpp>
#include <cmath>
#include <stdexcept>

// Embedded SPIR-V headers (generated at build time)
#include "shaders/lines_renderer_mesh_spv.hpp"
#include "shaders/lines_renderer_frag_spv.hpp"
#include "shaders/histogram_sample_vert_spv.hpp"
#include "shaders/histogram_sample_frag_spv.hpp"

namespace plot {

// Push constant structs must match shader layout exactly
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

struct CompositePC {
    float max_intensity;
};

// Additive blend state for histogram accumulation
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

static VkPipelineColorBlendAttachmentState alpha_blend() {
    return VkPipelineColorBlendAttachmentState{
        .blendEnable         = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp        = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp        = VK_BLEND_OP_ADD,
        .colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
}

struct EyeDiagramRenderer::Impl {
    vke::Context& ctx;

    uint32_t width;
    uint32_t height;

    // GPU resources
    vke::Buffer          sample_buffer;
    uint32_t             n_samples            = 0;
    uint32_t             n_samples_per_trace  = 0;

    // Histogram pass (mesh shader → R32_SFLOAT)
    vke::Image           histogram_image;
    vke::DescriptorLayout histogram_desc_layout;
    vke::DescriptorSet   histogram_desc_set;
    vke::Pipeline        histogram_pipeline;

    // Composite pass (fullscreen triangle → RGBA8)
    vke::Image           result_image;
    vke::Sampler         histogram_sampler;
    vke::DescriptorLayout composite_desc_layout;
    vke::DescriptorSet   composite_desc_set;
    vke::Pipeline        composite_pipeline;

    explicit Impl(vke::Context& ctx, uint32_t w, uint32_t h)
        : ctx(ctx), width(w), height(h)
    {
        create_images();
        create_sampler();
        create_descriptor_layouts();
        create_pipelines();
        write_composite_descriptor();
    }

    void create_images() {
        histogram_image = ctx.create_image({
            .width   = width,
            .height  = height,
            .format  = VK_FORMAT_R32_SFLOAT,
            .usage   = vke::ImageUsage::ColorAttachment | vke::ImageUsage::Sampled | vke::ImageUsage::TransferSrc,
            .debug_name = "eye_histogram",
        });
        result_image = ctx.create_image({
            .width   = width,
            .height  = height,
            .format  = VK_FORMAT_R8G8B8A8_UNORM,
            .usage   = vke::ImageUsage::ColorAttachment | vke::ImageUsage::TransferSrc,
            .debug_name = "eye_result",
        });
    }

    void create_sampler() {
        histogram_sampler = ctx.create_sampler({
            .mag_filter = VK_FILTER_NEAREST,
            .min_filter = VK_FILTER_NEAREST,
            .address_u  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .address_v  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        });
    }

    void create_descriptor_layouts() {
        // Histogram pass: binding 0 = storage buffer (samples)
        vke::DescriptorBinding hist_bindings[] = {{
            .binding = 0,
            .type    = vke::DescriptorType::StorageBuffer,
            .stages  = VK_SHADER_STAGE_MESH_BIT_EXT,
        }};
        histogram_desc_layout = ctx.create_descriptor_layout({
            .bindings   = hist_bindings,
            .debug_name = "eye_histogram_layout",
        });
        histogram_desc_set = histogram_desc_layout.allocate_set("eye_histogram_set");

        // Composite pass: binding 0 = combined image sampler (histogram)
        vke::DescriptorBinding comp_bindings[] = {{
            .binding = 0,
            .type    = vke::DescriptorType::CombinedImageSampler,
            .stages  = VK_SHADER_STAGE_FRAGMENT_BIT,
        }};
        composite_desc_layout = ctx.create_descriptor_layout({
            .bindings   = comp_bindings,
            .debug_name = "eye_composite_layout",
        });
        composite_desc_set = composite_desc_layout.allocate_set("eye_composite_set");
    }

    void create_pipelines() {
        // Histogram mesh pipeline
        auto mesh_shader = ctx.create_shader({
            .stage      = vke::ShaderStage::Mesh,
            .spirv_code = spv::lines_renderer_mesh,
        });
        auto hist_frag = ctx.create_shader({
            .stage      = vke::ShaderStage::Fragment,
            .spirv_code = spv::lines_renderer_frag,
        });

        vke::PushConstantRange hist_pc{
            .stages = VK_SHADER_STAGE_MESH_BIT_EXT,
            .size   = sizeof(HistogramPC),
        };
        auto hist_blend = additive_blend();
        VkFormat hist_fmt = VK_FORMAT_R32_SFLOAT;

        const vke::DescriptorLayout* hist_layouts[] = { &histogram_desc_layout };
        histogram_pipeline = ctx.create_pipeline(vke::MeshPipelineCreateInfo{
            .mesh_shader               = &mesh_shader,
            .fragment_shader           = &hist_frag,
            .descriptor_layouts        = hist_layouts,
            .push_constant_ranges      = { &hist_pc, 1 },
            .color_attachment_formats  = { &hist_fmt, 1 },
            .cull_mode                 = VK_CULL_MODE_NONE,
            .blend_attachments         = { &hist_blend, 1 },
            .debug_name                = "eye_histogram_pipeline",
        });

        // Composite graphics pipeline
        auto comp_vert = ctx.create_shader({
            .stage      = vke::ShaderStage::Vertex,
            .spirv_code = spv::histogram_sample_vert,
        });
        auto comp_frag = ctx.create_shader({
            .stage      = vke::ShaderStage::Fragment,
            .spirv_code = spv::histogram_sample_frag,
        });

        vke::PushConstantRange comp_pc{
            .stages = VK_SHADER_STAGE_FRAGMENT_BIT,
            .size   = sizeof(CompositePC),
        };
        auto comp_blend = alpha_blend();
        VkFormat comp_fmt = VK_FORMAT_R8G8B8A8_UNORM;

        const vke::DescriptorLayout* comp_layouts[] = { &composite_desc_layout };
        composite_pipeline = ctx.create_pipeline(vke::GraphicsPipelineCreateInfo{
            .vertex_shader            = &comp_vert,
            .fragment_shader          = &comp_frag,
            .descriptor_layouts       = comp_layouts,
            .push_constant_ranges     = { &comp_pc, 1 },
            .color_attachment_formats = { &comp_fmt, 1 },
            .cull_mode                = VK_CULL_MODE_NONE,
            .blend_attachments        = { &comp_blend, 1 },
            .debug_name               = "eye_composite_pipeline",
        });
    }

    void write_composite_descriptor() {
        composite_desc_set.write()
            .bind_combined_image_sampler(0, histogram_image, histogram_sampler,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .commit();
    }

    void update_sample_descriptor() {
        histogram_desc_set.write()
            .bind_storage_buffer(0, sample_buffer)
            .commit();
    }

    // Records one histogram batch into cmd. clear=true on first batch, false for accumulation.
    // Leaves histogram_image in COLOR_ATTACHMENT_OPTIMAL.
    void record_histogram_batch(vke::CommandBuffer& cmd, const HistogramPC& pc,
                                uint32_t batch_traces, uint32_t groups_y, bool clear) {
        cmd.image_barrier({
            .image           = histogram_image,
            .new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .src_stage_mask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .src_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_2_SHADER_READ_BIT,
            .dst_stage_mask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dst_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        });

        vke::CommandBuffer::ColorAttachmentInfo hist_color[] = {{
            .image       = &histogram_image,
            .load_op     = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
            .store_op    = VK_ATTACHMENT_STORE_OP_STORE,
            .clear_value = {.float32 = {0.f, 0.f, 0.f, 0.f}},
        }};
        cmd.begin_rendering({
            .width                   = width,
            .height                  = height,
            .color_attachments       = hist_color,
            .depth_attachment        = {},
            .auto_layout_transitions = false,
        });

        cmd.bind_pipeline(histogram_pipeline);
        cmd.set_viewport(0.f, 0.f, static_cast<float>(width), static_cast<float>(height));
        cmd.set_scissor(0, 0, width, height);

        vke::DescriptorSet sets[] = { histogram_desc_set };
        cmd.bind_descriptor_sets(histogram_pipeline, sets);
        cmd.push_constants(histogram_pipeline, VK_SHADER_STAGE_MESH_BIT_EXT, pc);
        cmd.draw_mesh_tasks(batch_traces, groups_y, 1);
        cmd.end_rendering();
    }

    // Submits histogram batches one submission at a time to avoid GPU watchdog timeouts.
    static constexpr uint32_t MAX_TRACES_PER_SUBMIT = 65535;

    void submit_histogram_batches(const HistogramPC& base_pc,
                                  uint32_t n_traces, uint32_t groups_y) {
        for (uint32_t offset = 0; offset < n_traces; offset += MAX_TRACES_PER_SUBMIT) {
            uint32_t batch    = std::min(n_traces - offset, MAX_TRACES_PER_SUBMIT);
            HistogramPC pc    = base_pc;
            pc.trace_offset   = offset;
            auto cmd = ctx.create_command_buffer();
            cmd.begin();
            record_histogram_batch(cmd, pc, batch, groups_y, /*clear=*/offset == 0);
            cmd.end();
            ctx.submit_and_wait(std::move(cmd));
        }
    }

    // Prepares the push constants and dispatch dimensions from current state.
    HistogramPC make_histogram_pc(const RenderParams& params,
                                  uint32_t& out_n_traces, uint32_t& out_groups_y) const {
        uint32_t spt      = n_samples_per_trace > 0 ? n_samples_per_trace : n_samples;
        out_n_traces      = n_samples_per_trace > 0 ? n_samples / n_samples_per_trace : 1;
        out_groups_y      = (spt > 2) ? ((spt - 2 + 31) / 32) : 1;
        float aspect      = static_cast<float>(height) / static_cast<float>(width);
        return HistogramPC{
            .center              = params.center,
            .zoom                = params.zoom,
            .viewport_ratio      = { aspect, 1.0f / aspect },
            .line_width          = params.line_width,
            .n_samples           = n_samples,
            .n_samples_per_trace = spt,
        };
    }
};

EyeDiagramRenderer::EyeDiagramRenderer() noexcept = default;

EyeDiagramRenderer::EyeDiagramRenderer(vke::Context& ctx, uint32_t width, uint32_t height)
    : impl_(std::make_unique<Impl>(ctx, width, height))
{}

EyeDiagramRenderer::~EyeDiagramRenderer() = default;
EyeDiagramRenderer::EyeDiagramRenderer(EyeDiagramRenderer&&) noexcept = default;
EyeDiagramRenderer& EyeDiagramRenderer::operator=(EyeDiagramRenderer&&) noexcept = default;

uint32_t EyeDiagramRenderer::width()  const noexcept { return impl_->width; }
uint32_t EyeDiagramRenderer::height() const noexcept { return impl_->height; }

void EyeDiagramRenderer::set_samples(std::span<const Sample> samples, uint32_t trace_length) {
    auto& im = *impl_;
    VkDeviceSize needed = sizeof(Sample) * samples.size();

    if (!im.sample_buffer || im.sample_buffer.size() < needed) {
        im.sample_buffer = im.ctx.create_buffer({
            .size       = needed,
            .usage      = vke::BufferUsage::Storage,
            .domain     = vke::MemoryDomain::Device,
            .debug_name = "eye_samples",
        });
        im.update_sample_descriptor();
    }

    im.sample_buffer.upload<Sample>(samples);
    im.n_samples           = static_cast<uint32_t>(samples.size());
    im.n_samples_per_trace = trace_length;
}

const vke::Image& EyeDiagramRenderer::render_histogram(const RenderParams& params) {
    auto& im = *impl_;
    if (im.n_samples == 0) return im.histogram_image;

    uint32_t n_traces, groups_y;
    HistogramPC pc = im.make_histogram_pc(params, n_traces, groups_y);
    im.submit_histogram_batches(pc, n_traces, groups_y);
    return im.histogram_image;
}

const vke::Image& EyeDiagramRenderer::render(const RenderParams& params) {
    auto& im = *impl_;
    if (im.n_samples == 0) return im.result_image;

    uint32_t n_traces, groups_y;
    HistogramPC hist_pc = im.make_histogram_pc(params, n_traces, groups_y);
    CompositePC comp_pc{ .max_intensity = params.max_intensity };

    im.submit_histogram_batches(hist_pc, n_traces, groups_y);

    // histogram_image is now COLOR_ATTACHMENT_OPTIMAL; composite it into result_image
    auto cmd = im.ctx.create_command_buffer();
    cmd.begin();

    cmd.image_barrier({
        .image           = im.histogram_image,
        .new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .src_stage_mask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .src_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .dst_stage_mask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT,
    });
    cmd.image_barrier({
        .image           = im.result_image,
        .new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .src_stage_mask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
        .src_access_mask = VK_ACCESS_2_NONE,
        .dst_stage_mask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dst_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
    });

    vke::CommandBuffer::ColorAttachmentInfo comp_color[] = {{
        .image       = &im.result_image,
        .load_op     = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .store_op    = VK_ATTACHMENT_STORE_OP_STORE,
        .clear_value = {.float32 = {0.f, 0.f, 0.f, 1.f}},
    }};
    cmd.begin_rendering({
        .width                   = im.width,
        .height                  = im.height,
        .color_attachments       = comp_color,
        .depth_attachment        = {},
        .auto_layout_transitions = false,
    });
    cmd.bind_pipeline(im.composite_pipeline);
    cmd.set_viewport(0.f, 0.f, static_cast<float>(im.width), static_cast<float>(im.height));
    cmd.set_scissor(0, 0, im.width, im.height);
    vke::DescriptorSet sets[] = { im.composite_desc_set };
    cmd.bind_descriptor_sets(im.composite_pipeline, sets);
    cmd.push_constants(im.composite_pipeline, VK_SHADER_STAGE_FRAGMENT_BIT, comp_pc);
    cmd.draw(3);
    cmd.end_rendering();

    cmd.end();
    im.ctx.submit_and_wait(std::move(cmd));
    return im.result_image;
}

} // namespace plot
