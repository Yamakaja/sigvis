#include <plot/roll_scope.hpp>
#include <vke/vke.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "shaders/roll_reduce_comp_spv.hpp"
#include "shaders/histogram_sample_vert_spv.hpp"
#include "shaders/roll_composite_frag_spv.hpp"

namespace plot {

// Matches roll_reduce.comp (6 * u32 + 4 * f32 == 40 bytes).
struct RollReducePC {
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
};
static_assert(sizeof(RollReducePC) == 40);

// Matches roll_composite.frag.
struct RollCompositePC {
    float    max_intensity;
    float    black_level;
    uint32_t head_col;
    uint32_t strip_width;
};
static_assert(sizeof(RollCompositePC) == 16);

namespace {
constexpr float    WEIGHT_SCALE     = 1024.0f;
constexpr uint32_t MAX_PENDING      = 1u << 20; // host upload cap per frame (samples)
} // namespace

struct RollScope::Impl {
    vke::Context& ctx;
    uint32_t      width;
    uint32_t      height;
    VkFormat      target_format;
    float         sample_rate;
    uint32_t      frames_in_flight;
    uint32_t      ring_capacity;   // C: max-window samples

    // Sample ring (device-local, circular) + per-slot host staging.
    vke::Buffer              ring;
    std::vector<vke::Buffer> host_slots;
    uint32_t                 slot_index = 0;

    // Strip image (R32_SFLOAT), horizontally circular.
    vke::Image strip;

    // Compute reduction.
    vke::DescriptorLayout reduce_layout;
    vke::DescriptorSet    reduce_set;
    vke::Pipeline         reduce_pipeline;

    // Composite.
    vke::Sampler          sampler;
    vke::DescriptorLayout composite_layout;
    vke::DescriptorSet    composite_set;
    vke::Pipeline         composite_pipeline;

    // Stripe bookkeeping (monotonic sample counters).
    uint64_t total_samples_   = 0; // ever pushed to the ring
    uint64_t samples_consumed_ = 0; // turned into stripes (multiple of K)
    uint64_t stripes_emitted_ = 0; // == samples_consumed_ / K
    uint32_t cur_K            = 0; // current samples-per-stripe (0 = unset)
    bool     needs_replay     = true;

    std::vector<float> pending_; // CPU-side samples awaiting upload

    Impl(vke::Context& c, uint32_t w, uint32_t h, VkFormat fmt, float fs,
         float max_window_s, uint32_t fif)
        : ctx(c), width(w), height(h), target_format(fmt),
          sample_rate(fs), frames_in_flight(std::max(1u, fif))
    {
        ring_capacity = std::max<uint32_t>(
            1, static_cast<uint32_t>(std::ceil(max_window_s * sample_rate)));

        ring = ctx.create_buffer({
            .size = static_cast<VkDeviceSize>(ring_capacity) * sizeof(float),
            .usage = vke::BufferUsage::Storage | vke::BufferUsage::TransferDst,
            .domain = vke::MemoryDomain::Device, .debug_name = "roll_sample_ring" });

        uint32_t host_cap = std::min(ring_capacity, MAX_PENDING);
        host_slots.resize(frames_in_flight);
        for (auto& b : host_slots)
            b = ctx.create_buffer({
                .size = static_cast<VkDeviceSize>(host_cap) * sizeof(float),
                .usage = vke::BufferUsage::TransferSrc,
                .domain = vke::MemoryDomain::Host, .debug_name = "roll_upload_host" });

        create_strip();
        create_pipelines();
        clear_strip_immediate();
    }

    void create_strip() {
        strip = ctx.create_image({
            .width = width, .height = height, .format = VK_FORMAT_R32_SFLOAT,
            .usage = vke::ImageUsage::ColorAttachment | vke::ImageUsage::Storage |
                     vke::ImageUsage::Sampled | vke::ImageUsage::TransferSrc,
            .debug_name = "roll_strip" });
        needs_replay = true;
    }

    void create_pipelines() {
        // ---- Compute reduction ----
        vke::DescriptorBinding rb[] = {
            { .binding = 0, .type = vke::DescriptorType::StorageBuffer,
              .stages = VK_SHADER_STAGE_COMPUTE_BIT },
            { .binding = 1, .type = vke::DescriptorType::StorageImage,
              .stages = VK_SHADER_STAGE_COMPUTE_BIT },
        };
        reduce_layout = ctx.create_descriptor_layout({ .bindings = rb,
            .debug_name = "roll_reduce_layout" });
        reduce_set = reduce_layout.allocate_set("roll_reduce_set");

        auto comp = ctx.create_shader({
            .stage = vke::ShaderStage::Compute, .spirv_code = spv::roll_reduce_comp });
        vke::PushConstantRange rpc{ .stages = VK_SHADER_STAGE_COMPUTE_BIT,
                                    .size = sizeof(RollReducePC) };
        const vke::DescriptorLayout* rl[] = { &reduce_layout };
        reduce_pipeline = ctx.create_pipeline(vke::ComputePipelineCreateInfo{
            .compute_shader = &comp, .descriptor_layouts = rl,
            .push_constant_ranges = { &rpc, 1 }, .debug_name = "roll_reduce_pipeline" });

        // ---- Composite ----
        sampler = ctx.create_sampler({
            .mag_filter = VK_FILTER_NEAREST, .min_filter = VK_FILTER_NEAREST,
            .address_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .address_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE });

        vke::DescriptorBinding cb{ .binding = 0,
            .type = vke::DescriptorType::CombinedImageSampler,
            .stages = VK_SHADER_STAGE_FRAGMENT_BIT };
        composite_layout = ctx.create_descriptor_layout({ .bindings = { &cb, 1 },
            .debug_name = "roll_composite_layout" });
        composite_set = composite_layout.allocate_set("roll_composite_set");

        auto fs_vert = ctx.create_shader({
            .stage = vke::ShaderStage::Vertex, .spirv_code = spv::histogram_sample_vert });
        auto comp_frag = ctx.create_shader({
            .stage = vke::ShaderStage::Fragment, .spirv_code = spv::roll_composite_frag });
        vke::PushConstantRange cpc{ .stages = VK_SHADER_STAGE_FRAGMENT_BIT,
                                    .size = sizeof(RollCompositePC) };
        const vke::DescriptorLayout* cl[] = { &composite_layout };
        composite_pipeline = ctx.create_pipeline(vke::GraphicsPipelineCreateInfo{
            .vertex_shader = &fs_vert, .fragment_shader = &comp_frag,
            .descriptor_layouts = cl, .push_constant_ranges = { &cpc, 1 },
            .color_attachment_formats = { &target_format, 1 },
            .cull_mode = VK_CULL_MODE_NONE, .debug_name = "roll_composite_pipeline" });

        write_descriptors();
    }

    void write_descriptors() {
        reduce_set.write()
            .bind_storage_buffer(0, ring)
            .bind_storage_image(1, strip, VK_IMAGE_LAYOUT_GENERAL)
            .commit();
        composite_set.write()
            .bind_combined_image_sampler(0, strip, sampler,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            .commit();
    }

    // One-shot clear so the strip starts black (leaves COLOR_ATTACHMENT_OPTIMAL).
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
        cmd.end();
        ctx.submit_and_wait(std::move(cmd));
    }

    // Record a clear of the strip into `cmd` (for replay). Leaves it in
    // COLOR_ATTACHMENT_OPTIMAL.
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
    }
};

RollScope::RollScope(vke::Context& ctx, uint32_t width, uint32_t height,
                     VkFormat target_format, float sample_rate_hz,
                     float max_window_seconds, uint32_t frames_in_flight)
    : impl_(std::make_unique<Impl>(ctx, width, height, target_format, sample_rate_hz,
                                   max_window_seconds, frames_in_flight)) {}

RollScope::~RollScope()                              = default;
RollScope::RollScope(RollScope&&) noexcept           = default;
RollScope& RollScope::operator=(RollScope&&) noexcept = default;

uint32_t RollScope::width()  const noexcept { return impl_->width; }
uint32_t RollScope::height() const noexcept { return impl_->height; }
uint32_t RollScope::samples_per_stripe() const noexcept { return impl_->cur_K; }
size_t   RollScope::pending() const { return impl_->pending_.size(); }

void RollScope::resize(uint32_t width, uint32_t height) {
    auto& im = *impl_;
    if (width == im.width && height == im.height) return;
    im.width = width;
    im.height = height;
    im.create_strip();
    im.write_descriptors();
    im.clear_strip_immediate();
}

void RollScope::push_chunk(std::span<const float> samples) {
    auto& im = *impl_;
    if (samples.empty()) return;
    im.pending_.insert(im.pending_.end(), samples.begin(), samples.end());
    // Bound CPU backlog: if rendering can't keep up, keep only the most recent.
    if (im.pending_.size() > MAX_PENDING)
        im.pending_.erase(im.pending_.begin(),
                          im.pending_.end() - MAX_PENDING);
}

void RollScope::render(vke::CommandBuffer& cmd, vke::Image& target,
                       const RollParams& params) {
    auto& im = *impl_;
    const uint32_t W = im.width;
    const uint32_t H = im.height;

    // ---- Determine samples-per-stripe (K) from the window; replay if it changed ----
    uint32_t K = static_cast<uint32_t>(std::lround(
        static_cast<double>(params.window_seconds) * im.sample_rate / std::max(1u, W)));
    K = std::max(1u, K);
    if (K != im.cur_K) { im.cur_K = K; im.needs_replay = true; }

    // ---- Upload pending samples into the ring (one or two wrapped copies) ----
    if (!im.pending_.empty()) {
        auto& host = im.host_slots[im.slot_index];
        im.slot_index = (im.slot_index + 1) % im.frames_in_flight;

        uint32_t host_cap = static_cast<uint32_t>(host.size() / sizeof(float));
        uint32_t n = static_cast<uint32_t>(std::min<size_t>(im.pending_.size(), host_cap));
        const float* src = im.pending_.data() + (im.pending_.size() - n);

        std::memcpy(host.mapped_as<float>().data(), src, n * sizeof(float));

        uint32_t start = static_cast<uint32_t>(im.total_samples_ % im.ring_capacity);
        uint32_t first = std::min(n, im.ring_capacity - start);
        cmd.copy_buffer(host, im.ring, 0, static_cast<VkDeviceSize>(start) * sizeof(float),
                        static_cast<VkDeviceSize>(first) * sizeof(float));
        if (n > first)
            cmd.copy_buffer(host, im.ring,
                            static_cast<VkDeviceSize>(first) * sizeof(float), 0,
                            static_cast<VkDeviceSize>(n - first) * sizeof(float));
        im.total_samples_ += n;
        im.pending_.clear();

        cmd.buffer_barrier({
            .buffer = im.ring,
            .src_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .src_access_mask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dst_stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT });
    }

    // ---- Replay: clear the strip and rewind counters to re-render recent history ----
    if (im.needs_replay) {
        im.clear_strip(cmd);
        uint64_t total_stripes = im.total_samples_ / K;
        uint64_t want = std::min<uint64_t>({ static_cast<uint64_t>(W),
                                             im.ring_capacity / K, total_stripes });
        im.stripes_emitted_  = total_stripes - want;
        im.samples_consumed_ = im.stripes_emitted_ * static_cast<uint64_t>(K);
        im.needs_replay = false;
    }

    // ---- How many whole new stripes are available, bounded by ring + strip width ----
    uint64_t avail   = im.total_samples_ - im.samples_consumed_;
    uint64_t n_new64 = avail / K;
    uint64_t max_safe = im.ring_capacity / K;          // stripes that fit in the ring
    if (n_new64 > max_safe) {                            // fell behind: drop oldest unread
        uint64_t skip = n_new64 - max_safe;
        im.samples_consumed_ += skip * K; im.stripes_emitted_ += skip; n_new64 = max_safe;
    }
    if (n_new64 > W) {                                   // more than a full screen: cap
        uint64_t skip = n_new64 - W;
        im.samples_consumed_ += skip * K; im.stripes_emitted_ += skip; n_new64 = W;
    }
    uint32_t n_new = static_cast<uint32_t>(n_new64);

    // ---- Compute reduction: write new columns into the circular strip ----
    if (n_new > 0) {
        cmd.image_barrier({
            .image = im.strip, .new_layout = VK_IMAGE_LAYOUT_GENERAL,
            .src_stage_mask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .src_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_2_SHADER_READ_BIT,
            .dst_stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dst_access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT });

        RollReducePC pc{
            .ring_capacity = im.ring_capacity, .strip_width = W, .height = H,
            .samples_per_stripe = K,
            .ring_start_phys = static_cast<uint32_t>(im.samples_consumed_ % im.ring_capacity),
            .col_start_phys  = static_cast<uint32_t>(im.stripes_emitted_ % W),
            .y_min = params.y_min, .y_max = params.y_max,
            .line_width_px = params.line_width_px, .weight_scale = WEIGHT_SCALE };

        cmd.bind_pipeline(im.reduce_pipeline);
        vke::DescriptorSet sets[] = { im.reduce_set };
        cmd.bind_descriptor_sets(im.reduce_pipeline, sets);
        cmd.push_constants(im.reduce_pipeline, VK_SHADER_STAGE_COMPUTE_BIT, pc);
        cmd.dispatch(n_new, 1, 1);

        im.stripes_emitted_  += n_new;
        im.samples_consumed_ += static_cast<uint64_t>(n_new) * K;
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
    RollCompositePC cpc{ .max_intensity = params.max_intensity,
                         .black_level = params.black_level,
                         .head_col = static_cast<uint32_t>(im.stripes_emitted_ % W),
                         .strip_width = W };
    cmd.push_constants(im.composite_pipeline, VK_SHADER_STAGE_FRAGMENT_BIT, cpc);
    cmd.draw(3);
    cmd.end_rendering();
}

} // namespace plot
