#include <vke/command_buffer.hpp>
#include <vke/image.hpp>
#include <vke/error.hpp>
#include <vector>

namespace vke {

template<typename PFN>
static PFN load(VkDevice device, const char* name) {
    return reinterpret_cast<PFN>(vkGetDeviceProcAddr(device, name));
}

CommandBuffer::CommandBuffer(VkDevice device, VkCommandPool pool, VkCommandBuffer cmd)
    : device_(device), pool_(pool), cmd_(cmd),
      pfn_draw_mesh_tasks_(load<PFN_vkCmdDrawMeshTasksEXT>(device, "vkCmdDrawMeshTasksEXT")),
      pfn_draw_mesh_tasks_indirect_(load<PFN_vkCmdDrawMeshTasksIndirectEXT>(device, "vkCmdDrawMeshTasksIndirectEXT")),
      pfn_begin_debug_label_(load<PFN_vkCmdBeginDebugUtilsLabelEXT>(device, "vkCmdBeginDebugUtilsLabelEXT")),
      pfn_end_debug_label_(load<PFN_vkCmdEndDebugUtilsLabelEXT>(device, "vkCmdEndDebugUtilsLabelEXT")),
      pfn_insert_debug_label_(load<PFN_vkCmdInsertDebugUtilsLabelEXT>(device, "vkCmdInsertDebugUtilsLabelEXT"))
{}

CommandBuffer::~CommandBuffer() {
    if (cmd_ != VK_NULL_HANDLE && state_ != State::Invalid)
        vkFreeCommandBuffers(device_, pool_, 1, &cmd_);
}

CommandBuffer::CommandBuffer(CommandBuffer&& o) noexcept
    : device_(o.device_), pool_(o.pool_), cmd_(o.cmd_),
      state_(o.state_), in_rendering_(o.in_rendering_),
      pfn_draw_mesh_tasks_(o.pfn_draw_mesh_tasks_),
      pfn_draw_mesh_tasks_indirect_(o.pfn_draw_mesh_tasks_indirect_),
      pfn_begin_debug_label_(o.pfn_begin_debug_label_),
      pfn_end_debug_label_(o.pfn_end_debug_label_),
      pfn_insert_debug_label_(o.pfn_insert_debug_label_)
{
    o.cmd_   = VK_NULL_HANDLE;
    o.state_ = State::Invalid;
}

CommandBuffer& CommandBuffer::operator=(CommandBuffer&& o) noexcept {
    if (this != &o) {
        if (cmd_ != VK_NULL_HANDLE && state_ != State::Invalid)
            vkFreeCommandBuffers(device_, pool_, 1, &cmd_);
        device_       = o.device_;
        pool_         = o.pool_;
        cmd_          = o.cmd_;
        state_        = o.state_;
        in_rendering_ = o.in_rendering_;
        pfn_draw_mesh_tasks_          = o.pfn_draw_mesh_tasks_;
        pfn_draw_mesh_tasks_indirect_ = o.pfn_draw_mesh_tasks_indirect_;
        pfn_begin_debug_label_        = o.pfn_begin_debug_label_;
        pfn_end_debug_label_          = o.pfn_end_debug_label_;
        pfn_insert_debug_label_       = o.pfn_insert_debug_label_;
        o.cmd_   = VK_NULL_HANDLE;
        o.state_ = State::Invalid;
    }
    return *this;
}

void CommandBuffer::begin() {
    VKE_ASSERT(state_ == State::Initial, "CommandBuffer::begin called in wrong state");
    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VKE_CHECK(vkBeginCommandBuffer(cmd_, &bi));
    state_ = State::Recording;
}

void CommandBuffer::end() {
    VKE_ASSERT(state_ == State::Recording, "CommandBuffer::end called in wrong state");
    VKE_ASSERT(!in_rendering_, "CommandBuffer::end called inside begin_rendering");
    VKE_CHECK(vkEndCommandBuffer(cmd_));
    state_ = State::Executable;
}

// ---- Barriers ----

static VkImageAspectFlags aspect_for_layout(VkImageLayout layout, VkFormat fmt) {
    bool depth = (fmt == VK_FORMAT_D16_UNORM || fmt == VK_FORMAT_D32_SFLOAT ||
                  fmt == VK_FORMAT_X8_D24_UNORM_PACK32 ||
                  fmt == VK_FORMAT_D16_UNORM_S8_UINT || fmt == VK_FORMAT_D24_UNORM_S8_UINT ||
                  fmt == VK_FORMAT_D32_SFLOAT_S8_UINT);
    if (depth) return VK_IMAGE_ASPECT_DEPTH_BIT;
    return VK_IMAGE_ASPECT_COLOR_BIT;
    (void)layout;
}

void CommandBuffer::image_barrier(const ImageBarrierInfo& info) {
    VKE_ASSERT(state_ == State::Recording, "image_barrier outside recording");
    VkImageMemoryBarrier2 barrier{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask     = info.src_stage_mask,
        .srcAccessMask    = info.src_access_mask,
        .dstStageMask     = info.dst_stage_mask,
        .dstAccessMask    = info.dst_access_mask,
        .oldLayout        = info.image.current_layout(),
        .newLayout        = info.new_layout,
        .image            = info.image.native_handle(),
        .subresourceRange = {
            .aspectMask = aspect_for_layout(info.new_layout, info.image.format()),
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };
    VkDependencyInfo di{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &barrier,
    };
    vkCmdPipelineBarrier2(cmd_, &di);
    info.image.set_assumed_layout(info.new_layout);
}

void CommandBuffer::image_barriers(std::span<const ImageBarrierInfo> infos) {
    VKE_ASSERT(state_ == State::Recording, "image_barriers outside recording");
    std::vector<VkImageMemoryBarrier2> barriers;
    barriers.reserve(infos.size());
    for (auto& info : infos) {
        barriers.push_back(VkImageMemoryBarrier2{
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask     = info.src_stage_mask,
            .srcAccessMask    = info.src_access_mask,
            .dstStageMask     = info.dst_stage_mask,
            .dstAccessMask    = info.dst_access_mask,
            .oldLayout        = info.image.current_layout(),
            .newLayout        = info.new_layout,
            .image            = info.image.native_handle(),
            .subresourceRange = {
                .aspectMask = aspect_for_layout(info.new_layout, info.image.format()),
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        });
    }
    VkDependencyInfo di{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
        .pImageMemoryBarriers    = barriers.data(),
    };
    vkCmdPipelineBarrier2(cmd_, &di);
    for (auto& info : infos)
        info.image.set_assumed_layout(info.new_layout);
}

void CommandBuffer::buffer_barrier(const BufferBarrierInfo& info) {
    VKE_ASSERT(state_ == State::Recording, "buffer_barrier outside recording");
    VkBufferMemoryBarrier2 barrier{
        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask  = info.src_stage_mask,
        .srcAccessMask = info.src_access_mask,
        .dstStageMask  = info.dst_stage_mask,
        .dstAccessMask = info.dst_access_mask,
        .buffer        = info.buffer.native_handle(),
        .offset        = info.offset,
        .size          = info.size,
    };
    VkDependencyInfo di{
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers    = &barrier,
    };
    vkCmdPipelineBarrier2(cmd_, &di);
}

void CommandBuffer::full_barrier() {
    VKE_ASSERT(state_ == State::Recording, "full_barrier outside recording");
    VkMemoryBarrier2 barrier{
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
    };
    VkDependencyInfo di{
        .sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers    = &barrier,
    };
    vkCmdPipelineBarrier2(cmd_, &di);
}

// ---- Dynamic Rendering ----

void CommandBuffer::begin_rendering(const RenderingInfo& info) {
    VKE_ASSERT(state_ == State::Recording, "begin_rendering outside recording");
    VKE_ASSERT(!in_rendering_, "nested begin_rendering");

    uint32_t w = info.width;
    uint32_t h = info.height;
    if (w == 0 && !info.color_attachments.empty() && info.color_attachments[0].image)
        w = info.color_attachments[0].image->width();
    if (h == 0 && !info.color_attachments.empty() && info.color_attachments[0].image)
        h = info.color_attachments[0].image->height();

    std::vector<VkRenderingAttachmentInfo> color_attachments;
    color_attachments.reserve(info.color_attachments.size());

    for (auto& ca : info.color_attachments) {
        VKE_ASSERT(ca.image != nullptr, "ColorAttachmentInfo::image is null");

        if (info.auto_layout_transitions &&
            ca.image->current_layout() != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            // Transition in-place
            const_cast<Image*>(ca.image)->set_assumed_layout(VK_IMAGE_LAYOUT_UNDEFINED);
            ImageBarrierInfo bar{
                .image      = *const_cast<Image*>(ca.image),
                .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .src_stage_mask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                .src_access_mask = VK_ACCESS_2_NONE,
                .dst_stage_mask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dst_access_mask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            };
            image_barrier(bar);
        }

        VkImageView view = ca.view ? ca.view : ca.image->native_default_view();
        color_attachments.push_back(VkRenderingAttachmentInfo{
            .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView   = view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp      = ca.load_op,
            .storeOp     = ca.store_op,
            .clearValue  = { .color = ca.clear_value },
        });
    }

    VkRenderingAttachmentInfo depth_attachment_info{};
    VkRenderingAttachmentInfo* p_depth = nullptr;
    if (info.depth_attachment) {
        auto& da = *info.depth_attachment;
        VKE_ASSERT(da.image != nullptr, "DepthAttachmentInfo::image is null");
        depth_attachment_info = VkRenderingAttachmentInfo{
            .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView   = da.image->native_default_view(),
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .loadOp      = da.load_op,
            .storeOp     = da.store_op,
            .clearValue  = { .depthStencil = { da.clear_depth, da.clear_stencil } },
        };
        p_depth = &depth_attachment_info;
    }

    VkRenderingInfo ri{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = { {static_cast<int32_t>(info.x), static_cast<int32_t>(info.y)},
                                  {w, h} },
        .layerCount           = info.layer_count,
        .colorAttachmentCount = static_cast<uint32_t>(color_attachments.size()),
        .pColorAttachments    = color_attachments.empty() ? nullptr : color_attachments.data(),
        .pDepthAttachment     = p_depth,
    };
    vkCmdBeginRendering(cmd_, &ri);
    in_rendering_ = true;
}

void CommandBuffer::end_rendering() {
    VKE_ASSERT(in_rendering_, "end_rendering without begin_rendering");
    vkCmdEndRendering(cmd_);
    in_rendering_ = false;
}

// ---- Pipeline + State ----

void CommandBuffer::bind_pipeline(const Pipeline& pipeline) {
    VKE_ASSERT(state_ == State::Recording, "bind_pipeline outside recording");
    VkPipelineBindPoint point;
    switch (pipeline.type()) {
    case Pipeline::Type::Compute:  point = VK_PIPELINE_BIND_POINT_COMPUTE;  break;
    default:                       point = VK_PIPELINE_BIND_POINT_GRAPHICS; break;
    }
    vkCmdBindPipeline(cmd_, point, pipeline.native_handle());
}

void CommandBuffer::set_viewport(float x, float y, float w, float h,
                                  float min_depth, float max_depth) {
    VkViewport vp{ x, y, w, h, min_depth, max_depth };
    vkCmdSetViewport(cmd_, 0, 1, &vp);
}

void CommandBuffer::set_scissor(int32_t x, int32_t y, uint32_t w, uint32_t h) {
    VkRect2D s{ {x, y}, {w, h} };
    vkCmdSetScissor(cmd_, 0, 1, &s);
}

void CommandBuffer::set_blend_constants(float r, float g, float b, float a) {
    const float c[4] = { r, g, b, a };
    vkCmdSetBlendConstants(cmd_, c);
}

void CommandBuffer::bind_descriptor_sets(const Pipeline& pipeline,
                                          std::span<const DescriptorSet> sets,
                                          uint32_t first_set) {
    VKE_ASSERT(state_ == State::Recording, "bind_descriptor_sets outside recording");
    std::vector<VkDescriptorSet> raw;
    raw.reserve(sets.size());
    for (auto& s : sets) raw.push_back(s.native_handle());

    VkPipelineBindPoint point = pipeline.type() == Pipeline::Type::Compute
        ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS;
    vkCmdBindDescriptorSets(cmd_, point, pipeline.native_layout_handle(),
                             first_set, static_cast<uint32_t>(raw.size()), raw.data(),
                             0, nullptr);
}

void CommandBuffer::bind_descriptor_sets(const Pipeline& pipeline,
                                          std::span<const DescriptorSet> sets,
                                          std::span<const uint32_t> dynamic_offsets,
                                          uint32_t first_set) {
    VKE_ASSERT(state_ == State::Recording, "bind_descriptor_sets outside recording");
    std::vector<VkDescriptorSet> raw;
    raw.reserve(sets.size());
    for (auto& s : sets) raw.push_back(s.native_handle());

    VkPipelineBindPoint point = pipeline.type() == Pipeline::Type::Compute
        ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS;
    vkCmdBindDescriptorSets(cmd_, point, pipeline.native_layout_handle(),
                             first_set, static_cast<uint32_t>(raw.size()), raw.data(),
                             static_cast<uint32_t>(dynamic_offsets.size()),
                             dynamic_offsets.data());
}

void CommandBuffer::push_constants_raw(const Pipeline& pipeline,
                                        VkShaderStageFlags stages,
                                        const void* data, uint32_t size, uint32_t offset) {
    vkCmdPushConstants(cmd_, pipeline.native_layout_handle(), stages, offset, size, data);
}

// ---- Draw ----

void CommandBuffer::draw(uint32_t vertex_count, uint32_t instance_count,
                          uint32_t first_vertex, uint32_t first_instance) {
    VKE_ASSERT(state_ == State::Recording, "draw outside recording");
    vkCmdDraw(cmd_, vertex_count, instance_count, first_vertex, first_instance);
}

void CommandBuffer::draw_indexed(uint32_t index_count, uint32_t instance_count,
                                  uint32_t first_index, int32_t vertex_offset,
                                  uint32_t first_instance) {
    VKE_ASSERT(state_ == State::Recording, "draw_indexed outside recording");
    vkCmdDrawIndexed(cmd_, index_count, instance_count, first_index, vertex_offset, first_instance);
}

void CommandBuffer::draw_mesh_tasks(uint32_t x, uint32_t y, uint32_t z) {
    VKE_ASSERT(state_ == State::Recording, "draw_mesh_tasks outside recording");
    pfn_draw_mesh_tasks_(cmd_, x, y, z);
}

void CommandBuffer::draw_mesh_tasks_indirect(const Buffer& buffer, VkDeviceSize offset,
                                              uint32_t draw_count, uint32_t stride) {
    VKE_ASSERT(state_ == State::Recording, "draw_mesh_tasks_indirect outside recording");
    pfn_draw_mesh_tasks_indirect_(cmd_, buffer.native_handle(), offset, draw_count, stride);
}

void CommandBuffer::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    VKE_ASSERT(state_ == State::Recording, "dispatch outside recording");
    vkCmdDispatch(cmd_, x, y, z);
}

// ---- Buffer / Index ----

void CommandBuffer::bind_vertex_buffers(std::span<const Buffer* const> buffers,
                                         std::span<const VkDeviceSize> offsets,
                                         uint32_t first_binding) {
    std::vector<VkBuffer> raw;
    raw.reserve(buffers.size());
    for (auto* b : buffers) raw.push_back(b->native_handle());
    vkCmdBindVertexBuffers(cmd_, first_binding, static_cast<uint32_t>(raw.size()),
                            raw.data(), offsets.data());
}

void CommandBuffer::bind_index_buffer(const Buffer& buffer, VkDeviceSize offset,
                                       VkIndexType type) {
    vkCmdBindIndexBuffer(cmd_, buffer.native_handle(), offset, type);
}

// ---- Copy ----

void CommandBuffer::copy_buffer(const Buffer& src, Buffer& dst,
                                 VkDeviceSize src_offset, VkDeviceSize dst_offset,
                                 VkDeviceSize size) {
    VkBufferCopy2 region{
        .sType     = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size      = (size == VK_WHOLE_SIZE) ? src.size() - src_offset : size,
    };
    VkCopyBufferInfo2 ci{
        .sType       = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
        .srcBuffer   = src.native_handle(),
        .dstBuffer   = dst.native_handle(),
        .regionCount = 1,
        .pRegions    = &region,
    };
    vkCmdCopyBuffer2(cmd_, &ci);
}

void CommandBuffer::copy_buffer_to_image(const Buffer& src, Image& dst,
                                          const VkBufferImageCopy2& region) {
    VkCopyBufferToImageInfo2 ci{
        .sType          = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
        .srcBuffer      = src.native_handle(),
        .dstImage       = dst.native_handle(),
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount    = 1,
        .pRegions       = &region,
    };
    vkCmdCopyBufferToImage2(cmd_, &ci);
}

void CommandBuffer::copy_image_to_buffer(const Image& src, Buffer& dst,
                                          const VkBufferImageCopy2& region) {
    VkCopyImageToBufferInfo2 ci{
        .sType          = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
        .srcImage       = src.native_handle(),
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstBuffer      = dst.native_handle(),
        .regionCount    = 1,
        .pRegions       = &region,
    };
    vkCmdCopyImageToBuffer2(cmd_, &ci);
}

void CommandBuffer::blit_image(const Image& src, Image& dst,
                                const VkImageBlit2& region, VkFilter filter) {
    VkBlitImageInfo2 bi{
        .sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
        .srcImage       = src.native_handle(),
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstImage       = dst.native_handle(),
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount    = 1,
        .pRegions       = &region,
        .filter         = filter,
    };
    vkCmdBlitImage2(cmd_, &bi);
}

// ---- Debug Labels ----

void CommandBuffer::begin_debug_label(const char* name, float r, float g, float b) {
#ifdef VKE_ENABLE_VALIDATION
    if (!pfn_begin_debug_label_) return;
    VkDebugUtilsLabelEXT label{
        .sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pLabelName = name,
        .color      = {r, g, b, 1.0f},
    };
    pfn_begin_debug_label_(cmd_, &label);
#else
    (void)name; (void)r; (void)g; (void)b;
#endif
}

void CommandBuffer::end_debug_label() {
#ifdef VKE_ENABLE_VALIDATION
    if (pfn_end_debug_label_) pfn_end_debug_label_(cmd_);
#endif
}

void CommandBuffer::insert_debug_label(const char* name) {
#ifdef VKE_ENABLE_VALIDATION
    if (!pfn_insert_debug_label_) return;
    VkDebugUtilsLabelEXT label{
        .sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pLabelName = name,
        .color      = {1.f, 1.f, 1.f, 1.f},
    };
    pfn_insert_debug_label_(cmd_, &label);
#else
    (void)name;
#endif
}

} // namespace vke
