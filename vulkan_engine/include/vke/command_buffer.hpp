#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <vulkan/vulkan.h>
#include "buffer.hpp"
#include "descriptor.hpp"
#include "pipeline.hpp"

namespace vke {

class Image;
class Context;

class CommandBuffer {
public:
    CommandBuffer() = default;
    ~CommandBuffer();

    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;
    CommandBuffer(CommandBuffer&&) noexcept;
    CommandBuffer& operator=(CommandBuffer&&) noexcept;

    explicit operator bool() const noexcept { return cmd_ != VK_NULL_HANDLE; }

    void begin();
    void end();

    // ---- Barriers (synchronization2) ----

    struct ImageBarrierInfo {
        Image&                image;
        VkImageLayout         new_layout;
        VkPipelineStageFlags2 src_stage_mask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkAccessFlags2        src_access_mask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        VkPipelineStageFlags2 dst_stage_mask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkAccessFlags2        dst_access_mask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    };

    void image_barrier(const ImageBarrierInfo& info);
    void image_barriers(std::span<const ImageBarrierInfo> infos);

    struct BufferBarrierInfo {
        const Buffer&         buffer;
        VkPipelineStageFlags2 src_stage_mask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkAccessFlags2        src_access_mask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        VkPipelineStageFlags2 dst_stage_mask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkAccessFlags2        dst_access_mask = VK_ACCESS_2_MEMORY_READ_BIT;
        VkDeviceSize          offset          = 0;
        VkDeviceSize          size            = VK_WHOLE_SIZE;
    };

    void buffer_barrier(const BufferBarrierInfo& info);
    void full_barrier();

    // ---- Dynamic Rendering ----

    struct ColorAttachmentInfo {
        const Image*        image       = nullptr;
        VkImageView         view        = VK_NULL_HANDLE; // null = default view
        VkAttachmentLoadOp  load_op     = VK_ATTACHMENT_LOAD_OP_CLEAR;
        VkAttachmentStoreOp store_op    = VK_ATTACHMENT_STORE_OP_STORE;
        VkClearColorValue   clear_value = {.float32 = {0.f, 0.f, 0.f, 1.f}};
    };

    struct DepthAttachmentInfo {
        const Image*        image         = nullptr;
        VkAttachmentLoadOp  load_op       = VK_ATTACHMENT_LOAD_OP_CLEAR;
        VkAttachmentStoreOp store_op      = VK_ATTACHMENT_STORE_OP_STORE;
        float               clear_depth   = 1.0f;
        uint32_t            clear_stencil = 0;
    };

    struct RenderingInfo {
        uint32_t x = 0, y = 0;
        uint32_t width  = 0; // 0 = first color attachment's width
        uint32_t height = 0;
        uint32_t layer_count = 1;
        std::span<const ColorAttachmentInfo>   color_attachments;
        std::optional<DepthAttachmentInfo>     depth_attachment;
        bool auto_layout_transitions = true;
    };

    void begin_rendering(const RenderingInfo& info);
    void end_rendering();

    // ---- Pipeline + State ----

    void bind_pipeline(const Pipeline& pipeline);
    void set_viewport(float x, float y, float w, float h,
                      float min_depth = 0.0f, float max_depth = 1.0f);
    void set_scissor(int32_t x, int32_t y, uint32_t w, uint32_t h);

    void bind_descriptor_sets(const Pipeline& pipeline,
                               std::span<const DescriptorSet> sets,
                               uint32_t first_set = 0);
    void bind_descriptor_sets(const Pipeline& pipeline,
                               std::span<const DescriptorSet> sets,
                               std::span<const uint32_t> dynamic_offsets,
                               uint32_t first_set = 0);

    template<typename T>
    void push_constants(const Pipeline& pipeline,
                        VkShaderStageFlags stages,
                        const T& data,
                        uint32_t offset = 0) {
        push_constants_raw(pipeline, stages, &data, sizeof(T), offset);
    }

    // ---- Draw ----

    void draw(uint32_t vertex_count, uint32_t instance_count = 1,
              uint32_t first_vertex = 0, uint32_t first_instance = 0);
    void draw_indexed(uint32_t index_count, uint32_t instance_count = 1,
                      uint32_t first_index = 0, int32_t vertex_offset = 0,
                      uint32_t first_instance = 0);
    void draw_mesh_tasks(uint32_t x, uint32_t y = 1, uint32_t z = 1);
    void draw_mesh_tasks_indirect(const Buffer& buffer, VkDeviceSize offset,
                                  uint32_t draw_count, uint32_t stride);
    void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);

    // ---- Buffer / Index Binding ----

    void bind_vertex_buffers(std::span<const Buffer* const> buffers,
                              std::span<const VkDeviceSize> offsets,
                              uint32_t first_binding = 0);
    void bind_index_buffer(const Buffer& buffer,
                            VkDeviceSize offset = 0,
                            VkIndexType type    = VK_INDEX_TYPE_UINT32);

    // ---- Copy ----

    void copy_buffer(const Buffer& src, Buffer& dst,
                     VkDeviceSize src_offset = 0,
                     VkDeviceSize dst_offset = 0,
                     VkDeviceSize size       = VK_WHOLE_SIZE);
    void copy_buffer_to_image(const Buffer& src, Image& dst,
                               const VkBufferImageCopy2& region);
    void copy_image_to_buffer(const Image& src, Buffer& dst,
                               const VkBufferImageCopy2& region);
    void blit_image(const Image& src, Image& dst,
                    const VkImageBlit2& region,
                    VkFilter filter = VK_FILTER_LINEAR);

    // ---- Debug Labels ----

    void begin_debug_label(const char* name, float r = 1, float g = 1, float b = 1);
    void end_debug_label();
    void insert_debug_label(const char* name);

    VkCommandBuffer native_handle() const noexcept { return cmd_; }

    // Relinquish ownership of the raw handles so the caller can manage lifetime.
    // After this the CommandBuffer destructor is a no-op.
    struct RawHandles { VkDevice device; VkCommandPool pool; VkCommandBuffer cmd; };
    RawHandles detach() noexcept {
        RawHandles h{ device_, pool_, cmd_ };
        device_ = VK_NULL_HANDLE;
        pool_   = VK_NULL_HANDLE;
        cmd_    = VK_NULL_HANDLE;
        return h;
    }

private:
    friend class Context;

    CommandBuffer(VkDevice device, VkCommandPool pool, VkCommandBuffer cmd);

    void push_constants_raw(const Pipeline& pipeline, VkShaderStageFlags stages,
                             const void* data, uint32_t size, uint32_t offset);

    enum class State { Initial, Recording, Executable, Invalid };

    VkDevice        device_ = VK_NULL_HANDLE;
    VkCommandPool   pool_   = VK_NULL_HANDLE;
    VkCommandBuffer cmd_    = VK_NULL_HANDLE;
    State           state_  = State::Initial;
    bool            in_rendering_ = false;

    PFN_vkCmdDrawMeshTasksEXT         pfn_draw_mesh_tasks_          = nullptr;
    PFN_vkCmdDrawMeshTasksIndirectEXT pfn_draw_mesh_tasks_indirect_ = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT  pfn_begin_debug_label_        = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT    pfn_end_debug_label_          = nullptr;
    PFN_vkCmdInsertDebugUtilsLabelEXT pfn_insert_debug_label_       = nullptr;
};

} // namespace vke
