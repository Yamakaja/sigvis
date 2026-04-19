#pragma once
#include <atomic>
#include <cstdint>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include "buffer.hpp"

namespace vke {

enum class ImageUsage : uint32_t {
    ColorAttachment = 1 << 0,
    DepthAttachment = 1 << 1,
    Sampled         = 1 << 2,
    Storage         = 1 << 3,
    TransferSrc     = 1 << 4,
    TransferDst     = 1 << 5,
};
inline ImageUsage operator|(ImageUsage a, ImageUsage b) {
    return static_cast<ImageUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline bool operator&(ImageUsage a, ImageUsage b) {
    return static_cast<uint32_t>(a) & static_cast<uint32_t>(b);
}

struct ImageCreateInfo {
    uint32_t      width         = 1;
    uint32_t      height        = 1;
    uint32_t      depth         = 1;
    uint32_t      mip_levels    = 1;
    uint32_t      array_layers  = 1;
    VkFormat      format        = VK_FORMAT_R8G8B8A8_UNORM;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    ImageUsage    usage         = ImageUsage::ColorAttachment;
    VkImageLayout initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    MemoryDomain  domain        = MemoryDomain::Device;
    const char*   debug_name    = nullptr;
};

class Image {
public:
    Image() = default;
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&&) noexcept;
    Image& operator=(Image&&) noexcept;

    explicit operator bool() const noexcept { return image_ != VK_NULL_HANDLE; }

    uint32_t  width()  const noexcept { return width_; }
    uint32_t  height() const noexcept { return height_; }
    uint32_t  depth()  const noexcept { return depth_; }
    VkFormat  format() const noexcept { return format_; }
    ImageUsage usage() const noexcept { return usage_; }

    VkImageLayout current_layout() const noexcept {
        return layout_.load(std::memory_order_relaxed);
    }
    void set_assumed_layout(VkImageLayout layout) noexcept {
        layout_.store(layout, std::memory_order_relaxed);
    }

    VkImageView create_view(const VkImageViewCreateInfo& info);

    std::vector<uint8_t> download() const;
    std::vector<uint8_t> download_rgba8() const;

    VkImage     native_handle()       const noexcept { return image_; }
    VkImageView native_default_view() const noexcept { return default_view_; }

private:
    friend class Context;
    friend class CommandBuffer;

    Image(VmaAllocator allocator, VkDevice device,
          VkImage image, VmaAllocation alloc, VkImageView default_view,
          uint32_t w, uint32_t h, uint32_t d,
          VkFormat format, ImageUsage usage, VkImageLayout initial_layout,
          VkQueue transfer_queue, VkCommandPool transfer_pool)
        : allocator_(allocator), device_(device),
          image_(image), alloc_(alloc), default_view_(default_view),
          width_(w), height_(h), depth_(d),
          format_(format), usage_(usage), layout_(initial_layout),
          transfer_queue_(transfer_queue), transfer_pool_(transfer_pool) {}

    VmaAllocator               allocator_      = nullptr;
    VkDevice                   device_         = VK_NULL_HANDLE;
    VkImage                    image_          = VK_NULL_HANDLE;
    VmaAllocation              alloc_          = nullptr;
    VkImageView                default_view_   = VK_NULL_HANDLE;
    uint32_t                   width_          = 0;
    uint32_t                   height_         = 0;
    uint32_t                   depth_          = 0;
    VkFormat                   format_         = VK_FORMAT_UNDEFINED;
    ImageUsage                 usage_          = ImageUsage::ColorAttachment;
    mutable std::atomic<VkImageLayout> layout_{VK_IMAGE_LAYOUT_UNDEFINED};
    VkQueue                    transfer_queue_ = VK_NULL_HANDLE;
    VkCommandPool              transfer_pool_  = VK_NULL_HANDLE;
};

} // namespace vke
