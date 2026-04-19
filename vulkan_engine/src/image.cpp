#include <vke/image.hpp>
#include <vke/error.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vke {

Image::~Image() {
    if (default_view_ != VK_NULL_HANDLE)
        vkDestroyImageView(device_, default_view_, nullptr);
    if (image_ != VK_NULL_HANDLE)
        vmaDestroyImage(allocator_, image_, alloc_);
}

Image::Image(Image&& o) noexcept
    : allocator_(o.allocator_), device_(o.device_),
      image_(o.image_), alloc_(o.alloc_), default_view_(o.default_view_),
      width_(o.width_), height_(o.height_), depth_(o.depth_),
      format_(o.format_), usage_(o.usage_),
      transfer_queue_(o.transfer_queue_), transfer_pool_(o.transfer_pool_)
{
    layout_.store(o.layout_.load());
    o.allocator_    = nullptr;
    o.image_        = VK_NULL_HANDLE;
    o.alloc_        = nullptr;
    o.default_view_ = VK_NULL_HANDLE;
}

Image& Image::operator=(Image&& o) noexcept {
    if (this != &o) {
        if (default_view_ != VK_NULL_HANDLE)
            vkDestroyImageView(device_, default_view_, nullptr);
        if (image_ != VK_NULL_HANDLE)
            vmaDestroyImage(allocator_, image_, alloc_);
        allocator_      = o.allocator_;
        device_         = o.device_;
        image_          = o.image_;
        alloc_          = o.alloc_;
        default_view_   = o.default_view_;
        width_          = o.width_;
        height_         = o.height_;
        depth_          = o.depth_;
        format_         = o.format_;
        usage_          = o.usage_;
        transfer_queue_ = o.transfer_queue_;
        transfer_pool_  = o.transfer_pool_;
        layout_.store(o.layout_.load());
        o.allocator_    = nullptr;
        o.image_        = VK_NULL_HANDLE;
        o.alloc_        = nullptr;
        o.default_view_ = VK_NULL_HANDLE;
    }
    return *this;
}

VkImageView Image::create_view(const VkImageViewCreateInfo& info) {
    VkImageView view;
    VKE_CHECK(vkCreateImageView(device_, &info, nullptr, &view));
    return view;
}

// Determine image aspect from format
static VkImageAspectFlags aspect_for_format(VkFormat fmt) {
    switch (fmt) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

// Bytes-per-pixel for simple formats used in download
static uint32_t bytes_per_pixel(VkFormat fmt) {
    switch (fmt) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_R32_SFLOAT:
        return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 16;
    default:
        return 4; // best-effort fallback
    }
}

namespace {

struct OneShot {
    VkDevice device; VkCommandPool pool; VkCommandBuffer cmd;
    OneShot(VkDevice d, VkCommandPool p) : device(d), pool(p) {
        VkCommandBufferAllocateInfo ai{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VKE_CHECK(vkAllocateCommandBuffers(device, &ai, &cmd));
        VkCommandBufferBeginInfo bi{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VKE_CHECK(vkBeginCommandBuffer(cmd, &bi));
    }
    void submit_and_wait(VkQueue queue) {
        VKE_CHECK(vkEndCommandBuffer(cmd));
        VkCommandBufferSubmitInfo csi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmd };
        VkSubmitInfo2 si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 1, .pCommandBufferInfos = &csi };
        VkFenceCreateInfo fi{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VkFence fence;
        VKE_CHECK(vkCreateFence(device, &fi, nullptr, &fence));
        VKE_CHECK(vkQueueSubmit2(queue, 1, &si, fence));
        VKE_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, pool, 1, &cmd);
    }
};

} // anonymous namespace

std::vector<uint8_t> Image::download() const {
    VKE_ASSERT(image_ != VK_NULL_HANDLE, "download on null image");
    VKE_ASSERT(depth_ == 1, "download only supported for 2D images");

    uint32_t bpp  = bytes_per_pixel(format_);
    VkDeviceSize row_pitch = static_cast<VkDeviceSize>(width_) * bpp;
    VkDeviceSize total     = row_pitch * height_;

    // Create readback buffer
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = total,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    VmaAllocationCreateInfo aci{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VmaAllocationInfo alloc_info;
    VmaAllocation staging_alloc;
    VkBuffer staging_buf;
    VKE_CHECK(vmaCreateBuffer(allocator_, &bci, &aci, &staging_buf, &staging_alloc, &alloc_info));

    // Record: transition → copy → transition back
    // We need queue/pool — stored externally per image.
    // For simplicity, we store transfer_queue_ / transfer_pool_ in Image just like Buffer.
    // However we didn't add those yet — handle via context's graphics queue for now.
    // This is set via Image::transfer_queue_ / transfer_pool_ added below.

    {
        OneShot cmd(device_, transfer_pool_);

        // Transition to TRANSFER_SRC if not already
        VkImageLayout old_layout = layout_.load();
        if (old_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            VkImageMemoryBarrier2 barrier{
                .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask     = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask    = VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask     = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask    = VK_ACCESS_2_TRANSFER_READ_BIT,
                .oldLayout        = old_layout,
                .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .image            = image_,
                .subresourceRange = {
                    .aspectMask = aspect_for_format(format_),
                    .levelCount = 1, .layerCount = 1,
                },
            };
            VkDependencyInfo di{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
            vkCmdPipelineBarrier2(cmd.cmd, &di);
        }

        VkBufferImageCopy2 region{
            .sType             = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
            .bufferOffset      = 0,
            .bufferRowLength   = 0,
            .bufferImageHeight = 0,
            .imageSubresource  = { .aspectMask = aspect_for_format(format_),
                                   .layerCount = 1 },
            .imageOffset       = {0, 0, 0},
            .imageExtent       = { width_, height_, 1 },
        };
        VkCopyImageToBufferInfo2 ci{
            .sType       = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
            .srcImage    = image_,
            .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .dstBuffer   = staging_buf,
            .regionCount = 1,
            .pRegions    = &region,
        };
        vkCmdCopyImageToBuffer2(cmd.cmd, &ci);

        // Transition back
        if (old_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
            old_layout != VK_IMAGE_LAYOUT_UNDEFINED) {
            VkImageMemoryBarrier2 barrier{
                .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask     = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask    = VK_ACCESS_2_TRANSFER_READ_BIT,
                .dstStageMask     = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .dstAccessMask    = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .newLayout        = old_layout,
                .image            = image_,
                .subresourceRange = {
                    .aspectMask = aspect_for_format(format_),
                    .levelCount = 1, .layerCount = 1,
                },
            };
            VkDependencyInfo di{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier };
            vkCmdPipelineBarrier2(cmd.cmd, &di);
        }

        cmd.submit_and_wait(transfer_queue_);
    }

    vmaInvalidateAllocation(allocator_, staging_alloc, 0, total);
    std::vector<uint8_t> result(total);
    std::memcpy(result.data(), alloc_info.pMappedData, total);
    vmaDestroyBuffer(allocator_, staging_buf, staging_alloc);
    return result;
}

std::vector<uint8_t> Image::download_rgba8() const {
    // For RGBA8 formats, download directly
    if (format_ == VK_FORMAT_R8G8B8A8_UNORM || format_ == VK_FORMAT_R8G8B8A8_SRGB)
        return download();

    // For other formats, download raw and convert
    auto raw = download();
    uint32_t pixel_count = width_ * height_;

    if (format_ == VK_FORMAT_R32_SFLOAT) {
        // Single channel float → grey RGBA8
        std::vector<uint8_t> out(pixel_count * 4);
        auto* src = reinterpret_cast<const float*>(raw.data());
        for (uint32_t i = 0; i < pixel_count; ++i) {
            uint8_t v = static_cast<uint8_t>(std::min(src[i], 1.0f) * 255.0f);
            out[i*4+0] = v; out[i*4+1] = v; out[i*4+2] = v; out[i*4+3] = 255;
        }
        return out;
    }

    if (format_ == VK_FORMAT_R16G16B16A16_SFLOAT) {
        // RGBA16F → RGBA8 (simple clamp, no tone mapping)
        std::vector<uint8_t> out(pixel_count * 4);
        // 16-bit float conversion via memcpy trick isn't portable; use uint16 interpretation
        auto* src = reinterpret_cast<const uint16_t*>(raw.data());
        auto f16_to_f32 = [](uint16_t h) -> float {
            uint32_t sign     = (h >> 15) & 0x1;
            uint32_t exponent = (h >> 10) & 0x1f;
            uint32_t mantissa = h & 0x3ff;
            if (exponent == 0) {
                if (mantissa == 0) return sign ? -0.f : 0.f;
                float m = mantissa / 1024.f;
                return (sign ? -1.f : 1.f) * m * std::pow(2.f, -14.f);
            }
            if (exponent == 31)
                return sign ? -std::numeric_limits<float>::infinity()
                             : std::numeric_limits<float>::infinity();
            float m = 1.f + mantissa / 1024.f;
            return (sign ? -1.f : 1.f) * m * std::pow(2.f, static_cast<float>(exponent) - 15.f);
        };
        for (uint32_t i = 0; i < pixel_count; ++i) {
            for (int c = 0; c < 4; ++c) {
                float v = std::clamp(f16_to_f32(src[i*4+c]), 0.f, 1.f);
                out[i*4+c] = static_cast<uint8_t>(v * 255.f);
            }
        }
        return out;
    }

    throw PreconditionError("download_rgba8: unsupported format");
}

} // namespace vke
