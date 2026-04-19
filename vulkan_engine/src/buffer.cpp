#include <vke/buffer.hpp>
#include <vke/error.hpp>
#include <cstring>

namespace vke {

Buffer::~Buffer() {
    if (buffer_ != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator_, buffer_, alloc_);
}

Buffer::Buffer(Buffer&& o) noexcept
    : allocator_(o.allocator_), device_(o.device_),
      buffer_(o.buffer_), alloc_(o.alloc_),
      size_(o.size_), usage_(o.usage_), domain_(o.domain_),
      mapped_ptr_(o.mapped_ptr_),
      transfer_queue_(o.transfer_queue_), transfer_pool_(o.transfer_pool_)
{
    o.allocator_  = nullptr;
    o.buffer_     = VK_NULL_HANDLE;
    o.alloc_      = nullptr;
    o.mapped_ptr_ = nullptr;
}

Buffer& Buffer::operator=(Buffer&& o) noexcept {
    if (this != &o) {
        if (buffer_ != VK_NULL_HANDLE)
            vmaDestroyBuffer(allocator_, buffer_, alloc_);
        allocator_      = o.allocator_;
        device_         = o.device_;
        buffer_         = o.buffer_;
        alloc_          = o.alloc_;
        size_           = o.size_;
        usage_          = o.usage_;
        domain_         = o.domain_;
        mapped_ptr_     = o.mapped_ptr_;
        transfer_queue_ = o.transfer_queue_;
        transfer_pool_  = o.transfer_pool_;
        o.allocator_  = nullptr;
        o.buffer_     = VK_NULL_HANDLE;
        o.alloc_      = nullptr;
        o.mapped_ptr_ = nullptr;
    }
    return *this;
}

VkDeviceAddress Buffer::device_address() const {
    VKE_ASSERT(buffer_ != VK_NULL_HANDLE, "device_address on null buffer");
    VkBufferDeviceAddressInfo info{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buffer_,
    };
    return vkGetBufferDeviceAddress(device_, &info);
}

// One-shot command buffer helper for staging transfers
namespace {

struct OneShot {
    VkDevice        device;
    VkCommandPool   pool;
    VkCommandBuffer cmd;

    explicit OneShot(VkDevice d, VkCommandPool p) : device(d), pool(p) {
        VkCommandBufferAllocateInfo ai{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = pool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
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
        VkCommandBufferSubmitInfo csi{
            .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = cmd,
        };
        VkSubmitInfo2 si{
            .sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos    = &csi,
        };
        VkFenceCreateInfo fi{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VkFence fence;
        VKE_CHECK(vkCreateFence(device, &fi, nullptr, &fence));
        VKE_CHECK(vkQueueSubmit2(queue, 1, &si, fence));
        VKE_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, pool, 1, &cmd);
    }
};

VkBuffer create_staging(VmaAllocator allocator, VkDeviceSize size,
                         VmaAllocation& out_alloc, void*& out_ptr) {
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    };
    VmaAllocationCreateInfo aci{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VmaAllocationInfo info;
    VkBuffer buf;
    VKE_CHECK(vmaCreateBuffer(allocator, &bci, &aci, &buf, &out_alloc, &info));
    out_ptr = info.pMappedData;
    return buf;
}

} // anonymous namespace

void Buffer::upload_raw(const void* data, VkDeviceSize bytes, VkDeviceSize offset) {
    VKE_ASSERT(buffer_ != VK_NULL_HANDLE, "upload on null buffer");
    VKE_ASSERT(offset + bytes <= size_, "upload out of buffer range");

    if (domain_ != MemoryDomain::Device) {
        std::memcpy(static_cast<uint8_t*>(mapped_ptr_) + offset, data, bytes);
        vmaFlushAllocation(allocator_, alloc_, offset, bytes);
        return;
    }

    VKE_ASSERT(transfer_queue_ != VK_NULL_HANDLE,
               "Device-local buffer upload requires a transfer queue");

    VmaAllocation staging_alloc;
    void* staging_ptr;
    VkBuffer staging = create_staging(allocator_, bytes, staging_alloc, staging_ptr);
    std::memcpy(staging_ptr, data, bytes);
    vmaFlushAllocation(allocator_, staging_alloc, 0, bytes);

    {
        OneShot cmd(device_, transfer_pool_);
        VkBufferCopy2 region{
            .sType     = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            .srcOffset = 0,
            .dstOffset = offset,
            .size      = bytes,
        };
        VkCopyBufferInfo2 ci{
            .sType       = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
            .srcBuffer   = staging,
            .dstBuffer   = buffer_,
            .regionCount = 1,
            .pRegions    = &region,
        };
        vkCmdCopyBuffer2(cmd.cmd, &ci);
        cmd.submit_and_wait(transfer_queue_);
    }

    vmaDestroyBuffer(allocator_, staging, staging_alloc);
}

std::vector<uint8_t> Buffer::download_raw(VkDeviceSize offset, VkDeviceSize bytes) const {
    VKE_ASSERT(buffer_ != VK_NULL_HANDLE, "download on null buffer");
    if (bytes == VK_WHOLE_SIZE) bytes = size_ - offset;
    VKE_ASSERT(offset + bytes <= size_, "download out of buffer range");

    if (domain_ != MemoryDomain::Device) {
        vmaInvalidateAllocation(allocator_, alloc_, offset, bytes);
        std::vector<uint8_t> result(bytes);
        std::memcpy(result.data(), static_cast<const uint8_t*>(mapped_ptr_) + offset, bytes);
        return result;
    }

    // Readback staging buffer
    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    VmaAllocationCreateInfo aci{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VmaAllocationInfo alloc_info;
    VmaAllocation readback_alloc;
    VkBuffer readback_buf;
    VKE_CHECK(vmaCreateBuffer(allocator_, &bci, &aci, &readback_buf, &readback_alloc, &alloc_info));

    {
        OneShot cmd(device_, transfer_pool_);
        VkBufferCopy2 region{
            .sType     = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            .srcOffset = offset,
            .dstOffset = 0,
            .size      = bytes,
        };
        VkCopyBufferInfo2 ci{
            .sType       = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
            .srcBuffer   = buffer_,
            .dstBuffer   = readback_buf,
            .regionCount = 1,
            .pRegions    = &region,
        };
        vkCmdCopyBuffer2(cmd.cmd, &ci);
        cmd.submit_and_wait(transfer_queue_);
    }

    vmaInvalidateAllocation(allocator_, readback_alloc, 0, bytes);
    std::vector<uint8_t> result(bytes);
    std::memcpy(result.data(), alloc_info.pMappedData, bytes);
    vmaDestroyBuffer(allocator_, readback_buf, readback_alloc);
    return result;
}

} // namespace vke
