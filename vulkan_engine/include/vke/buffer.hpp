#pragma once
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace vke {

enum class BufferUsage : uint32_t {
    Vertex      = 1 << 0,
    Index       = 1 << 1,
    Uniform     = 1 << 2,
    Storage     = 1 << 3,
    Indirect    = 1 << 4,
    TransferSrc = 1 << 5,
    TransferDst = 1 << 6,
};
inline BufferUsage operator|(BufferUsage a, BufferUsage b) {
    return static_cast<BufferUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline bool operator&(BufferUsage a, BufferUsage b) {
    return static_cast<uint32_t>(a) & static_cast<uint32_t>(b);
}

enum class MemoryDomain { Device, Host, DeviceHost };

struct BufferCreateInfo {
    VkDeviceSize     size   = 0;
    BufferUsage      usage  = BufferUsage::Storage;
    MemoryDomain     domain = MemoryDomain::Device;
    const char*      debug_name = nullptr;
};

class Buffer {
public:
    Buffer() = default;
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;

    explicit operator bool() const noexcept { return buffer_ != VK_NULL_HANDLE; }

    VkDeviceSize size()   const noexcept { return size_; }
    BufferUsage  usage()  const noexcept { return usage_; }
    MemoryDomain domain() const noexcept { return domain_; }

    template<typename T>
    std::span<T> mapped_as() {
        return {static_cast<T*>(mapped_ptr_), size_ / sizeof(T)};
    }
    template<typename T>
    std::span<const T> mapped_as() const {
        return {static_cast<const T*>(mapped_ptr_), size_ / sizeof(T)};
    }

    template<typename T>
    void upload(std::span<const T> data, VkDeviceSize byte_offset = 0) {
        upload_raw(data.data(), data.size_bytes(), byte_offset);
    }

    template<typename T>
    std::vector<T> download(VkDeviceSize byte_offset = 0,
                            VkDeviceSize byte_count  = VK_WHOLE_SIZE) const {
        auto raw = download_raw(byte_offset, byte_count);
        std::vector<T> result(raw.size() / sizeof(T));
        std::memcpy(result.data(), raw.data(), result.size() * sizeof(T));
        return result;
    }

    VkBuffer        native_handle()  const noexcept { return buffer_; }
    VkDeviceAddress device_address() const;

private:
    friend class Context;

    Buffer(VmaAllocator allocator, VkDevice device,
           VkBuffer buffer, VmaAllocation alloc,
           VkDeviceSize size, BufferUsage usage, MemoryDomain domain,
           void* mapped_ptr,
           VkQueue transfer_queue, VkCommandPool transfer_pool)
        : allocator_(allocator), device_(device),
          buffer_(buffer), alloc_(alloc),
          size_(size), usage_(usage), domain_(domain),
          mapped_ptr_(mapped_ptr),
          transfer_queue_(transfer_queue), transfer_pool_(transfer_pool) {}

    void upload_raw(const void* data, VkDeviceSize bytes, VkDeviceSize offset);
    std::vector<uint8_t> download_raw(VkDeviceSize offset, VkDeviceSize bytes) const;

    VmaAllocator  allocator_       = nullptr;
    VkDevice      device_          = VK_NULL_HANDLE;
    VkBuffer      buffer_          = VK_NULL_HANDLE;
    VmaAllocation alloc_           = nullptr;
    VkDeviceSize  size_            = 0;
    BufferUsage   usage_           = BufferUsage::Storage;
    MemoryDomain  domain_          = MemoryDomain::Device;
    void*         mapped_ptr_      = nullptr;
    VkQueue       transfer_queue_  = VK_NULL_HANDLE;
    VkCommandPool transfer_pool_   = VK_NULL_HANDLE;
};

} // namespace vke
