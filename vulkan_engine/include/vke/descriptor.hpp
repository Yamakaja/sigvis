#pragma once
#include <cstdint>
#include <span>
#include <vector>
#include <vulkan/vulkan.h>
#include "sampler.hpp"

namespace vke {

class Buffer;
class Image;

enum class DescriptorType {
    UniformBuffer,
    StorageBuffer,
    CombinedImageSampler,
    StorageImage,
    UniformBufferDynamic,
    StorageBufferDynamic,
};

struct DescriptorBinding {
    uint32_t           binding = 0;
    DescriptorType     type    = DescriptorType::UniformBuffer;
    uint32_t           count   = 1;
    VkShaderStageFlags stages  = VK_SHADER_STAGE_ALL;
};

struct DescriptorLayoutCreateInfo {
    std::span<const DescriptorBinding> bindings;
    uint32_t    max_sets   = 64;
    const char* debug_name = nullptr;
};

// ---- DescriptorSet (copyable non-owning handle) ----

class DescriptorSetWriter;

class DescriptorSet {
public:
    DescriptorSet() = default;

    explicit operator bool() const noexcept { return set_ != VK_NULL_HANDLE; }

    DescriptorSetWriter write();

    VkDescriptorSet native_handle() const noexcept { return set_; }

private:
    friend class DescriptorLayout;
    friend class DescriptorSetWriter;

    explicit DescriptorSet(VkDevice device, VkDescriptorSet set)
        : device_(device), set_(set) {}

    VkDevice        device_ = VK_NULL_HANDLE;
    VkDescriptorSet set_    = VK_NULL_HANDLE;
};

// ---- DescriptorSetWriter (fluent, auto-commits on destruction) ----

class DescriptorSetWriter {
public:
    explicit DescriptorSetWriter(DescriptorSet& set) : set_(set) {}
    ~DescriptorSetWriter() { if (!committed_) commit(); }

    DescriptorSetWriter(const DescriptorSetWriter&) = delete;
    DescriptorSetWriter& operator=(const DescriptorSetWriter&) = delete;

    DescriptorSetWriter& bind_uniform_buffer(uint32_t binding, const Buffer& buf,
                                              VkDeviceSize offset = 0,
                                              VkDeviceSize range  = VK_WHOLE_SIZE);

    DescriptorSetWriter& bind_storage_buffer(uint32_t binding, const Buffer& buf,
                                              VkDeviceSize offset = 0,
                                              VkDeviceSize range  = VK_WHOLE_SIZE);

    // Dynamic storage buffer: the bound region's base is the supplied offset plus a
    // per-bind dynamic offset (see CommandBuffer::bind_descriptor_sets overload).
    DescriptorSetWriter& bind_storage_buffer_dynamic(uint32_t binding, const Buffer& buf,
                                                     VkDeviceSize offset = 0,
                                                     VkDeviceSize range  = VK_WHOLE_SIZE);

    DescriptorSetWriter& bind_combined_image_sampler(
        uint32_t binding, const Image& image, const Sampler& sampler,
        VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    DescriptorSetWriter& bind_storage_image(uint32_t binding, const Image& image,
                                             VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL);

    void commit();

private:
    DescriptorSet&                    set_;
    std::vector<VkWriteDescriptorSet> writes_;
    // Storage to keep buffer/image info alive until commit()
    std::vector<VkDescriptorBufferInfo> buffer_infos_;
    std::vector<VkDescriptorImageInfo>  image_infos_;
    bool committed_ = false;
};

// ---- DescriptorLayout ----

class DescriptorLayout {
public:
    DescriptorLayout() = default;
    ~DescriptorLayout();

    DescriptorLayout(const DescriptorLayout&) = delete;
    DescriptorLayout& operator=(const DescriptorLayout&) = delete;
    DescriptorLayout(DescriptorLayout&&) noexcept;
    DescriptorLayout& operator=(DescriptorLayout&&) noexcept;

    explicit operator bool() const noexcept { return layout_ != VK_NULL_HANDLE; }

    DescriptorSet allocate_set(const char* debug_name = nullptr);
    void          reset_pool();

    VkDescriptorSetLayout native_handle() const noexcept { return layout_; }

private:
    friend class Context;

    DescriptorLayout(VkDevice device, VkDescriptorSetLayout layout, VkDescriptorPool pool)
        : device_(device), layout_(layout), pool_(pool) {}

    VkDevice              device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool      pool_   = VK_NULL_HANDLE;
};

} // namespace vke
