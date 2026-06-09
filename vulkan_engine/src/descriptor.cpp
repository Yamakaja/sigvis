#include <vke/descriptor.hpp>
#include <vke/buffer.hpp>
#include <vke/image.hpp>
#include <vke/error.hpp>

namespace vke {

// ---- DescriptorLayout ----

DescriptorLayout::~DescriptorLayout() {
    if (pool_   != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, pool_, nullptr);
    if (layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
}

DescriptorLayout::DescriptorLayout(DescriptorLayout&& o) noexcept
    : device_(o.device_), layout_(o.layout_), pool_(o.pool_)
{
    o.device_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE;
    o.pool_   = VK_NULL_HANDLE;
}

DescriptorLayout& DescriptorLayout::operator=(DescriptorLayout&& o) noexcept {
    if (this != &o) {
        if (pool_   != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, pool_, nullptr);
        if (layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
        device_ = o.device_;
        layout_ = o.layout_;
        pool_   = o.pool_;
        o.device_ = VK_NULL_HANDLE;
        o.layout_ = VK_NULL_HANDLE;
        o.pool_   = VK_NULL_HANDLE;
    }
    return *this;
}

DescriptorSet DescriptorLayout::allocate_set(const char* debug_name) {
    VkDescriptorSetAllocateInfo ai{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = pool_,
        .descriptorSetCount = 1,
        .pSetLayouts        = &layout_,
    };
    VkDescriptorSet set;
    VKE_CHECK(vkAllocateDescriptorSets(device_, &ai, &set));
    (void)debug_name;
    return DescriptorSet(device_, set);
}

void DescriptorLayout::reset_pool() {
    VKE_CHECK(vkResetDescriptorPool(device_, pool_, 0));
}

// ---- DescriptorSet ----

DescriptorSetWriter DescriptorSet::write() {
    return DescriptorSetWriter(*this);
}

// ---- DescriptorSetWriter ----


DescriptorSetWriter& DescriptorSetWriter::bind_uniform_buffer(
    uint32_t binding, const Buffer& buf, VkDeviceSize offset, VkDeviceSize range)
{
    buffer_infos_.push_back({ buf.native_handle(), offset, range });
    writes_.push_back(VkWriteDescriptorSet{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = set_.native_handle(),
        .dstBinding      = binding,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo     = &buffer_infos_.back(),
    });
    return *this;
}

DescriptorSetWriter& DescriptorSetWriter::bind_storage_buffer(
    uint32_t binding, const Buffer& buf, VkDeviceSize offset, VkDeviceSize range)
{
    buffer_infos_.push_back({ buf.native_handle(), offset, range });
    writes_.push_back(VkWriteDescriptorSet{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = set_.native_handle(),
        .dstBinding      = binding,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo     = &buffer_infos_.back(),
    });
    return *this;
}

DescriptorSetWriter& DescriptorSetWriter::bind_storage_buffer_dynamic(
    uint32_t binding, const Buffer& buf, VkDeviceSize offset, VkDeviceSize range)
{
    buffer_infos_.push_back({ buf.native_handle(), offset, range });
    writes_.push_back(VkWriteDescriptorSet{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = set_.native_handle(),
        .dstBinding      = binding,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
        .pBufferInfo     = &buffer_infos_.back(),
    });
    return *this;
}

DescriptorSetWriter& DescriptorSetWriter::bind_combined_image_sampler(
    uint32_t binding, const Image& image, const Sampler& sampler, VkImageLayout layout)
{
    image_infos_.push_back({
        .sampler     = sampler.native_handle(),
        .imageView   = image.native_default_view(),
        .imageLayout = layout,
    });
    writes_.push_back(VkWriteDescriptorSet{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = set_.native_handle(),
        .dstBinding      = binding,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo      = &image_infos_.back(),
    });
    return *this;
}

DescriptorSetWriter& DescriptorSetWriter::bind_storage_image(
    uint32_t binding, const Image& image, VkImageLayout layout)
{
    image_infos_.push_back({
        .sampler     = VK_NULL_HANDLE,
        .imageView   = image.native_default_view(),
        .imageLayout = layout,
    });
    writes_.push_back(VkWriteDescriptorSet{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = set_.native_handle(),
        .dstBinding      = binding,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo      = &image_infos_.back(),
    });
    return *this;
}

void DescriptorSetWriter::commit() {
    if (committed_) return;
    committed_ = true;
    if (writes_.empty()) return;

    // The bind_* helpers stored &buffer_infos_.back() / &image_infos_.back() into each
    // write, but later push_back()s may have reallocated those vectors, dangling the
    // earlier pointers. The vectors are stable now, so re-point each write from them in
    // bind order (a write is a buffer write iff it set pBufferInfo, else an image write).
    size_t bi = 0, ii = 0;
    for (auto& w : writes_) {
        if (w.pBufferInfo)      w.pBufferInfo = &buffer_infos_[bi++];
        else if (w.pImageInfo)  w.pImageInfo  = &image_infos_[ii++];
    }
    vkUpdateDescriptorSets(set_.device_, static_cast<uint32_t>(writes_.size()),
                            writes_.data(), 0, nullptr);
}

} // namespace vke
