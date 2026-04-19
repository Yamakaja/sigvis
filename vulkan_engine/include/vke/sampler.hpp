#pragma once
#include <memory>
#include <vulkan/vulkan.h>

namespace vke {

struct SamplerCreateInfo {
    VkFilter             mag_filter        = VK_FILTER_LINEAR;
    VkFilter             min_filter        = VK_FILTER_LINEAR;
    VkSamplerMipmapMode  mipmap_mode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    VkSamplerAddressMode address_u         = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSamplerAddressMode address_v         = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSamplerAddressMode address_w         = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    float                min_lod           = 0.0f;
    float                max_lod           = VK_LOD_CLAMP_NONE;
    bool                 anisotropy_enable = false;
    float                max_anisotropy    = 1.0f;
    const char*          debug_name        = nullptr;
};

// Copyable via shared ownership — last copy destroys the VkSampler.
class Sampler {
public:
    Sampler() = default;

    Sampler(const Sampler&) noexcept = default;
    Sampler& operator=(const Sampler&) noexcept = default;
    Sampler(Sampler&&) noexcept = default;
    Sampler& operator=(Sampler&&) noexcept = default;

    explicit operator bool() const noexcept { return impl_ != nullptr; }

    VkSampler native_handle() const noexcept {
        return impl_ ? impl_->sampler : VK_NULL_HANDLE;
    }

private:
    friend class Context;

    struct Impl {
        VkDevice  device  = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        ~Impl();
    };

    explicit Sampler(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

    std::shared_ptr<Impl> impl_;
};

} // namespace vke
