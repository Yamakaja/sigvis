#include <vke/sampler.hpp>
#include <vke/error.hpp>

namespace vke {

Sampler::Impl::~Impl() {
    if (sampler != VK_NULL_HANDLE)
        vkDestroySampler(device, sampler, nullptr);
}

} // namespace vke
