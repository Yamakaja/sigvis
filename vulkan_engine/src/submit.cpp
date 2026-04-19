#include <vke/submit.hpp>
#include <vke/error.hpp>

namespace vke {

SubmitHandle::~SubmitHandle() {
    // Fence is recycled by the Context pool — do not destroy here.
    // Abandoning an in-flight handle leaks a fence until the Context is destroyed.
}

SubmitHandle::SubmitHandle(SubmitHandle&& o) noexcept
    : device_(o.device_), fence_(o.fence_)
{
    o.device_ = VK_NULL_HANDLE;
    o.fence_  = VK_NULL_HANDLE;
}

SubmitHandle& SubmitHandle::operator=(SubmitHandle&& o) noexcept {
    if (this != &o) {
        device_ = o.device_;
        fence_  = o.fence_;
        o.device_ = VK_NULL_HANDLE;
        o.fence_  = VK_NULL_HANDLE;
    }
    return *this;
}

bool SubmitHandle::is_complete() const noexcept {
    if (fence_ == VK_NULL_HANDLE) return true;
    return vkGetFenceStatus(device_, fence_) == VK_SUCCESS;
}

void SubmitHandle::wait() {
    if (fence_ == VK_NULL_HANDLE) return;
    VKE_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));
}

} // namespace vke
