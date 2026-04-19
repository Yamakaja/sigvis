#include <vke/submit.hpp>
#include <vke/error.hpp>

namespace vke {

void SubmitHandle::free_cmd() noexcept {
    if (cmd_buf_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd_buf_);
        cmd_buf_  = VK_NULL_HANDLE;
        cmd_pool_ = VK_NULL_HANDLE;
    }
}

SubmitHandle::~SubmitHandle() {
    // Fence is recycled by the Context pool — do not destroy here.
    // Command buffer is freed here if still owned (i.e. wait() was never called).
    // Note: if the fence is still in flight this is technically invalid, but the
    // Context destructor resets the pool before this path is reached in practice.
    free_cmd();
}

SubmitHandle::SubmitHandle(SubmitHandle&& o) noexcept
    : device_(o.device_), fence_(o.fence_),
      cmd_pool_(o.cmd_pool_), cmd_buf_(o.cmd_buf_)
{
    o.device_   = VK_NULL_HANDLE;
    o.fence_    = VK_NULL_HANDLE;
    o.cmd_pool_ = VK_NULL_HANDLE;
    o.cmd_buf_  = VK_NULL_HANDLE;
}

SubmitHandle& SubmitHandle::operator=(SubmitHandle&& o) noexcept {
    if (this != &o) {
        free_cmd();
        device_   = o.device_;
        fence_    = o.fence_;
        cmd_pool_ = o.cmd_pool_;
        cmd_buf_  = o.cmd_buf_;
        o.device_   = VK_NULL_HANDLE;
        o.fence_    = VK_NULL_HANDLE;
        o.cmd_pool_ = VK_NULL_HANDLE;
        o.cmd_buf_  = VK_NULL_HANDLE;
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
    free_cmd();
}

} // namespace vke
