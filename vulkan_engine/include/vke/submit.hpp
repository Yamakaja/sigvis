#pragma once
#include <vulkan/vulkan.h>

namespace vke {

class SubmitHandle {
public:
    SubmitHandle() = default;
    ~SubmitHandle();

    SubmitHandle(const SubmitHandle&) = delete;
    SubmitHandle& operator=(const SubmitHandle&) = delete;
    SubmitHandle(SubmitHandle&&) noexcept;
    SubmitHandle& operator=(SubmitHandle&&) noexcept;

    bool is_valid()    const noexcept { return fence_ != VK_NULL_HANDLE; }
    bool is_complete() const noexcept; // non-blocking poll
    void wait();                       // blocks until complete

    VkFence native_fence() const noexcept { return fence_; }
    void    release_fence() noexcept { fence_ = VK_NULL_HANDLE; }

private:
    friend class Context;
    SubmitHandle(VkDevice device, VkFence fence)
        : device_(device), fence_(fence) {}

    VkDevice device_ = VK_NULL_HANDLE;
    VkFence  fence_  = VK_NULL_HANDLE;
};

} // namespace vke
