#pragma once
#include <cstdint>
#include <optional>
#include <vector>
#include <vulkan/vulkan.h>
#include "image.hpp"

namespace vke {

class Context;

// Owns a VkSwapchainKHR and per-image views (wrapped as non-owning vke::Image).
// Requires a Context created with a surface_factory (see ContextCreateInfo).
class Swapchain {
public:
    Swapchain() = default;
    Swapchain(Context& ctx, uint32_t width, uint32_t height, bool vsync = true);
    ~Swapchain();

    Swapchain(const Swapchain&)            = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&&) noexcept;
    Swapchain& operator=(Swapchain&&) noexcept;

    explicit operator bool() const noexcept { return swapchain_ != VK_NULL_HANDLE; }

    // Tear down and rebuild at a new size. The caller must vkDeviceWaitIdle first.
    void recreate(uint32_t width, uint32_t height);

    // Acquire the next image, signalling image_available. Returns the image index,
    // or nullopt if the swapchain is out of date (caller should recreate).
    std::optional<uint32_t> acquire(VkSemaphore image_available);

    // Present image_index, waiting on render_finished. Returns true if the swapchain
    // is out of date / suboptimal and should be recreated.
    bool present(uint32_t image_index, VkSemaphore render_finished);

    VkFormat   format()      const noexcept { return surface_format_.format; }
    VkExtent2D extent()      const noexcept { return extent_; }
    uint32_t   image_count() const noexcept { return static_cast<uint32_t>(images_.size()); }
    Image&     image(uint32_t i) noexcept   { return images_[i]; }

    VkSwapchainKHR native_handle() const noexcept { return swapchain_; }

private:
    void create(uint32_t width, uint32_t height);
    void destroy();

    VkDevice         device_        = VK_NULL_HANDLE;
    VkPhysicalDevice phys_          = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_       = VK_NULL_HANDLE;
    VkQueue          present_queue_ = VK_NULL_HANDLE;

    VkSwapchainKHR     swapchain_      = VK_NULL_HANDLE;
    VkSurfaceFormatKHR surface_format_ = {};
    VkPresentModeKHR   present_mode_   = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D         extent_         = {};
    bool               vsync_          = true;

    std::vector<Image> images_;  // non-owning wrappers (own only the views)
};

} // namespace vke
