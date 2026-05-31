#include <vke/swapchain.hpp>
#include <vke/context.hpp>
#include <vke/error.hpp>
#include <algorithm>

namespace vke {

Swapchain::Swapchain(Context& ctx, uint32_t width, uint32_t height, bool vsync)
    : device_(ctx.native_device()),
      phys_(ctx.native_physical_device()),
      surface_(ctx.native_surface()),
      present_queue_(ctx.present_queue()),
      vsync_(vsync)
{
    if (surface_ == VK_NULL_HANDLE)
        throw PreconditionError("Swapchain requires a Context created with a surface_factory");
    create(width, height);
}

Swapchain::~Swapchain() {
    destroy();
}

Swapchain::Swapchain(Swapchain&& o) noexcept
    : device_(o.device_), phys_(o.phys_), surface_(o.surface_),
      present_queue_(o.present_queue_), swapchain_(o.swapchain_),
      surface_format_(o.surface_format_), present_mode_(o.present_mode_),
      extent_(o.extent_), vsync_(o.vsync_), images_(std::move(o.images_))
{
    o.swapchain_ = VK_NULL_HANDLE;
}

Swapchain& Swapchain::operator=(Swapchain&& o) noexcept {
    if (this != &o) {
        destroy();
        device_         = o.device_;
        phys_           = o.phys_;
        surface_        = o.surface_;
        present_queue_  = o.present_queue_;
        swapchain_      = o.swapchain_;
        surface_format_ = o.surface_format_;
        present_mode_   = o.present_mode_;
        extent_         = o.extent_;
        vsync_          = o.vsync_;
        images_         = std::move(o.images_);
        o.swapchain_    = VK_NULL_HANDLE;
    }
    return *this;
}

static VkSurfaceFormatKHR choose_format(VkPhysicalDevice phys, VkSurfaceKHR surface) {
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &n, formats.data());

    // Prefer BGRA8 UNORM (no implicit sRGB encode — our composite writes final colors).
    for (const auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    return formats.empty() ? VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM,
                                                VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}
                           : formats[0];
}

static VkPresentModeKHR choose_present_mode(VkPhysicalDevice phys, VkSurfaceKHR surface,
                                            bool vsync) {
    if (vsync) return VK_PRESENT_MODE_FIFO_KHR; // always supported
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &n, modes.data());
    for (auto m : modes)
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

void Swapchain::create(uint32_t width, uint32_t height) {
    VkSurfaceCapabilitiesKHR caps{};
    VKE_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_, surface_, &caps));

    surface_format_ = choose_format(phys_, surface_);
    present_mode_   = choose_present_mode(phys_, surface_, vsync_);

    if (caps.currentExtent.width != UINT32_MAX) {
        extent_ = caps.currentExtent;
    } else {
        extent_.width  = std::clamp(width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
        extent_.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount)
        image_count = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci{
        .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface          = surface_,
        .minImageCount    = image_count,
        .imageFormat      = surface_format_.format,
        .imageColorSpace  = surface_format_.colorSpace,
        .imageExtent      = extent_,
        .imageArrayLayers = 1,
        .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform     = caps.currentTransform,
        .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode      = present_mode_,
        .clipped          = VK_TRUE,
        .oldSwapchain     = VK_NULL_HANDLE,
    };
    VKE_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));

    uint32_t n = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
    std::vector<VkImage> raw(n);
    vkGetSwapchainImagesKHR(device_, swapchain_, &n, raw.data());

    images_.clear();
    images_.reserve(n);
    for (VkImage img : raw) {
        VkImageViewCreateInfo vci{
            .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image    = img,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format   = surface_format_.format,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1,
            },
        };
        VkImageView view;
        VKE_CHECK(vkCreateImageView(device_, &vci, nullptr, &view));
        images_.push_back(Image::from_external(device_, img, view,
                                               extent_.width, extent_.height,
                                               surface_format_.format,
                                               ImageUsage::ColorAttachment));
    }
}

void Swapchain::destroy() {
    images_.clear(); // destroys the views (not the images)
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void Swapchain::recreate(uint32_t width, uint32_t height) {
    destroy();
    create(width, height);
}

std::optional<uint32_t> Swapchain::acquire(VkSemaphore image_available) {
    uint32_t index = 0;
    VkResult r = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                       image_available, VK_NULL_HANDLE, &index);
    if (r == VK_ERROR_OUT_OF_DATE_KHR)
        return std::nullopt;
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR)
        throw VulkanError(r, "vkAcquireNextImageKHR");

    // The acquired image's previous contents are undefined as far as we're concerned.
    images_[index].set_assumed_layout(VK_IMAGE_LAYOUT_UNDEFINED);
    return index;
}

bool Swapchain::present(uint32_t image_index, VkSemaphore render_finished) {
    VkPresentInfoKHR pi{
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = &render_finished,
        .swapchainCount     = 1,
        .pSwapchains        = &swapchain_,
        .pImageIndices      = &image_index,
    };
    VkResult r = vkQueuePresentKHR(present_queue_, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
        return true;
    if (r != VK_SUCCESS)
        throw VulkanError(r, "vkQueuePresentKHR");
    return false;
}

} // namespace vke
