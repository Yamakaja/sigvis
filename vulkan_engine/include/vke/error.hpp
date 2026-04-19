#pragma once
#include <stdexcept>
#include <string>
#include <vulkan/vulkan.h>

namespace vke {

class VulkanError : public std::runtime_error {
public:
    explicit VulkanError(VkResult result, const char* fn, const char* context = "")
        : std::runtime_error(make_msg(result, fn, context)), result_(result) {}
    VkResult result() const noexcept { return result_; }

private:
    VkResult result_;
    static std::string make_msg(VkResult r, const char* fn, const char* ctx);
};

class PreconditionError : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

// Internal macro — throws VulkanError on non-VK_SUCCESS
#define VKE_CHECK(expr)                                          \
    do {                                                         \
        VkResult _r = (expr);                                    \
        if (_r != VK_SUCCESS) throw ::vke::VulkanError(_r, #expr); \
    } while (0)

#ifdef VKE_ENABLE_VALIDATION
#  define VKE_ASSERT(cond, msg)                                      \
     do {                                                             \
         if (!(cond)) throw ::vke::PreconditionError(msg);           \
     } while (0)
#else
#  define VKE_ASSERT(cond, msg) ((void)0)
#endif

} // namespace vke
