#pragma once
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vulkan/vulkan.h>

namespace vke {

enum class ShaderStage {
    Vertex,
    Fragment,
    Compute,
    Task,
    Mesh,
};

struct ShaderCreateInfo {
    ShaderStage stage = ShaderStage::Vertex;

    // Exactly one of these must be set:
    std::filesystem::path     spirv_path = {};
    std::span<const uint32_t> spirv_code = {};

    const char* entry_point = "main";
    const char* debug_name  = nullptr;
};

class Shader;

namespace detail {
    Shader make_shader(VkDevice device, const ShaderCreateInfo& info);
} // namespace detail

class Shader {
public:
    Shader() = default;
    ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;
    Shader(Shader&&) noexcept;
    Shader& operator=(Shader&&) noexcept;

    explicit operator bool() const noexcept { return module_ != VK_NULL_HANDLE; }

    ShaderStage  stage()           const noexcept { return stage_; }
    const char*  entry_point()     const noexcept { return entry_point_.c_str(); }
    VkShaderModule native_handle() const noexcept { return module_; }

private:
    friend class Context;
    friend Shader detail::make_shader(VkDevice, const ShaderCreateInfo&);

    Shader(VkDevice device, VkShaderModule module,
           ShaderStage stage, std::string entry_point)
        : device_(device), module_(module),
          stage_(stage), entry_point_(std::move(entry_point)) {}

    VkDevice       device_      = VK_NULL_HANDLE;
    VkShaderModule module_      = VK_NULL_HANDLE;
    ShaderStage    stage_       = ShaderStage::Vertex;
    std::string    entry_point_;
};

} // namespace vke
