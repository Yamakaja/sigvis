#include <vke/shader.hpp>
#include <vke/error.hpp>
#include <fstream>
#include <vector>

namespace vke {

Shader::~Shader() {
    if (module_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, module_, nullptr);
}

Shader::Shader(Shader&& o) noexcept
    : device_(o.device_), module_(o.module_),
      stage_(o.stage_), entry_point_(std::move(o.entry_point_))
{
    o.device_ = VK_NULL_HANDLE;
    o.module_ = VK_NULL_HANDLE;
}

Shader& Shader::operator=(Shader&& o) noexcept {
    if (this != &o) {
        if (module_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, module_, nullptr);
        device_      = o.device_;
        module_      = o.module_;
        stage_       = o.stage_;
        entry_point_ = std::move(o.entry_point_);
        o.device_ = VK_NULL_HANDLE;
        o.module_ = VK_NULL_HANDLE;
    }
    return *this;
}

// Context::create_shader implementation (called from context.cpp)
namespace detail {

static std::vector<uint32_t> load_spirv_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw PreconditionError("Cannot open SPIR-V file: " + path.string());
    auto size = static_cast<size_t>(file.tellg());
    if (size % 4 != 0)
        throw PreconditionError("SPIR-V file size not a multiple of 4: " + path.string());
    file.seekg(0);
    std::vector<uint32_t> code(size / 4);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));
    return code;
}

VkShaderModule create_shader_module(VkDevice device,
                                     std::span<const uint32_t> code) {
    VkShaderModuleCreateInfo ci{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size_bytes(),
        .pCode    = code.data(),
    };
    VkShaderModule mod;
    VKE_CHECK(vkCreateShaderModule(device, &ci, nullptr, &mod));
    return mod;
}

Shader make_shader(VkDevice device, const ShaderCreateInfo& info) {
    VkShaderModule mod;
    if (!info.spirv_path.empty()) {
        auto code = load_spirv_file(info.spirv_path);
        mod = create_shader_module(device, code);
    } else {
        if (info.spirv_code.empty())
            throw PreconditionError("ShaderCreateInfo: neither spirv_path nor spirv_code set");
        mod = create_shader_module(device, info.spirv_code);
    }
    return Shader(device, mod, info.stage,
                  info.entry_point ? info.entry_point : "main");
}

} // namespace detail
} // namespace vke
