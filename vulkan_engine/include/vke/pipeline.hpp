#pragma once
#include <cstdint>
#include <span>
#include <vulkan/vulkan.h>

namespace vke {

class Shader;
class DescriptorLayout;

struct PushConstantRange {
    VkShaderStageFlags stages = 0;
    uint32_t           offset = 0;
    uint32_t           size   = 0;
};

struct VertexAttribute {
    uint32_t location = 0;
    uint32_t binding  = 0;
    VkFormat format   = VK_FORMAT_R32G32_SFLOAT;
    uint32_t offset   = 0;
};

struct VertexBinding {
    uint32_t          binding    = 0;
    uint32_t          stride     = 0;
    VkVertexInputRate input_rate = VK_VERTEX_INPUT_RATE_VERTEX;
};

// ---- Graphics pipeline (vertex + fragment) ----
struct GraphicsPipelineCreateInfo {
    Shader* vertex_shader   = nullptr;
    Shader* fragment_shader = nullptr;

    std::span<const DescriptorLayout* const> descriptor_layouts      = {};
    std::span<const PushConstantRange>       push_constant_ranges    = {};
    std::span<const VertexBinding>           vertex_bindings         = {};
    std::span<const VertexAttribute>         vertex_attributes       = {};

    std::span<const VkFormat> color_attachment_formats               = {};
    VkFormat depth_attachment_format   = VK_FORMAT_UNDEFINED;
    VkFormat stencil_attachment_format = VK_FORMAT_UNDEFINED;

    VkPrimitiveTopology   topology    = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCullModeFlags       cull_mode   = VK_CULL_MODE_BACK_BIT;
    VkFrontFace           front_face  = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool                  depth_test_enable  = false;
    bool                  depth_write_enable = false;
    VkCompareOp           depth_compare_op   = VK_COMPARE_OP_LESS;
    VkSampleCountFlagBits rasterization_samples = VK_SAMPLE_COUNT_1_BIT;

    std::span<const VkPipelineColorBlendAttachmentState> blend_attachments = {};
    bool dynamic_blend_constants = false; // adds VK_DYNAMIC_STATE_BLEND_CONSTANTS
    const char* debug_name = nullptr;
    // Viewport and scissor are always dynamic state
};

// ---- Mesh pipeline (task + mesh + fragment) ----
struct MeshPipelineCreateInfo {
    Shader* task_shader     = nullptr; // optional
    Shader* mesh_shader     = nullptr;
    Shader* fragment_shader = nullptr;

    std::span<const DescriptorLayout* const> descriptor_layouts   = {};
    std::span<const PushConstantRange>       push_constant_ranges = {};

    std::span<const VkFormat> color_attachment_formats            = {};
    VkFormat depth_attachment_format   = VK_FORMAT_UNDEFINED;
    VkFormat stencil_attachment_format = VK_FORMAT_UNDEFINED;

    VkCullModeFlags       cull_mode   = VK_CULL_MODE_BACK_BIT;
    VkFrontFace           front_face  = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool                  depth_test_enable  = false;
    bool                  depth_write_enable = false;
    VkCompareOp           depth_compare_op   = VK_COMPARE_OP_LESS;
    VkSampleCountFlagBits rasterization_samples = VK_SAMPLE_COUNT_1_BIT;

    std::span<const VkPipelineColorBlendAttachmentState> blend_attachments = {};
    bool dynamic_blend_constants = false; // adds VK_DYNAMIC_STATE_BLEND_CONSTANTS
    const char* debug_name = nullptr;
};

// ---- Compute pipeline ----
struct ComputePipelineCreateInfo {
    Shader* compute_shader = nullptr;

    std::span<const DescriptorLayout* const> descriptor_layouts   = {};
    std::span<const PushConstantRange>       push_constant_ranges = {};

    const char* debug_name = nullptr;
};

// ---- Pipeline ----
class Pipeline {
public:
    enum class Type { Graphics, Mesh, Compute };

    Pipeline() = default;
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;

    explicit operator bool() const noexcept { return pipeline_ != VK_NULL_HANDLE; }

    Type             type()                 const noexcept { return type_; }
    VkPipeline       native_handle()        const noexcept { return pipeline_; }
    VkPipelineLayout native_layout_handle() const noexcept { return layout_; }

private:
    friend class Context;

    Pipeline(VkDevice device, VkPipeline pipeline, VkPipelineLayout layout, Type type)
        : device_(device), pipeline_(pipeline), layout_(layout), type_(type) {}

    VkDevice         device_   = VK_NULL_HANDLE;
    VkPipeline       pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_   = VK_NULL_HANDLE;
    Type             type_     = Type::Graphics;
};

} // namespace vke
