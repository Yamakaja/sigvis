#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "buffer.hpp"
#include "command_buffer.hpp"
#include "fwd.hpp"
#include "descriptor.hpp"
#include "error.hpp"
#include "image.hpp"
#include "pipeline.hpp"
#include "sampler.hpp"
#include "shader.hpp"
#include "submit.hpp"

namespace vke {

struct MeshShaderLimits {
    uint32_t max_work_group_count[3];  // maxMeshWorkGroupCount[0..2]
    uint32_t max_work_group_total;     // maxMeshWorkGroupTotalCount
    uint32_t max_output_vertices;
    uint32_t max_output_primitives;
    uint32_t preferred_work_group_size;
};

struct PhysicalDeviceInfo {
    std::string              device_name;
    VkPhysicalDeviceType     device_type;
    uint32_t                 api_version;
    bool                     has_mesh_shaders;
    bool                     has_task_shaders;
    VkPhysicalDeviceLimits   limits;
    MeshShaderLimits         mesh_limits;
};

struct ContextCreateInfo {
    bool enable_validation_layers = false;
    PFN_vkDebugUtilsMessengerCallbackEXT debug_callback = nullptr;

    std::optional<uint32_t> force_physical_device_index = {};

    bool require_mesh_shaders      = true;
    bool require_dynamic_rendering = true;
    bool require_synchronization2  = true;

    std::span<const uint8_t> pipeline_cache_data = {};

    // When non-null: enables VK_KHR_swapchain, selects present queue.
    // Context must be initialized in two phases (see create_instance / init_device)
    // when a surface is needed; leave null for headless use.
    VkSurfaceKHR surface = VK_NULL_HANDLE;
};

class Context {
public:
    explicit Context(const ContextCreateInfo& info = {});
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;

    // ---- Resource factories ----

    Buffer           create_buffer(const BufferCreateInfo& info);
    Image            create_image(const ImageCreateInfo& info);
    Shader           create_shader(const ShaderCreateInfo& info);
    DescriptorLayout create_descriptor_layout(const DescriptorLayoutCreateInfo& info);
    Pipeline         create_pipeline(const GraphicsPipelineCreateInfo& info);
    Pipeline         create_pipeline(const MeshPipelineCreateInfo& info);
    Pipeline         create_pipeline(const ComputePipelineCreateInfo& info);
    Sampler          create_sampler(const SamplerCreateInfo& info);
    CommandBuffer    create_command_buffer(QueueType queue = QueueType::Graphics);

    // ---- Submission ----

    void         submit_and_wait(CommandBuffer cmd);
    SubmitHandle submit(CommandBuffer cmd);
    void         wait(SubmitHandle& handle);
    bool         is_complete(const SubmitHandle& handle) const noexcept;

    // ---- Pipeline cache ----

    std::vector<uint8_t> pipeline_cache_data() const;

    // ---- Introspection ----

    const PhysicalDeviceInfo& physical_device_info() const noexcept { return device_info_; }

    // ---- Escape hatches ----

    VkDevice     native_device()    const noexcept { return device_; }
    VkInstance   native_instance()  const noexcept { return instance_; }
    VmaAllocator native_allocator() const noexcept { return allocator_; }

private:
    // Vulkan init helpers
    void create_instance(const ContextCreateInfo& info);
    void setup_debug_messenger(PFN_vkDebugUtilsMessengerCallbackEXT cb);
    void pick_physical_device(const ContextCreateInfo& info);
    void create_device(const ContextCreateInfo& info);
    void create_allocator();
    void create_command_pools();
    void create_pipeline_cache(std::span<const uint8_t> data);

    VkPipelineLayout build_pipeline_layout(
        std::span<const DescriptorLayout* const> layouts,
        std::span<const PushConstantRange> pc_ranges);

    VkFence acquire_fence();
    void    release_fence(VkFence fence);

    static VKAPI_ATTR VkBool32 VKAPI_CALL default_debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void* user_data);

    VkInstance               instance_     = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_    = VK_NULL_HANDLE;
    VkPhysicalDevice         physical_dev_ = VK_NULL_HANDLE;
    VkDevice                 device_       = VK_NULL_HANDLE;
    VmaAllocator             allocator_    = nullptr;
    VkPipelineCache          pipeline_cache_ = VK_NULL_HANDLE;

    struct QueueFamily {
        uint32_t index = UINT32_MAX;
        VkQueue  queue = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
    };
    QueueFamily graphics_;
    QueueFamily compute_;
    QueueFamily transfer_;

    PhysicalDeviceInfo device_info_;

    // Recycled fences for submit_and_wait / submit
    std::vector<VkFence> fence_pool_;
};

} // namespace vke
