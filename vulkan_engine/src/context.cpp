#include <vke/context.hpp>
#include <vke/error.hpp>
#include <vk_mem_alloc.h>
#include <fmt/format.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace vke {

namespace detail {
    Shader make_shader(VkDevice device, const ShaderCreateInfo& info);
} // namespace detail

// ---- Internal helpers ----

namespace {

bool check_layer(const char* name) {
    uint32_t n = 0;
    vkEnumerateInstanceLayerProperties(&n, nullptr);
    std::vector<VkLayerProperties> layers(n);
    vkEnumerateInstanceLayerProperties(&n, layers.data());
    for (auto& l : layers)
        if (std::strcmp(l.layerName, name) == 0) return true;
    return false;
}

bool check_instance_ext(const char* name) {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data());
    for (auto& e : exts)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

bool check_device_ext(VkPhysicalDevice phys, const char* name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, exts.data());
    for (auto& e : exts)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

} // anonymous namespace

// ---- Context construction / destruction ----

Context::Context(const ContextCreateInfo& info) {
    create_instance(info);
    if (info.enable_validation_layers)
        setup_debug_messenger(info.debug_callback);
    pick_physical_device(info);
    create_device(info);
    create_allocator();
    create_command_pools();
    create_pipeline_cache(info.pipeline_cache_data);
}

Context::~Context() {
    for (auto f : fence_pool_)
        vkDestroyFence(device_, f, nullptr);

    if (pipeline_cache_ != VK_NULL_HANDLE)
        vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);

    // Queues may share a family (and thus a pool); destroy each unique pool once.
    VkCommandPool seen[3] = {};
    int n_seen = 0;
    for (auto* qf : {&graphics_, &compute_, &transfer_}) {
        if (qf->pool == VK_NULL_HANDLE) continue;
        bool dup = false;
        for (int k = 0; k < n_seen; ++k) dup |= (seen[k] == qf->pool);
        if (!dup) { seen[n_seen++] = qf->pool; vkDestroyCommandPool(device_, qf->pool, nullptr); }
    }

    if (allocator_ != nullptr)
        vmaDestroyAllocator(allocator_);

    if (device_ != VK_NULL_HANDLE)
        vkDestroyDevice(device_, nullptr);

    if (messenger_ != VK_NULL_HANDLE) {
        auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (fn) fn(instance_, messenger_, nullptr);
    }

    if (instance_ != VK_NULL_HANDLE)
        vkDestroyInstance(instance_, nullptr);
}

Context::Context(Context&& o) noexcept
    : instance_(o.instance_), messenger_(o.messenger_),
      physical_dev_(o.physical_dev_), device_(o.device_),
      allocator_(o.allocator_), pipeline_cache_(o.pipeline_cache_),
      graphics_(o.graphics_), compute_(o.compute_), transfer_(o.transfer_),
      device_info_(std::move(o.device_info_)),
      fence_pool_(std::move(o.fence_pool_))
{
    o.instance_         = VK_NULL_HANDLE;
    o.messenger_        = VK_NULL_HANDLE;
    o.device_           = VK_NULL_HANDLE;
    o.allocator_        = nullptr;
    o.pipeline_cache_   = VK_NULL_HANDLE;
    o.graphics_.pool    = VK_NULL_HANDLE;
    o.compute_.pool     = VK_NULL_HANDLE;
    o.transfer_.pool    = VK_NULL_HANDLE;
}

Context& Context::operator=(Context&& o) noexcept {
    if (this != &o) {
        this->~Context();
        new (this) Context(std::move(o));
    }
    return *this;
}

// ---- Instance creation ----

VKAPI_ATTR VkBool32 VKAPI_CALL Context::default_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
        std::cerr << "[vke] " << data->pMessage << "\n";
    return VK_FALSE;
}

void Context::create_instance(const ContextCreateInfo& info) {
    VkApplicationInfo app{
        .sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "sigvis",
        .apiVersion = VK_API_VERSION_1_3,
    };

    std::vector<const char*> layers;
    std::vector<const char*> exts{ VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };

    if (info.enable_validation_layers) {
        if (check_layer("VK_LAYER_KHRONOS_validation"))
            layers.push_back("VK_LAYER_KHRONOS_validation");
        if (check_instance_ext(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
            exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkValidationFeatureEnableEXT enabled_features[] = {
        VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT,
    };
    VkValidationFeaturesEXT validation_features{
        .sType                         = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
        .enabledValidationFeatureCount = 1,
        .pEnabledValidationFeatures    = enabled_features,
    };

    VkInstanceCreateInfo ci{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext                   = info.enable_validation_layers ? &validation_features : nullptr,
        .pApplicationInfo        = &app,
        .enabledLayerCount       = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames     = layers.empty() ? nullptr : layers.data(),
        .enabledExtensionCount   = static_cast<uint32_t>(exts.size()),
        .ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data(),
    };
    VKE_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
}

void Context::setup_debug_messenger(PFN_vkDebugUtilsMessengerCallbackEXT cb) {
    auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (!fn) return;

    VkDebugUtilsMessengerCreateInfoEXT ci{
        .sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = cb ? cb : default_debug_callback,
    };
    VKE_CHECK(fn(instance_, &ci, nullptr, &messenger_));
}

// ---- Physical device selection ----

void Context::pick_physical_device(const ContextCreateInfo& info) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    if (n == 0) throw VulkanError(VK_ERROR_INITIALIZATION_FAILED,
                                   "vkEnumeratePhysicalDevices", "no Vulkan devices found");
    std::vector<VkPhysicalDevice> devices(n);
    vkEnumeratePhysicalDevices(instance_, &n, devices.data());

    if (info.force_physical_device_index) {
        uint32_t idx = *info.force_physical_device_index;
        if (idx >= n) throw PreconditionError("force_physical_device_index out of range");
        physical_dev_ = devices[idx];
        return;
    }

    // Prefer discrete GPU, then integrated
    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    for (auto dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        bool has_mesh = check_device_ext(dev, VK_EXT_MESH_SHADER_EXTENSION_NAME);
        if (info.require_mesh_shaders && !has_mesh) continue;

        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_dev_ = dev;
            return;
        }
        if (fallback == VK_NULL_HANDLE)
            fallback = dev;
    }
    physical_dev_ = fallback;
    if (physical_dev_ == VK_NULL_HANDLE)
        throw VulkanError(VK_ERROR_INITIALIZATION_FAILED, "pick_physical_device",
                          "no suitable GPU found (mesh shaders required but unavailable)");
}

// ---- Logical device creation ----

void Context::create_device(const ContextCreateInfo& info) {
    // Find queue families
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_dev_, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_dev_, &n, families.data());

    auto find_queue = [&](VkQueueFlags required, VkQueueFlags preferred_exclusive) -> uint32_t {
        // Try dedicated queue first
        for (uint32_t i = 0; i < n; ++i)
            if ((families[i].queueFlags & required) == required &&
                !(families[i].queueFlags & preferred_exclusive))
                return i;
        // Fall back to any queue with required flags
        for (uint32_t i = 0; i < n; ++i)
            if ((families[i].queueFlags & required) == required)
                return i;
        return UINT32_MAX;
    };

    graphics_.index = find_queue(VK_QUEUE_GRAPHICS_BIT, 0);
    compute_.index  = find_queue(VK_QUEUE_COMPUTE_BIT,  VK_QUEUE_GRAPHICS_BIT);
    transfer_.index = find_queue(VK_QUEUE_TRANSFER_BIT, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);

    if (graphics_.index == UINT32_MAX)
        throw PreconditionError("no graphics queue family found");
    if (compute_.index  == UINT32_MAX) compute_.index  = graphics_.index;
    if (transfer_.index == UINT32_MAX) transfer_.index = graphics_.index;

    // Collect unique queue families
    std::vector<uint32_t> unique_families = { graphics_.index };
    if (compute_.index  != graphics_.index) unique_families.push_back(compute_.index);
    if (transfer_.index != graphics_.index && transfer_.index != compute_.index)
        unique_families.push_back(transfer_.index);

    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queue_cis;
    for (uint32_t idx : unique_families) {
        queue_cis.push_back(VkDeviceQueueCreateInfo{
            .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = idx,
            .queueCount       = 1,
            .pQueuePriorities = &priority,
        });
    }

    // Features chain — sync2 and dynamicRendering are Vulkan 1.3 core;
    // set them in Vk13Features only (not via separate extension structs).
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh_features{
        .sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
        .taskShader = VK_TRUE,
        .meshShader = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features vk12{
        .sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext               = &mesh_features,
        .bufferDeviceAddress = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features vk13{
        .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext            = &vk12,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    };

    std::vector<const char*> exts{ VK_EXT_MESH_SHADER_EXTENSION_NAME };

    VkDeviceCreateInfo ci{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &vk13,
        .queueCreateInfoCount    = static_cast<uint32_t>(queue_cis.size()),
        .pQueueCreateInfos       = queue_cis.data(),
        .enabledExtensionCount   = static_cast<uint32_t>(exts.size()),
        .ppEnabledExtensionNames = exts.data(),
    };
    VKE_CHECK(vkCreateDevice(physical_dev_, &ci, nullptr, &device_));

    vkGetDeviceQueue(device_, graphics_.index, 0, &graphics_.queue);
    vkGetDeviceQueue(device_, compute_.index,  0, &compute_.queue);
    vkGetDeviceQueue(device_, transfer_.index, 0, &transfer_.queue);

    // Fill device info
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_dev_, &props);
    device_info_.device_name      = props.deviceName;
    device_info_.device_type      = props.deviceType;
    device_info_.api_version      = props.apiVersion;
    device_info_.has_mesh_shaders = check_device_ext(physical_dev_, VK_EXT_MESH_SHADER_EXTENSION_NAME);
    device_info_.has_task_shaders = device_info_.has_mesh_shaders;
    device_info_.limits           = props.limits;

    if (device_info_.has_mesh_shaders) {
        VkPhysicalDeviceMeshShaderPropertiesEXT mesh_props{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT,
        };
        VkPhysicalDeviceProperties2 props2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &mesh_props,
        };
        vkGetPhysicalDeviceProperties2(physical_dev_, &props2);
        auto& ml = device_info_.mesh_limits;
        ml.max_work_group_count[0]  = mesh_props.maxMeshWorkGroupCount[0];
        ml.max_work_group_count[1]  = mesh_props.maxMeshWorkGroupCount[1];
        ml.max_work_group_count[2]  = mesh_props.maxMeshWorkGroupCount[2];
        ml.max_work_group_total     = mesh_props.maxMeshWorkGroupTotalCount;
        ml.max_output_vertices      = mesh_props.maxMeshOutputVertices;
        ml.max_output_primitives    = mesh_props.maxMeshOutputPrimitives;
        ml.preferred_work_group_size = mesh_props.maxPreferredMeshWorkGroupInvocations;
    }
}

// ---- VMA ----

void Context::create_allocator() {
    VmaVulkanFunctions vk_fns{ .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
                                .vkGetDeviceProcAddr   = vkGetDeviceProcAddr };
    VmaAllocatorCreateInfo ci{
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = physical_dev_,
        .device           = device_,
        .pVulkanFunctions = &vk_fns,
        .instance         = instance_,
        .vulkanApiVersion = VK_API_VERSION_1_3,
    };
    VKE_CHECK(vmaCreateAllocator(&ci, &allocator_));
}

// ---- Command pools ----

void Context::create_command_pools() {
    auto make_pool = [&](uint32_t family) -> VkCommandPool {
        VkCommandPoolCreateInfo ci{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = family,
        };
        VkCommandPool pool;
        VKE_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &pool));
        return pool;
    };
    graphics_.pool = make_pool(graphics_.index);
    compute_.pool  = (compute_.index  == graphics_.index) ? graphics_.pool : make_pool(compute_.index);
    transfer_.pool = (transfer_.index == graphics_.index) ? graphics_.pool :
                     (transfer_.index == compute_.index)  ? compute_.pool  :
                                                            make_pool(transfer_.index);
}

// ---- Pipeline cache ----

void Context::create_pipeline_cache(std::span<const uint8_t> data) {
    VkPipelineCacheCreateInfo ci{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .initialDataSize = data.size(),
        .pInitialData    = data.empty() ? nullptr : data.data(),
    };
    VKE_CHECK(vkCreatePipelineCache(device_, &ci, nullptr, &pipeline_cache_));
}

std::vector<uint8_t> Context::pipeline_cache_data() const {
    size_t size = 0;
    vkGetPipelineCacheData(device_, pipeline_cache_, &size, nullptr);
    std::vector<uint8_t> data(size);
    vkGetPipelineCacheData(device_, pipeline_cache_, &size, data.data());
    return data;
}

// ---- Resource factories ----

Buffer Context::create_buffer(const BufferCreateInfo& info) {
    VkBufferUsageFlags vk_usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (info.usage & BufferUsage::Vertex)      vk_usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (info.usage & BufferUsage::Index)       vk_usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (info.usage & BufferUsage::Uniform)     vk_usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (info.usage & BufferUsage::Storage)     vk_usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (info.usage & BufferUsage::Indirect)    vk_usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (info.usage & BufferUsage::TransferSrc) vk_usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (info.usage & BufferUsage::TransferDst) vk_usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    // Device-local buffers always need TransferDst for staging uploads
    if (info.domain == MemoryDomain::Device)   vk_usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VkBufferCreateInfo bci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = info.size,
        .usage = vk_usage,
    };

    VmaAllocationCreateFlags vma_flags = 0;
    VmaMemoryUsage vma_usage;
    switch (info.domain) {
    case MemoryDomain::Device:
        vma_usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        break;
    case MemoryDomain::Host:
        vma_usage  = VMA_MEMORY_USAGE_AUTO;
        vma_flags  = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
        break;
    case MemoryDomain::DeviceHost:
        vma_usage  = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vma_flags  = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
        break;
    }

    VmaAllocationCreateInfo aci{ .flags = vma_flags, .usage = vma_usage };
    VmaAllocationInfo alloc_info;
    VkBuffer buf;
    VmaAllocation alloc;
    VKE_CHECK(vmaCreateBuffer(allocator_, &bci, &aci, &buf, &alloc, &alloc_info));

    void* mapped = (info.domain != MemoryDomain::Device) ? alloc_info.pMappedData : nullptr;

    return Buffer(allocator_, device_, buf, alloc,
                  info.size, info.usage, info.domain, mapped,
                  transfer_.queue, transfer_.pool);
}

Image Context::create_image(const ImageCreateInfo& info) {
    VkImageUsageFlags vk_usage = 0;
    if (info.usage & ImageUsage::ColorAttachment) vk_usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (info.usage & ImageUsage::DepthAttachment) vk_usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (info.usage & ImageUsage::Sampled)         vk_usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (info.usage & ImageUsage::Storage)         vk_usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (info.usage & ImageUsage::TransferSrc)     vk_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (info.usage & ImageUsage::TransferDst)     vk_usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // Always allow transfer for download
    vk_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    VkImageType img_type = (info.depth > 1) ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;

    VkImageCreateInfo ici{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType     = img_type,
        .format        = info.format,
        .extent        = { info.width, info.height, info.depth },
        .mipLevels     = info.mip_levels,
        .arrayLayers   = info.array_layers,
        .samples       = info.samples,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = vk_usage,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VmaAllocationCreateInfo aci{ .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
    VkImage image;
    VmaAllocation alloc;
    VKE_CHECK(vmaCreateImage(allocator_, &ici, &aci, &image, &alloc, nullptr));

    // Determine aspect
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    if (info.usage & ImageUsage::DepthAttachment) {
        aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (info.format == VK_FORMAT_D16_UNORM_S8_UINT ||
            info.format == VK_FORMAT_D24_UNORM_S8_UINT ||
            info.format == VK_FORMAT_D32_SFLOAT_S8_UINT)
            aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }

    VkImageViewType view_type = (info.depth > 1) ? VK_IMAGE_VIEW_TYPE_3D :
                                (info.array_layers > 1) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY :
                                VK_IMAGE_VIEW_TYPE_2D;

    VkImageViewCreateInfo vci{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = image,
        .viewType = view_type,
        .format   = info.format,
        .subresourceRange = {
            .aspectMask     = aspect,
            .levelCount     = info.mip_levels,
            .layerCount     = info.array_layers,
        },
    };
    VkImageView view;
    VKE_CHECK(vkCreateImageView(device_, &vci, nullptr, &view));

    return Image(allocator_, device_, image, alloc, view,
                 info.width, info.height, info.depth,
                 info.format, info.usage, info.initial_layout,
                 graphics_.queue, graphics_.pool);
}

Shader Context::create_shader(const ShaderCreateInfo& info) {
    namespace detail = vke::detail;
    return detail::make_shader(device_, info);
}

DescriptorLayout Context::create_descriptor_layout(const DescriptorLayoutCreateInfo& info) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(info.bindings.size());

    auto to_vk_type = [](DescriptorType t) -> VkDescriptorType {
        switch (t) {
        case DescriptorType::UniformBuffer:        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case DescriptorType::StorageBuffer:        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        case DescriptorType::CombinedImageSampler: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        case DescriptorType::StorageImage:         return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case DescriptorType::UniformBufferDynamic: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        case DescriptorType::StorageBufferDynamic: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        }
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    };

    for (auto& b : info.bindings) {
        bindings.push_back(VkDescriptorSetLayoutBinding{
            .binding            = b.binding,
            .descriptorType     = to_vk_type(b.type),
            .descriptorCount    = b.count,
            .stageFlags         = b.stages,
        });
    }

    VkDescriptorSetLayoutCreateInfo lci{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings    = bindings.data(),
    };
    VkDescriptorSetLayout layout;
    VKE_CHECK(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &layout));

    // Build pool sizes
    std::vector<VkDescriptorPoolSize> pool_sizes;
    for (auto& b : info.bindings) {
        pool_sizes.push_back({ to_vk_type(b.type),
                               static_cast<uint32_t>(b.count * info.max_sets) });
    }

    VkDescriptorPoolCreateInfo pci{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets       = info.max_sets,
        .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
        .pPoolSizes    = pool_sizes.data(),
    };
    VkDescriptorPool pool;
    VKE_CHECK(vkCreateDescriptorPool(device_, &pci, nullptr, &pool));

    return DescriptorLayout(device_, layout, pool);
}

// ---- Pipeline creation helpers ----

VkPipelineLayout Context::build_pipeline_layout(
    std::span<const DescriptorLayout* const> layouts,
    std::span<const PushConstantRange> pc_ranges)
{
    std::vector<VkDescriptorSetLayout> set_layouts;
    set_layouts.reserve(layouts.size());
    for (auto* l : layouts) set_layouts.push_back(l->native_handle());

    std::vector<VkPushConstantRange> push_ranges;
    push_ranges.reserve(pc_ranges.size());
    for (auto& r : pc_ranges)
        push_ranges.push_back({ r.stages, r.offset, r.size });

    VkPipelineLayoutCreateInfo ci{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = static_cast<uint32_t>(set_layouts.size()),
        .pSetLayouts            = set_layouts.empty() ? nullptr : set_layouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(push_ranges.size()),
        .pPushConstantRanges    = push_ranges.empty() ? nullptr : push_ranges.data(),
    };
    VkPipelineLayout layout;
    VKE_CHECK(vkCreatePipelineLayout(device_, &ci, nullptr, &layout));
    return layout;
}

static VkShaderStageFlagBits to_vk_stage(ShaderStage s) {
    switch (s) {
    case ShaderStage::Vertex:   return VK_SHADER_STAGE_VERTEX_BIT;
    case ShaderStage::Fragment: return VK_SHADER_STAGE_FRAGMENT_BIT;
    case ShaderStage::Compute:  return VK_SHADER_STAGE_COMPUTE_BIT;
    case ShaderStage::Task:     return VK_SHADER_STAGE_TASK_BIT_EXT;
    case ShaderStage::Mesh:     return VK_SHADER_STAGE_MESH_BIT_EXT;
    }
    return VK_SHADER_STAGE_VERTEX_BIT;
}

Pipeline Context::create_pipeline(const GraphicsPipelineCreateInfo& info) {
    VKE_ASSERT(info.vertex_shader   != nullptr, "GraphicsPipeline: vertex_shader is null");
    VKE_ASSERT(info.fragment_shader != nullptr, "GraphicsPipeline: fragment_shader is null");

    VkPipelineLayout layout = build_pipeline_layout(info.descriptor_layouts,
                                                    info.push_constant_ranges);

    VkPipelineShaderStageCreateInfo stages[2] = {
        { .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage  = to_vk_stage(info.vertex_shader->stage()),
          .module = info.vertex_shader->native_handle(),
          .pName  = info.vertex_shader->entry_point() },
        { .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage  = to_vk_stage(info.fragment_shader->stage()),
          .module = info.fragment_shader->native_handle(),
          .pName  = info.fragment_shader->entry_point() },
    };

    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attribs;
    for (auto& b : info.vertex_bindings)
        bindings.push_back({ b.binding, b.stride, b.input_rate });
    for (auto& a : info.vertex_attributes)
        attribs.push_back({ a.location, a.binding, a.format, a.offset });

    VkPipelineVertexInputStateCreateInfo vertex_input{
        .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount   = static_cast<uint32_t>(bindings.size()),
        .pVertexBindingDescriptions      = bindings.empty() ? nullptr : bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attribs.size()),
        .pVertexAttributeDescriptions    = attribs.empty() ? nullptr : attribs.data(),
    };
    VkPipelineInputAssemblyStateCreateInfo input_assembly{
        .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = info.topology,
    };
    VkPipelineViewportStateCreateInfo viewport_state{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount  = 1,
    };
    VkPipelineRasterizationStateCreateInfo raster{
        .sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode    = info.cull_mode,
        .frontFace   = info.front_face,
        .lineWidth   = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo ms{
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = info.rasterization_samples,
    };
    VkPipelineDepthStencilStateCreateInfo depth{
        .sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable  = info.depth_test_enable  ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = info.depth_write_enable ? VK_TRUE : VK_FALSE,
        .depthCompareOp   = info.depth_compare_op,
    };

    // Default opaque blend if no attachments specified
    VkPipelineColorBlendAttachmentState default_blend{
        .blendEnable    = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    std::vector<VkPipelineColorBlendAttachmentState> blend_states;
    if (info.blend_attachments.empty()) {
        blend_states.resize(info.color_attachment_formats.size(), default_blend);
    } else {
        blend_states = { info.blend_attachments.begin(), info.blend_attachments.end() };
    }
    VkPipelineColorBlendStateCreateInfo color_blend{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<uint32_t>(blend_states.size()),
        .pAttachments    = blend_states.data(),
    };

    VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates    = dynamic_states,
    };

    // Dynamic rendering — no VkRenderPass
    VkPipelineRenderingCreateInfo rendering{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount    = static_cast<uint32_t>(info.color_attachment_formats.size()),
        .pColorAttachmentFormats = info.color_attachment_formats.data(),
        .depthAttachmentFormat   = info.depth_attachment_format,
        .stencilAttachmentFormat = info.stencil_attachment_format,
    };

    VkGraphicsPipelineCreateInfo ci{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = &rendering,
        .stageCount          = 2,
        .pStages             = stages,
        .pVertexInputState   = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState      = &viewport_state,
        .pRasterizationState = &raster,
        .pMultisampleState   = &ms,
        .pDepthStencilState  = &depth,
        .pColorBlendState    = &color_blend,
        .pDynamicState       = &dynamic,
        .layout              = layout,
        .renderPass          = VK_NULL_HANDLE,
    };
    VkPipeline pipeline;
    VKE_CHECK(vkCreateGraphicsPipelines(device_, pipeline_cache_, 1, &ci, nullptr, &pipeline));
    return Pipeline(device_, pipeline, layout, Pipeline::Type::Graphics);
}

Pipeline Context::create_pipeline(const MeshPipelineCreateInfo& info) {
    VKE_ASSERT(info.mesh_shader     != nullptr, "MeshPipeline: mesh_shader is null");
    VKE_ASSERT(info.fragment_shader != nullptr, "MeshPipeline: fragment_shader is null");

    VkPipelineLayout layout = build_pipeline_layout(info.descriptor_layouts,
                                                    info.push_constant_ranges);

    std::vector<VkPipelineShaderStageCreateInfo> stages;
    if (info.task_shader) {
        stages.push_back({ .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                           .stage  = VK_SHADER_STAGE_TASK_BIT_EXT,
                           .module = info.task_shader->native_handle(),
                           .pName  = info.task_shader->entry_point() });
    }
    stages.push_back({ .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       .stage  = VK_SHADER_STAGE_MESH_BIT_EXT,
                       .module = info.mesh_shader->native_handle(),
                       .pName  = info.mesh_shader->entry_point() });
    stages.push_back({ .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
                       .module = info.fragment_shader->native_handle(),
                       .pName  = info.fragment_shader->entry_point() });

    VkPipelineViewportStateCreateInfo viewport_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1,
    };
    VkPipelineRasterizationStateCreateInfo raster{
        .sType     = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .cullMode  = info.cull_mode,
        .frontFace = info.front_face,
        .lineWidth = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo ms{
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = info.rasterization_samples,
    };
    VkPipelineDepthStencilStateCreateInfo depth{
        .sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable  = info.depth_test_enable  ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = info.depth_write_enable ? VK_TRUE : VK_FALSE,
        .depthCompareOp   = info.depth_compare_op,
    };

    VkPipelineColorBlendAttachmentState default_blend{
        .blendEnable    = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    std::vector<VkPipelineColorBlendAttachmentState> blend_states;
    if (info.blend_attachments.empty()) {
        blend_states.resize(info.color_attachment_formats.size(), default_blend);
    } else {
        blend_states = { info.blend_attachments.begin(), info.blend_attachments.end() };
    }
    VkPipelineColorBlendStateCreateInfo color_blend{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<uint32_t>(blend_states.size()),
        .pAttachments    = blend_states.data(),
    };

    VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates    = dynamic_states,
    };

    VkPipelineRenderingCreateInfo rendering{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount    = static_cast<uint32_t>(info.color_attachment_formats.size()),
        .pColorAttachmentFormats = info.color_attachment_formats.data(),
        .depthAttachmentFormat   = info.depth_attachment_format,
        .stencilAttachmentFormat = info.stencil_attachment_format,
    };

    VkGraphicsPipelineCreateInfo ci{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = &rendering,
        .stageCount          = static_cast<uint32_t>(stages.size()),
        .pStages             = stages.data(),
        .pViewportState      = &viewport_state,
        .pRasterizationState = &raster,
        .pMultisampleState   = &ms,
        .pDepthStencilState  = &depth,
        .pColorBlendState    = &color_blend,
        .pDynamicState       = &dynamic,
        .layout              = layout,
        .renderPass          = VK_NULL_HANDLE,
    };
    VkPipeline pipeline;
    VKE_CHECK(vkCreateGraphicsPipelines(device_, pipeline_cache_, 1, &ci, nullptr, &pipeline));
    return Pipeline(device_, pipeline, layout, Pipeline::Type::Mesh);
}

Pipeline Context::create_pipeline(const ComputePipelineCreateInfo& info) {
    VKE_ASSERT(info.compute_shader != nullptr, "ComputePipeline: compute_shader is null");

    VkPipelineLayout layout = build_pipeline_layout(info.descriptor_layouts,
                                                    info.push_constant_ranges);
    VkComputePipelineCreateInfo ci{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = { .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = info.compute_shader->native_handle(),
                    .pName  = info.compute_shader->entry_point() },
        .layout = layout,
    };
    VkPipeline pipeline;
    VKE_CHECK(vkCreateComputePipelines(device_, pipeline_cache_, 1, &ci, nullptr, &pipeline));
    return Pipeline(device_, pipeline, layout, Pipeline::Type::Compute);
}

Sampler Context::create_sampler(const SamplerCreateInfo& info) {
    VkSamplerCreateInfo ci{
        .sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter        = info.mag_filter,
        .minFilter        = info.min_filter,
        .mipmapMode       = info.mipmap_mode,
        .addressModeU     = info.address_u,
        .addressModeV     = info.address_v,
        .addressModeW     = info.address_w,
        .anisotropyEnable = info.anisotropy_enable ? VK_TRUE : VK_FALSE,
        .maxAnisotropy    = info.max_anisotropy,
        .minLod           = info.min_lod,
        .maxLod           = info.max_lod,
    };
    VkSampler sampler;
    VKE_CHECK(vkCreateSampler(device_, &ci, nullptr, &sampler));

    auto impl = std::make_shared<Sampler::Impl>();
    impl->device  = device_;
    impl->sampler = sampler;
    return Sampler(std::move(impl));
}

CommandBuffer Context::create_command_buffer(QueueType queue) {
    QueueFamily* qf;
    switch (queue) {
    case QueueType::Compute:  qf = &compute_;  break;
    case QueueType::Transfer: qf = &transfer_; break;
    default:                  qf = &graphics_; break;
    }

    VkCommandBufferAllocateInfo ai{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = qf->pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    VKE_CHECK(vkAllocateCommandBuffers(device_, &ai, &cmd));
    return CommandBuffer(device_, qf->pool, cmd);
}

// ---- Submission ----

VkFence Context::acquire_fence() {
    if (!fence_pool_.empty()) {
        VkFence f = fence_pool_.back();
        fence_pool_.pop_back();
        VKE_CHECK(vkResetFences(device_, 1, &f));
        return f;
    }
    VkFenceCreateInfo fi{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence f;
    VKE_CHECK(vkCreateFence(device_, &fi, nullptr, &f));
    return f;
}

void Context::release_fence(VkFence fence) {
    fence_pool_.push_back(fence);
}

void Context::submit_and_wait(CommandBuffer cmd) {
    VkCommandBuffer raw = cmd.native_handle();
    VkCommandBufferSubmitInfo csi{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = raw,
    };
    VkSubmitInfo2 si{
        .sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos    = &csi,
    };
    VkFence fence = acquire_fence();
    VKE_CHECK(vkQueueSubmit2(graphics_.queue, 1, &si, fence));
    VKE_CHECK(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX));
    release_fence(fence);
    // cmd destructor will free the command buffer back to pool
}

SubmitHandle Context::submit(CommandBuffer cmd) {
    VkCommandBuffer raw = cmd.native_handle();
    VkCommandBufferSubmitInfo csi{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = raw,
    };
    VkSubmitInfo2 si{
        .sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos    = &csi,
    };
    VkFence fence = acquire_fence();
    VKE_CHECK(vkQueueSubmit2(graphics_.queue, 1, &si, fence));
    auto [dev, pool, buf] = cmd.detach(); // transfer ownership to SubmitHandle
    return SubmitHandle(device_, fence, pool, buf);
}

void Context::wait(SubmitHandle& handle) {
    handle.wait();
    // Return the fence to the pool; SubmitHandle exposes native_fence() for this
    if (VkFence f = handle.native_fence(); f != VK_NULL_HANDLE) {
        release_fence(f);
        handle.release_fence();
    }
}

bool Context::is_complete(const SubmitHandle& handle) const noexcept {
    return handle.is_complete();
}

} // namespace vke
