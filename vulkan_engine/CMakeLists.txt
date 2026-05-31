add_library(vulkan_engine STATIC
    src/vma.cpp            # VMA implementation (VMA_IMPLEMENTATION defined only here)
    src/context.cpp
    src/buffer.cpp
    src/image.cpp
    src/shader.cpp
    src/pipeline.cpp
    src/descriptor.cpp
    src/command_buffer.cpp
    src/sampler.cpp
    src/submit.cpp
    src/swapchain.cpp
    src/error.cpp
)
add_library(vke::vulkan_engine ALIAS vulkan_engine)

target_include_directories(vulkan_engine
    PUBLIC  include
    PRIVATE src
)

target_link_libraries(vulkan_engine
    PUBLIC
        Vulkan::Vulkan
        GPUOpen::VulkanMemoryAllocator
        glm::glm
    PRIVATE
        fmt::fmt
)

target_compile_definitions(vulkan_engine
    PUBLIC
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_FORCE_RADIANS
        $<$<BOOL:${SIGVIS_VALIDATION}>:VKE_ENABLE_VALIDATION>
    PRIVATE
        VMA_STATIC_VULKAN_FUNCTIONS=0
        VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)

target_compile_options(vulkan_engine PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers>
)

set_target_properties(vulkan_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
