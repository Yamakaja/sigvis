find_package(Vulkan REQUIRED)
find_package(fmt REQUIRED)
find_package(glm REQUIRED)
find_package(VulkanMemoryAllocator REQUIRED)

# pybind11 (only needed when SIGVIS_PYTHON=ON)
if(SIGVIS_PYTHON)
    find_package(pybind11 REQUIRED)
endif()
