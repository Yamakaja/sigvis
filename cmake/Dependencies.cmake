find_package(Vulkan REQUIRED)
find_package(fmt REQUIRED)
find_package(glm REQUIRED)
find_package(VulkanMemoryAllocator REQUIRED)

# pybind11 (only needed when SIGVIS_PYTHON=ON)
if(SIGVIS_PYTHON)
    find_package(pybind11 REQUIRED)
endif()

# Viewer dependencies (only needed when SIGVIS_VIEWER=ON)
if(SIGVIS_VIEWER)
    find_package(glfw3 REQUIRED)
    # ImGui is vendored as source; point IMGUI_DIR at the checkout.
    set(IMGUI_DIR "$ENV{HOME}/rendering/imgui" CACHE PATH "Dear ImGui source directory")
    if(NOT EXISTS "${IMGUI_DIR}/imgui.cpp")
        message(FATAL_ERROR "IMGUI_DIR='${IMGUI_DIR}' does not contain imgui.cpp; set -DIMGUI_DIR=...")
    endif()
endif()
