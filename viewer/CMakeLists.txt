# ---- Dear ImGui (vendored source target) ----
add_library(imgui STATIC
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
)
target_include_directories(imgui PUBLIC
    "${IMGUI_DIR}"
    "${IMGUI_DIR}/backends"
)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)
set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Viewer application ----
add_executable(viewer
    src/main.cpp
    src/data_source.cpp
    src/audio_source.cpp
)

# PulseAudio simple API for live audio capture (works against PipeWire's pulse server).
find_package(PkgConfig REQUIRED)
pkg_check_modules(PULSE_SIMPLE REQUIRED IMPORTED_TARGET libpulse-simple)

target_link_libraries(viewer PRIVATE
    plot::plot_renderer
    vke::vulkan_engine
    imgui
    glfw
    glm::glm
    fmt::fmt
    PkgConfig::PULSE_SIMPLE
)

target_compile_options(viewer PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
