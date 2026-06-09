add_library(plot_renderer STATIC
    src/eye_diagram_renderer.cpp
    src/histogram.cpp
    src/waveform_scope.cpp
    src/roll_scope.cpp
    src/sample_ring.cpp
    src/stream_scope.cpp
)
add_library(plot::plot_renderer ALIAS plot_renderer)

target_include_directories(plot_renderer
    PUBLIC  include
    PRIVATE src
            ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_link_libraries(plot_renderer
    PUBLIC  vke::vulkan_engine
    PRIVATE fmt::fmt
)

target_compile_options(plot_renderer PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers>
)

set_target_properties(plot_renderer PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Compile shaders and embed as C++ headers
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
compile_shaders(TARGET plot_renderer SOURCES
    "${SHADER_DIR}/lines_renderer.mesh"
    "${SHADER_DIR}/lines_renderer.frag"
    "${SHADER_DIR}/waveform_renderer.mesh"
    "${SHADER_DIR}/points_renderer.mesh"
    "${SHADER_DIR}/points_renderer.frag"
    "${SHADER_DIR}/histogram_sample.vert"
    "${SHADER_DIR}/histogram_sample.frag"
    "${SHADER_DIR}/decay.frag"
    "${SHADER_DIR}/roll_reduce.comp"
    "${SHADER_DIR}/roll_composite.frag"
    "${SHADER_DIR}/stream_reduce.comp"
    "${SHADER_DIR}/stream_trigger.mesh"
)
