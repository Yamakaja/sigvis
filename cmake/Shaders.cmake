# compile_shaders(TARGET <target> SOURCES <shader files...>)
# Compiles each GLSL shader to SPIR-V (.spv) in the build directory,
# then generates a C++ header embedding the SPIR-V words via xxd-style array.
# Adds a custom target that <TARGET> depends on.
function(compile_shaders)
    cmake_parse_arguments(ARG "" "TARGET" "SOURCES" ${ARGN})

    find_program(GLSLC glslc REQUIRED)

    set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders")
    file(MAKE_DIRECTORY "${GENERATED_DIR}")

    set(GENERATED_HEADERS "")

    foreach(SHADER_SRC ${ARG_SOURCES})
        get_filename_component(SHADER_NAME "${SHADER_SRC}" NAME)
        string(REPLACE "." "_" HEADER_STEM "${SHADER_NAME}")
        set(SPV_FILE  "${GENERATED_DIR}/${SHADER_NAME}.spv")
        set(HDR_FILE  "${GENERATED_DIR}/${HEADER_STEM}_spv.hpp")

        # Determine extra flags per stage
        set(EXTRA_FLAGS "")
        if(SHADER_SRC MATCHES "\\.mesh$" OR SHADER_SRC MATCHES "\\.task$")
            list(APPEND EXTRA_FLAGS "--target-spv=spv1.4")
        endif()

        add_custom_command(
            OUTPUT "${SPV_FILE}"
            COMMAND "${GLSLC}"
                --target-env=vulkan1.3
                ${EXTRA_FLAGS}
                -o "${SPV_FILE}"
                "${SHADER_SRC}"
            DEPENDS "${SHADER_SRC}"
            COMMENT "Compiling shader: ${SHADER_NAME}"
            VERBATIM
        )

        # Generate embeddable C++ header from .spv
        add_custom_command(
            OUTPUT "${HDR_FILE}"
            COMMAND ${CMAKE_COMMAND}
                -DSPV_FILE=${SPV_FILE}
                -DHDR_FILE=${HDR_FILE}
                -DVAR_NAME=${HEADER_STEM}
                -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EmbedSpirv.cmake"
            DEPENDS "${SPV_FILE}"
            COMMENT "Embedding SPIR-V: ${SHADER_NAME}"
            VERBATIM
        )

        list(APPEND GENERATED_HEADERS "${HDR_FILE}")
    endforeach()

    set(SHADER_TARGET "${ARG_TARGET}_shaders")
    add_custom_target("${SHADER_TARGET}" DEPENDS ${GENERATED_HEADERS})
    add_dependencies("${ARG_TARGET}" "${SHADER_TARGET}")

    # Expose generated dir as include path
    target_include_directories("${ARG_TARGET}" PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
endfunction()
