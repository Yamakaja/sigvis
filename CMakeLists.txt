cmake_minimum_required(VERSION 3.25)
project(sigvis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SIGVIS_PYTHON     "Build pyrendering Python extension" OFF)
option(SIGVIS_VALIDATION "Enable C++-level vke validation checks" OFF)
option(SIGVIS_VIEWER     "Build the realtime waveform viewer app" OFF)

include(cmake/Dependencies.cmake)
include(cmake/Shaders.cmake)

add_subdirectory(vulkan_engine)
add_subdirectory(plot_renderer)

if(SIGVIS_PYTHON)
    add_subdirectory(pyrendering)
endif()

if(SIGVIS_VIEWER)
    add_subdirectory(viewer)
endif()
