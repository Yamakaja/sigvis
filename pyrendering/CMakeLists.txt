pybind11_add_module(pyrendering
    src/bindings.cpp
)

target_link_libraries(pyrendering PRIVATE
    plot::plot_renderer
)

target_compile_options(pyrendering PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
)
