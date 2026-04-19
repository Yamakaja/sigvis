#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>

#include <plot/eye_diagram_renderer.hpp>
#include <vke/vke.hpp>

#include <unordered_set>

namespace py = pybind11;

// Module-level shared context (created lazily on first use)
static std::unique_ptr<vke::Context> g_context;
static bool g_validation_layers = false;

static vke::Context& get_context() {
    if (!g_context) {
        g_context = std::make_unique<vke::Context>(vke::ContextCreateInfo{
            .enable_validation_layers = g_validation_layers,
            .require_mesh_shaders     = true,
        });
    }
    return *g_context;
}

class PyEyeDiagramRenderer;
static std::unordered_set<PyEyeDiagramRenderer*> g_live_renderers;

class PyEyeDiagramRenderer {
public:
    PyEyeDiagramRenderer(uint32_t width, uint32_t height)
        : renderer_(get_context(), width, height)
    {
        g_live_renderers.insert(this);
    }

    ~PyEyeDiagramRenderer() {
        g_live_renderers.erase(this);
    }

    // Called by shutdown() before the context is destroyed.
    void release_gpu_resources() {
        renderer_ = plot::EyeDiagramRenderer();
    }

    void set_samples(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        auto buf = arr.request();
        if (buf.ndim != 3 || buf.shape[2] != 2)
            throw py::value_error("samples must be shape (n_traces, trace_length, 2): [x, y]");

        auto* ptr = static_cast<float*>(buf.ptr);
        size_t n_traces      = static_cast<size_t>(buf.shape[0]);
        size_t trace_length  = static_cast<size_t>(buf.shape[1]);
        size_t n             = n_traces * trace_length;

        std::vector<plot::Sample> points(n);
        for (size_t i = 0; i < n; ++i)
            points[i].position = { ptr[i*2+0], ptr[i*2+1] };

        renderer_.set_samples(points, static_cast<uint32_t>(trace_length));
    }

    static plot::RenderParams make_params(
        std::pair<float,float> x_range, std::pair<float,float> y_range,
        float line_width, uint32_t width, uint32_t height, float max_intensity = 1.0f)
    {
        return plot::RenderParams{
            .center        = { (x_range.first + x_range.second) * 0.5f,
                               (y_range.first + y_range.second) * 0.5f },
            .zoom          = { 2.0f / (x_range.second - x_range.first),
                               2.0f / (y_range.second - y_range.first) },
            .line_width    = line_width,
            .width         = width,
            .height        = height,
            .max_intensity = max_intensity,
        };
    }

    py::array_t<uint8_t> render(
        std::pair<float,float> x_range,
        std::pair<float,float> y_range,
        float line_width,
        float max_intensity)
    {
        auto params = make_params(x_range, y_range, line_width,
                                  renderer_.width(), renderer_.height(), max_intensity);

        const vke::Image& result = renderer_.render(params);
        auto pixels = result.download_rgba8();

        return py::array_t<uint8_t>(
            { static_cast<py::ssize_t>(renderer_.height()),
              static_cast<py::ssize_t>(renderer_.width()),
              static_cast<py::ssize_t>(4) },
            pixels.data());
    }

    py::array_t<float> render_histogram(
        std::pair<float,float> x_range,
        std::pair<float,float> y_range,
        float line_width)
    {
        auto params = make_params(x_range, y_range, line_width,
                                  renderer_.width(), renderer_.height());

        const vke::Image& hist = renderer_.render_histogram(params);
        auto raw = hist.download();

        py::array_t<float> out(
            { static_cast<py::ssize_t>(renderer_.height()),
              static_cast<py::ssize_t>(renderer_.width()) });
        std::memcpy(out.mutable_data(), raw.data(), raw.size());
        return out;
    }

    uint32_t width()  const { return renderer_.width(); }
    uint32_t height() const { return renderer_.height(); }

private:
    plot::EyeDiagramRenderer renderer_;
};

PYBIND11_MODULE(pyrendering, m) {
    m.doc() = "GPU-accelerated eye diagram / histogram renderer";

    py::class_<PyEyeDiagramRenderer>(m, "EyeDiagramRenderer")
        .def(py::init<uint32_t, uint32_t>(),
             py::arg("width") = 1920, py::arg("height") = 1080,
             "Create an off-screen eye diagram renderer.")

        .def("set_samples", &PyEyeDiagramRenderer::set_samples,
             py::arg("samples"),
             "Upload sample data. samples: float32 numpy array of shape (n_traces, trace_length, 2) [x, y].")

        .def("render", &PyEyeDiagramRenderer::render,
             py::arg("x_range")      = std::make_pair(-1.0f, 1.0f),
             py::arg("y_range")      = std::make_pair(-1.0f, 1.0f),
             py::arg("line_width")   = 1.0f,
             py::arg("max_intensity") = 1.0f,
             "Render and return a (H, W, 4) uint8 RGBA numpy array.")

        .def("render_histogram", &PyEyeDiagramRenderer::render_histogram,
             py::arg("x_range")    = std::make_pair(-1.0f, 1.0f),
             py::arg("y_range")    = std::make_pair(-1.0f, 1.0f),
             py::arg("line_width") = 1.0f,
             "Render histogram only and return a (H, W) float32 numpy array.")

        .def_property_readonly("width",  &PyEyeDiagramRenderer::width)
        .def_property_readonly("height", &PyEyeDiagramRenderer::height);

    m.def("init", [](bool validation_layers) {
              if (g_context)
                  throw std::runtime_error("Context already initialised; call shutdown() first.");
              g_validation_layers = validation_layers;
          },
          py::arg("validation_layers") = false,
          "Configure context options before first use. Must be called before creating any renderer.");

    m.def("shutdown", []() {
              // Release GPU resources on all live renderers before destroying the context.
              for (auto* r : g_live_renderers)
                  r->release_gpu_resources();
              g_context.reset();
              g_validation_layers = false;
          },
          "Release the shared Vulkan context. Safe to call while renderer objects still exist.");
}
