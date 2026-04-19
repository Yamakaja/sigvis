#include <vke/pipeline.hpp>
#include <vke/descriptor.hpp>
#include <vke/shader.hpp>
#include <vke/error.hpp>
#include <vector>

namespace vke {

Pipeline::~Pipeline() {
    if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_   != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, layout_, nullptr);
}

Pipeline::Pipeline(Pipeline&& o) noexcept
    : device_(o.device_), pipeline_(o.pipeline_), layout_(o.layout_), type_(o.type_)
{
    o.device_   = VK_NULL_HANDLE;
    o.pipeline_ = VK_NULL_HANDLE;
    o.layout_   = VK_NULL_HANDLE;
}

Pipeline& Pipeline::operator=(Pipeline&& o) noexcept {
    if (this != &o) {
        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
        if (layout_   != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, layout_, nullptr);
        device_   = o.device_;
        pipeline_ = o.pipeline_;
        layout_   = o.layout_;
        type_     = o.type_;
        o.device_   = VK_NULL_HANDLE;
        o.pipeline_ = VK_NULL_HANDLE;
        o.layout_   = VK_NULL_HANDLE;
    }
    return *this;
}

} // namespace vke
