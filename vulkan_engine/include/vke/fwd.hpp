#pragma once

namespace vke {

class Context;
class Buffer;
class Image;
class Shader;
class Pipeline;
class DescriptorLayout;
class DescriptorSet;
class DescriptorSetWriter;
class Sampler;
class CommandBuffer;
class SubmitHandle;

enum class QueueType { Graphics, Compute, Transfer };
enum class BufferUsage : uint32_t;
enum class MemoryDomain;
enum class ImageUsage : uint32_t;
enum class ShaderStage;
enum class DescriptorType;

} // namespace vke
