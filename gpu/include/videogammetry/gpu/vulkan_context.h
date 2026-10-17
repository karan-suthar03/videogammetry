#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <volk.h>

namespace vg::gpu {

// One Vulkan device with a compute queue, shared by every GPU extractor.
// Vulkan is loaded at run time: create() fails cleanly (and the caller falls
// back to the CPU) when there is no driver or no suitable device.
class VulkanContext {
public:
    static std::shared_ptr<VulkanContext> create(std::string& error);
    ~VulkanContext();
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    VkDevice device() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physical_; }
    std::uint32_t queueFamily() const { return queueFamily_; }
    const std::string& deviceName() const { return deviceName_; }

    // Submits and waits on `fence` (the queue is shared, so submission locks).
    void submitAndWait(VkCommandBuffer cmd, VkFence fence);

    // Memory type index with all `required` flags (and `preferred` if any
    // such type exists); -1 if none.
    int memoryType(std::uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred = 0) const;

private:
    VulkanContext() = default;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queueFamily_ = 0;
    std::string deviceName_;
    VkPhysicalDeviceMemoryProperties memory_{};
    std::mutex queueMutex_;
};

// A buffer with its own memory allocation. Host-visible buffers stay mapped.
class Buffer {
public:
    enum class Kind { Device, Upload, Readback };

    Buffer() = default;
    Buffer(VulkanContext& ctx, VkDeviceSize size, Kind kind);
    ~Buffer();
    Buffer(Buffer&& o) noexcept { *this = std::move(o); }
    Buffer& operator=(Buffer&& o) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    VkBuffer handle() const { return buffer_; }
    VkDeviceSize size() const { return size_; }
    void* mapped() const { return mapped_; }
    // Makes GPU writes visible to mapped reads (no-op for coherent memory).
    void invalidate() const;
    // Makes mapped writes visible to the GPU (no-op for coherent memory).
    void flush() const;

private:
    void release();

    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    void* mapped_ = nullptr;
    bool coherent_ = true;
};

// Throws std::runtime_error with the call's name on failure.
void check(VkResult result, const char* what);

}  // namespace vg::gpu
