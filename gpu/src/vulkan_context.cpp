#include "videogammetry/gpu/vulkan_context.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace vg::gpu {

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed (VkResult " + std::to_string(result) + ")");
}

std::shared_ptr<VulkanContext> VulkanContext::create(std::string& error) {
    // volk loads the Vulkan loader once per process.
    static const VkResult loaded = volkInitialize();
    if (loaded != VK_SUCCESS) {
        error = "no Vulkan driver (vulkan-1 not found)";
        return nullptr;
    }
    std::shared_ptr<VulkanContext> ctx(new VulkanContext());
    try {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "videogammetry";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        check(vkCreateInstance(&ici, nullptr, &ctx->instance_), "vkCreateInstance");
        volkLoadInstanceOnly(ctx->instance_);

        std::uint32_t count = 0;
        vkEnumeratePhysicalDevices(ctx->instance_, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(ctx->instance_, &count, devices.data());

        // Prefer a discrete GPU, then integrated, then anything with compute.
        int bestRank = -1;
        for (VkPhysicalDevice d : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(d, &props);
            if (props.apiVersion < VK_API_VERSION_1_1) continue;
            std::uint32_t nq = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
            std::vector<VkQueueFamilyProperties> families(nq);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, families.data());
            for (std::uint32_t q = 0; q < nq; ++q) {
                if (!(families[q].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
                const int rank = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 3
                                 : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                                 : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU            ? 0
                                                                                              : 1;
                if (rank > bestRank) {
                    bestRank = rank;
                    ctx->physical_ = d;
                    ctx->queueFamily_ = q;
                    ctx->deviceName_ = props.deviceName;
                }
                break;
            }
        }
        if (!ctx->physical_) {
            error = "no Vulkan 1.1 device with a compute queue";
            return nullptr;
        }
        vkGetPhysicalDeviceMemoryProperties(ctx->physical_, &ctx->memory_);

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = ctx->queueFamily_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        check(vkCreateDevice(ctx->physical_, &dci, nullptr, &ctx->device_), "vkCreateDevice");
        volkLoadDevice(ctx->device_);
        vkGetDeviceQueue(ctx->device_, ctx->queueFamily_, 0, &ctx->queue_);
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
    return ctx;
}

VulkanContext::~VulkanContext() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

void VulkanContext::submitAndWait(VkCommandBuffer cmd, VkFence fence) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        check(vkQueueSubmit(queue_, 1, &si, fence), "vkQueueSubmit");
    }
    check(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    check(vkResetFences(device_, 1, &fence), "vkResetFences");
}

int VulkanContext::memoryType(std::uint32_t typeBits, VkMemoryPropertyFlags required,
                              VkMemoryPropertyFlags preferred) const {
    int fallback = -1;
    for (std::uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
        if (!(typeBits & (1u << i))) continue;
        const VkMemoryPropertyFlags f = memory_.memoryTypes[i].propertyFlags;
        if ((f & required) != required) continue;
        if ((f & preferred) == preferred) return static_cast<int>(i);
        if (fallback < 0) fallback = static_cast<int>(i);
    }
    return fallback;
}

Buffer::Buffer(VulkanContext& ctx, VkDeviceSize size, Kind kind) : device_(ctx.device()), size_(size) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device_, &bci, nullptr, &buffer_), "vkCreateBuffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buffer_, &req);
    int type = -1;
    switch (kind) {
        case Kind::Device: type = ctx.memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); break;
        case Kind::Upload:
            type = ctx.memoryType(req.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            break;
        case Kind::Readback:
            type = ctx.memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                  VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            break;
    }
    if (type < 0) type = ctx.memoryType(req.memoryTypeBits, 0);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = static_cast<std::uint32_t>(type);
    check(vkAllocateMemory(device_, &mai, nullptr, &memory_), "vkAllocateMemory");
    check(vkBindBufferMemory(device_, buffer_, memory_, 0), "vkBindBufferMemory");

    if (kind != Kind::Device) {
        check(vkMapMemory(device_, memory_, 0, VK_WHOLE_SIZE, 0, &mapped_), "vkMapMemory");
        VkPhysicalDeviceMemoryProperties props;
        vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice(), &props);
        coherent_ = (props.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }
}

Buffer::~Buffer() { release(); }

Buffer& Buffer::operator=(Buffer&& o) noexcept {
    if (this != &o) {
        release();
        device_ = std::exchange(o.device_, VK_NULL_HANDLE);
        buffer_ = std::exchange(o.buffer_, VK_NULL_HANDLE);
        memory_ = std::exchange(o.memory_, VK_NULL_HANDLE);
        size_ = std::exchange(o.size_, 0);
        mapped_ = std::exchange(o.mapped_, nullptr);
        coherent_ = o.coherent_;
    }
    return *this;
}

void Buffer::release() {
    if (!device_) return;
    if (buffer_) vkDestroyBuffer(device_, buffer_, nullptr);
    if (memory_) vkFreeMemory(device_, memory_, nullptr);  // also unmaps
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
}

void Buffer::invalidate() const {
    if (coherent_ || !mapped_) return;
    VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    r.memory = memory_;
    r.size = VK_WHOLE_SIZE;
    vkInvalidateMappedMemoryRanges(device_, 1, &r);
}

void Buffer::flush() const {
    if (coherent_ || !mapped_) return;
    VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    r.memory = memory_;
    r.size = VK_WHOLE_SIZE;
    vkFlushMappedMemoryRanges(device_, 1, &r);
}

}  // namespace vg::gpu
