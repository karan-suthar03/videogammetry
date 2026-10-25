#include "videogammetry/gpu/gpu_mvs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <unordered_set>
#include <vector>

#include "videogammetry/gpu/vulkan_context.h"

#include "patch_match.h"

namespace vg::gpu {

namespace {

constexpr int kMaxSources = 16;  // as in shaders/patch_match.comp
constexpr std::uint32_t kPushBytes = 128;
enum Mode { kRandomInit = 0, kInitFromCoarse = 1, kSeeds = 2, kPropagate = 3, kFilter = 4 };

// Keep in sync with the push constant block of shaders/patch_match.comp.
struct Push {
    std::int32_t mode, width, height, round;
    std::int32_t color, sources, bestViews, prevWidth;
    std::int32_t prevHeight, seedCount, infoBase, geoBase;
    std::int32_t stateBase, prevBase;
    float fx, fy, cx, cy;
    float minDepth, maxDepth, scale, minPatchStd;
    float maxCost;
};
static_assert(sizeof(Push) <= kPushBytes);

std::uint32_t groups(std::int64_t n, int size) { return static_cast<std::uint32_t>((n + size - 1) / size); }

// Image pyramid level sizes as in mvs.cpp: each level half the one below (floor, at least 1).
std::vector<std::pair<int, int>> levelSizes(int width, int height, int levels) {
    std::vector<std::pair<int, int>> sizes{{width, height}};
    for (int l = 1; l < levels; ++l)
        sizes.push_back({std::max(1, sizes.back().first / 2), std::max(1, sizes.back().second / 2)});
    return sizes;
}

// 2x2 average, as mvs.cpp's halve().
void halve(const float* src, int sw, int sh, float* dst, int dw, int dh) {
    for (int y = 0; y < dh; ++y) {
        const int y0 = std::min(2 * y, sh - 1), y1 = std::min(2 * y + 1, sh - 1);
        for (int x = 0; x < dw; ++x) {
            const int x0 = std::min(2 * x, sw - 1), x1 = std::min(2 * x + 1, sw - 1);
            dst[static_cast<std::size_t>(y) * dw + x] =
                0.25f * (src[static_cast<std::size_t>(y0) * sw + x0] + src[static_cast<std::size_t>(y0) * sw + x1] +
                         src[static_cast<std::size_t>(y1) * sw + x0] + src[static_cast<std::size_t>(y1) * sw + x1]);
        }
    }
}

// A = Ks R Kr^-1, b = Ks t for the relative pose reference -> source.
void sourceGeometry(const Pose& ref, const Pose& src, const PinholeCamera& k, float* out) {
    const Mat3 R = src.R * ref.R.transposed();
    const Vec3 t = src.t - R * ref.t;
    Mat3 K, Kinv;
    K.m = {k.fx, 0, k.cx, 0, k.fy, k.cy, 0, 0, 1};
    Kinv.m = {1 / k.fx, 0, -k.cx / k.fx, 0, 1 / k.fy, -k.cy / k.fy, 0, 0, 1};
    const Mat3 A = K * R * Kinv;
    const Vec3 b = K * t;
    for (int i = 0; i < 9; ++i) out[i] = static_cast<float>(A.m[i]);
    out[9] = static_cast<float>(b.x);
    out[10] = static_cast<float>(b.y);
    out[11] = static_cast<float>(b.z);
}

class GpuPatchMatch {
public:
    enum Binding { kImages, kState, kGeo, kInfo, kSeedBuf, kBindingCount };

    explicit GpuPatchMatch(std::shared_ptr<VulkanContext> c) : ctx_(std::move(c)), dev_(ctx_->device()) {
        std::array<VkDescriptorSetLayoutBinding, kBindingCount> b{};
        for (std::uint32_t i = 0; i < kBindingCount; ++i) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = kBindingCount;
        lci.pBindings = b.data();
        check(vkCreateDescriptorSetLayout(dev_, &lci, nullptr, &setLayout_), "vkCreateDescriptorSetLayout");

        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
        VkPipelineLayoutCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pci.setLayoutCount = 1;
        pci.pSetLayouts = &setLayout_;
        pci.pushConstantRangeCount = 1;
        pci.pPushConstantRanges = &range;
        check(vkCreatePipelineLayout(dev_, &pci, nullptr, &pipelineLayout_), "vkCreatePipelineLayout");

        VkShaderModuleCreateInfo mci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mci.codeSize = sizeof(kPatchMatchSpv);
        mci.pCode = kPatchMatchSpv;
        VkShaderModule module;
        check(vkCreateShaderModule(dev_, &mci, nullptr, &module), "vkCreateShaderModule");
        VkComputePipelineCreateInfo cci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = module;
        cci.stage.pName = "main";
        cci.layout = pipelineLayout_;
        const VkResult r = vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cci, nullptr, &pipeline_);
        vkDestroyShaderModule(dev_, module, nullptr);
        check(r, "vkCreateComputePipelines");

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindingCount};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        check(vkCreateDescriptorPool(dev_, &dpi, nullptr, &pool_), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &setLayout_;
        check(vkAllocateDescriptorSets(dev_, &dai, &set_), "vkAllocateDescriptorSets");

        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = ctx_->queueFamily();
        check(vkCreateCommandPool(dev_, &cpi, nullptr, &cmdPool_), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cmdPool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(dev_, &cai, &cmd_), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(dev_, &fci, nullptr, &fence_), "vkCreateFence");
    }

    ~GpuPatchMatch() {
        vkDeviceWaitIdle(dev_);
        for (Buffer& b : buffers_) b = Buffer();
        staging_ = Buffer();
        readback_ = Buffer();
        vkDestroyFence(dev_, fence_, nullptr);
        vkDestroyCommandPool(dev_, cmdPool_, nullptr);
        vkDestroyDescriptorPool(dev_, pool_, nullptr);
        vkDestroyPipeline(dev_, pipeline_, nullptr);
        vkDestroyPipelineLayout(dev_, pipelineLayout_, nullptr);
        vkDestroyDescriptorSetLayout(dev_, setLayout_, nullptr);
    }

    bool run(const MvsView& ref, const std::vector<const MvsView*>& sourceViews, const PinholeCamera& camera,
             double minDepth, double maxDepth, const std::vector<Vec3>& seeds, const DenseParams& params,
             const std::atomic<bool>* cancel, DepthMap& out) {
        const int w = ref.gray.width, h = ref.gray.height;
        out = DepthMap{};
        out.width = w;
        out.height = h;
        const std::size_t n0 = static_cast<std::size_t>(w) * h;
        out.depth.assign(n0, 0.0f);
        out.normal.assign(n0, Vec3{0, 0, -1});
        out.cost.assign(n0, 2.0f);
        if (sourceViews.empty() || n0 == 0 || !(minDepth > 0) || !(maxDepth > minDepth)) return true;
        const int nsrc = std::min(static_cast<int>(sourceViews.size()), kMaxSources);
        for (int s = 0; s < nsrc; ++s)
            if (sourceViews[static_cast<std::size_t>(s)]->gray.width != w || sourceViews[static_cast<std::size_t>(s)]->gray.height != h)
                return false;
        const int scales = std::max(1, params.scales);
        const auto sizes = levelSizes(w, h, scales);

        // Layout: per level, the reference then the sources (floats); the
        // state of every level; per level the info and geometry tables.
        std::vector<std::size_t> levelImage(scales), levelState(scales);
        std::size_t imageFloats = 0, stateFloats = 0;
        for (int l = 0; l < scales; ++l) {
            const std::size_t px = static_cast<std::size_t>(sizes[l].first) * sizes[l].second;
            levelImage[l] = imageFloats;
            imageFloats += px * (1 + nsrc);
            levelState[l] = stateFloats;
            stateFloats += px * 5;
        }
        if (imageFloats > 0x7FFFFFFF || stateFloats > 0x7FFFFFFF) return false;  // int indices in the shader

        std::vector<std::int32_t> info(static_cast<std::size_t>(scales) * (1 + kMaxSources), 0);
        std::vector<float> geo(static_cast<std::size_t>(scales) * kMaxSources * 12, 0.0f);
        for (int l = 0; l < scales; ++l) {
            const std::size_t px = static_cast<std::size_t>(sizes[l].first) * sizes[l].second;
            for (int i = 0; i <= nsrc; ++i)
                info[static_cast<std::size_t>(l) * (1 + kMaxSources) + i] = static_cast<std::int32_t>(levelImage[l] + px * i);
            const PinholeCamera k = camera.scaled(std::pow(2.0, l));
            for (int s = 0; s < nsrc; ++s)
                sourceGeometry(ref.pose, sourceViews[static_cast<std::size_t>(s)]->pose, k,
                               &geo[(static_cast<std::size_t>(l) * kMaxSources + s) * 12]);
        }

        // Seeds at the coarsest level, one per pixel.
        const int coarse = scales - 1;
        const PinholeCamera kc = camera.scaled(std::pow(2.0, coarse));
        std::vector<float> seedData;
        std::unordered_set<std::int64_t> taken;
        for (const Vec3& wpt : seeds) {
            const Vec3 c = ref.pose.transform(wpt);
            if (c.z <= 0) continue;
            const long x = std::lround(kc.fx * c.x / c.z + kc.cx), y = std::lround(kc.fy * c.y / c.z + kc.cy);
            if (x < 0 || y < 0 || x >= sizes[coarse].first || y >= sizes[coarse].second) continue;
            if (!taken.insert(static_cast<std::int64_t>(y) * sizes[coarse].first + x).second) continue;
            seedData.insert(seedData.end(), {static_cast<float>(x), static_cast<float>(y), static_cast<float>(c.z)});
        }

        // Upload: images (through staging), tables and seeds.
        const std::size_t imageBytes = imageFloats * 4;
        ensureStaging(imageBytes);
        float* stage = static_cast<float*>(staging_.mapped());
        const auto fillLevels = [&](const Image& img, int index) {
            float* base = stage + levelImage[0] + n0 * index;
            for (std::size_t i = 0; i < n0; ++i) base[i] = img.data[i];
            for (int l = 1; l < scales; ++l) {
                const std::size_t pp = static_cast<std::size_t>(sizes[l - 1].first) * sizes[l - 1].second;
                const std::size_t pc = static_cast<std::size_t>(sizes[l].first) * sizes[l].second;
                halve(stage + levelImage[l - 1] + pp * index, sizes[l - 1].first, sizes[l - 1].second,
                      stage + levelImage[l] + pc * index, sizes[l].first, sizes[l].second);
            }
        };
        fillLevels(ref.gray, 0);
        for (int s = 0; s < nsrc; ++s) fillLevels(sourceViews[static_cast<std::size_t>(s)]->gray, 1 + s);
        staging_.flush();
        ensure(kImages, imageBytes, Buffer::Kind::Device);
        ensure(kState, stateFloats * 4, Buffer::Kind::Device);
        ensure(kGeo, geo.size() * 4, Buffer::Kind::Upload);
        ensure(kInfo, info.size() * 4, Buffer::Kind::Upload);
        ensure(kSeedBuf, std::max<std::size_t>(seedData.size(), 3) * 4, Buffer::Kind::Upload);
        std::memcpy(buffers_[kGeo].mapped(), geo.data(), geo.size() * 4);
        std::memcpy(buffers_[kInfo].mapped(), info.data(), info.size() * 4);
        if (!seedData.empty()) std::memcpy(buffers_[kSeedBuf].mapped(), seedData.data(), seedData.size() * 4);
        buffers_[kGeo].flush();
        buffers_[kInfo].flush();
        buffers_[kSeedBuf].flush();

        Push base{};
        base.sources = nsrc;
        base.bestViews = params.bestViews;
        base.seedCount = static_cast<std::int32_t>(seedData.size() / 3);
        base.minDepth = static_cast<float>(minDepth);
        base.maxDepth = static_cast<float>(maxDepth);
        base.minPatchStd = static_cast<float>(params.minPatchStd);
        base.maxCost = static_cast<float>(1.0 - params.minNcc);

        // One submission per propagation round keeps every submission short
        // (no driver timeout on full-size images).
        int round = 0;
        bool first = true;
        for (int l = scales - 1; l >= 0; --l) {
            Push p = base;
            const PinholeCamera k = camera.scaled(std::pow(2.0, l));
            p.width = sizes[l].first;
            p.height = sizes[l].second;
            p.fx = static_cast<float>(k.fx), p.fy = static_cast<float>(k.fy);
            p.cx = static_cast<float>(k.cx), p.cy = static_cast<float>(k.cy);
            p.infoBase = l * (1 + kMaxSources);
            p.geoBase = l * kMaxSources * 12;
            p.stateBase = static_cast<std::int32_t>(levelState[l]);
            const std::uint32_t gx = groups(p.width, 16), gy = groups(p.height, 8);

            begin();
            if (first) {
                VkBufferCopy copy{0, 0, imageBytes};
                vkCmdCopyBuffer(cmd_, staging_.handle(), buffers_[kImages].handle(), 1, &copy);
                barrier();
            }
            if (l == scales - 1) {
                p.mode = kRandomInit;
                dispatch(p, gx, gy);
                if (base.seedCount > 0) {
                    barrier();
                    p.mode = kSeeds;
                    dispatch(p, groups(base.seedCount, 16), 1);
                }
            } else {
                p.mode = kInitFromCoarse;
                p.prevWidth = sizes[l + 1].first;
                p.prevHeight = sizes[l + 1].second;
                p.prevBase = static_cast<std::int32_t>(levelState[l + 1]);
                dispatch(p, gx, gy);
            }
            submit();
            first = false;

            const int iterations = l == scales - 1 ? params.iterations : params.fineIterations;
            for (int it = 0; it < iterations; ++it, ++round) {
                if (cancel && *cancel) return true;
                begin();
                p.mode = kPropagate;
                p.round = round;
                p.scale = static_cast<float>(std::pow(0.5, it));
                for (int color = 0; color < 2; ++color) {
                    p.color = color;
                    barrier();
                    dispatch(p, groups((p.width + 1) / 2, 16), gy);
                }
                submit();
            }
            if (l == 0) {
                begin();
                barrier();
                p.mode = kFilter;
                dispatch(p, gx, gy);
                const VkDeviceSize bytes = n0 * 5 * 4;
                ensureReadback(bytes);
                barrierToTransfer();
                VkBufferCopy copy{levelState[0] * 4, 0, bytes};
                vkCmdCopyBuffer(cmd_, buffers_[kState].handle(), readback_.handle(), 1, &copy);
                barrierToHost();
                submit();
                readback_.invalidate();
                const float* st = static_cast<const float*>(readback_.mapped());
                for (std::size_t i = 0; i < n0; ++i) {
                    out.depth[i] = st[i * 5];
                    out.normal[i] = {st[i * 5 + 1], st[i * 5 + 2], st[i * 5 + 3]};
                    out.cost[i] = st[i * 5 + 4];
                }
            }
        }
        return true;
    }

private:
    void ensure(Binding b, std::size_t bytes, Buffer::Kind kind) {
        bytes = std::max<std::size_t>(bytes, 64);
        if (buffers_[b].handle() && buffers_[b].size() >= bytes) return;
        buffers_[b] = Buffer(*ctx_, static_cast<VkDeviceSize>(bytes + bytes / 8), kind);
        VkDescriptorBufferInfo info{buffers_[b].handle(), 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wr.dstSet = set_;
        wr.dstBinding = b;
        wr.descriptorCount = 1;
        wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr.pBufferInfo = &info;
        vkUpdateDescriptorSets(dev_, 1, &wr, 0, nullptr);
    }
    void ensureStaging(std::size_t bytes) {
        if (!staging_.handle() || staging_.size() < bytes)
            staging_ = Buffer(*ctx_, static_cast<VkDeviceSize>(std::max<std::size_t>(bytes, 64)), Buffer::Kind::Upload);
    }
    void ensureReadback(std::size_t bytes) {
        if (!readback_.handle() || readback_.size() < bytes)
            readback_ = Buffer(*ctx_, static_cast<VkDeviceSize>(std::max<std::size_t>(bytes, 64)), Buffer::Kind::Readback);
    }

    void begin() {
        check(vkResetCommandBuffer(cmd_, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd_, &bi), "vkBeginCommandBuffer");
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &set_, 0, nullptr);
    }
    void submit() {
        check(vkEndCommandBuffer(cmd_), "vkEndCommandBuffer");
        ctx_->submitAndWait(cmd_, fence_);
    }
    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    void barrierToTransfer() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    }
    void barrierToHost() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0,
                             nullptr);
    }
    void dispatch(const Push& p, std::uint32_t gx, std::uint32_t gy) {
        if (gx == 0 || gy == 0) return;
        vkCmdPushConstants(cmd_, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &p);
        vkCmdDispatch(cmd_, gx, gy, 1);
    }

    std::shared_ptr<VulkanContext> ctx_;
    VkDevice dev_;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    std::array<Buffer, kBindingCount> buffers_;
    Buffer staging_, readback_;
};

}  // namespace

DepthMapSolver makeGpuDepthSolver(std::string& status) {
    std::string error;
    std::shared_ptr<VulkanContext> ctx = VulkanContext::create(error);
    if (!ctx) {
        status = "CPU (no GPU: " + error + ")";
        return {};
    }
    std::shared_ptr<GpuPatchMatch> solver;
    try {
        solver = std::make_shared<GpuPatchMatch>(ctx);
    } catch (const std::exception& e) {
        status = std::string("CPU (GPU setup failed: ") + e.what() + ")";
        return {};
    }
    status = "GPU (" + ctx->deviceName() + ", Vulkan)";
    return [solver](const MvsView& ref, const std::vector<const MvsView*>& sources, const PinholeCamera& camera,
                    double minDepth, double maxDepth, const std::vector<Vec3>& seeds, const DenseParams& params,
                    const std::atomic<bool>* cancel, DepthMap& out) {
        try {
            return solver->run(ref, sources, camera, minDepth, maxDepth, seeds, params, cancel, out);
        } catch (const std::exception&) {
            return false;  // e.g. out of GPU memory: the CPU takes over
        }
    };
}

}  // namespace vg::gpu
