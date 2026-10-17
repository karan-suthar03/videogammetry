#include "videogammetry/gpu/gpu_orb.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "videogammetry/orb_training.h"

#include "blur_h.h"
#include "blur_v.h"
#include "describe.h"
#include "fast_cells.h"
#include "fast_collect.h"
#include "gray.h"
#include "resize.h"

namespace vg::gpu {

namespace {

// Keep in sync with shaders/common.glsl.
constexpr int kParamCandidateCapacity = 0, kParamMaximaCapacity = 1, kParamPattern = 2, kParamLevels = 3;
constexpr int kLevelWords = 16;
enum LevelField {
    kLvWidth, kLvHeight, kLvImage, kLvStride, kLvBlurred, kLvScore, kLvCells, kLvX0, kLvY0, kLvX1, kLvY1,
    kLvCell, kLvCols, kLvThreshold, kLvMinThreshold
};
constexpr std::uint32_t kResCandidates = 0, kResMaxima = 1, kResHeader = 16, kRecordWords = 16;
constexpr int kMaxGpuLevels = 32;          // more (absurd scale factors) go to the CPU
constexpr int kMaxGpuSide = 65535;         // keypoint coordinates are packed in 16 bits
constexpr std::uint32_t kPushBytes = 64;

enum Shader { kGray, kResize, kBlurH, kBlurV, kFastCells, kFastCollect, kDescribe, kShaderCount };

struct Spv {
    const std::uint32_t* code;
    std::size_t bytes;
};

const std::array<Spv, kShaderCount>& shaderCode() {
    static const std::array<Spv, kShaderCount> code = {{
        {kGraySpv, sizeof(kGraySpv)},
        {kResizeSpv, sizeof(kResizeSpv)},
        {kBlurHSpv, sizeof(kBlurHSpv)},
        {kBlurVSpv, sizeof(kBlurVSpv)},
        {kFastCellsSpv, sizeof(kFastCellsSpv)},
        {kFastCollectSpv, sizeof(kFastCollectSpv)},
        {kDescribeSpv, sizeof(kDescribeSpv)},
    }};
    return code;
}

int strideWords(int width) { return (width + 3) / 4; }

std::uint32_t groups(std::int64_t n, int size) { return static_cast<std::uint32_t>((n + size - 1) / size); }

}  // namespace

struct GpuFeatureExtractor::Impl {
    enum Binding { kImages, kTemp, kScores, kCells, kCandidates, kResults, kParams, kFrame, kBindingCount };

    std::shared_ptr<VulkanContext> ctx;
    VkDevice dev;
    int processSize;
    OrbParams orb;
    OrbExtractor plan;  // level scales and budgets, exactly as the CPU uses them
    std::unique_ptr<FeatureExtractor> cpu;  // fallback

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    std::array<VkPipeline, kShaderCount> pipelines{};
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::array<Buffer, kBindingCount> buffers;

    std::uint32_t candidateCapacity = 0, maximaCapacity = 0;

    // Per-frame layout.
    struct Level {
        int width = 0, height = 0, image = 0, stride = 0, blurred = 0, score = 0, cells = 0;
        int budget = 0;
        OrbGrid grid{};
        bool detect = false;
        int tapsX = 0, tapsY = 0;  // into params: resize from the previous level (or the gray frame)
    };
    std::vector<Level> levels;
    std::vector<int> params;
    std::vector<int> taps, tapWeights;

    Impl(std::shared_ptr<VulkanContext> c, int size, const OrbParams& o)
        : ctx(std::move(c)), dev(ctx->device()), processSize(size), orb(o), plan(o) {
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
        check(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &setLayout), "vkCreateDescriptorSetLayout");

        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
        VkPipelineLayoutCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pci.setLayoutCount = 1;
        pci.pSetLayouts = &setLayout;
        pci.pushConstantRangeCount = 1;
        pci.pPushConstantRanges = &range;
        check(vkCreatePipelineLayout(dev, &pci, nullptr, &pipelineLayout), "vkCreatePipelineLayout");

        for (int s = 0; s < kShaderCount; ++s) {
            VkShaderModuleCreateInfo mci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            mci.codeSize = shaderCode()[s].bytes;
            mci.pCode = shaderCode()[s].code;
            VkShaderModule module;
            check(vkCreateShaderModule(dev, &mci, nullptr, &module), "vkCreateShaderModule");
            VkComputePipelineCreateInfo cci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cci.stage.module = module;
            cci.stage.pName = "main";
            cci.layout = pipelineLayout;
            const VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cci, nullptr, &pipelines[s]);
            vkDestroyShaderModule(dev, module, nullptr);
            check(r, "vkCreateComputePipelines");
        }

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindingCount};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        check(vkCreateDescriptorPool(dev, &dpi, nullptr, &pool), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &setLayout;
        check(vkAllocateDescriptorSets(dev, &dai, &set), "vkAllocateDescriptorSets");

        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = ctx->queueFamily();
        check(vkCreateCommandPool(dev, &cpi, nullptr, &cmdPool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cmdPool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(dev, &cai, &cmd), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(dev, &fci, nullptr, &fence), "vkCreateFence");
    }

    ~Impl() {
        vkDeviceWaitIdle(dev);
        for (Buffer& b : buffers) b = Buffer();
        vkDestroyFence(dev, fence, nullptr);
        vkDestroyCommandPool(dev, cmdPool, nullptr);
        vkDestroyDescriptorPool(dev, pool, nullptr);
        for (VkPipeline p : pipelines) vkDestroyPipeline(dev, p, nullptr);
        vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(dev, setLayout, nullptr);
    }

    // Grows buffer `b` to at least `bytes` (with headroom) and rebinds it.
    void ensure(Binding b, std::size_t bytes, Buffer::Kind kind) {
        bytes = std::max<std::size_t>(bytes, 64);
        if (buffers[b].handle() && buffers[b].size() >= bytes) return;
        buffers[b] = Buffer(*ctx, static_cast<VkDeviceSize>(bytes + bytes / 4), kind);
        VkDescriptorBufferInfo info{buffers[b].handle(), 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = b;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &info;
        vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    }

    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    template <typename Push>
    void dispatch(Shader s, const Push& push, std::uint32_t gx, std::uint32_t gy = 1) {
        static_assert(sizeof(Push) <= kPushBytes);
        if (gx == 0 || gy == 0) return;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[s]);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
        vkCmdDispatch(cmd, gx, gy, 1);
    }

    // Appends resize taps from `src` to `dst` samples to the params table.
    int addTaps(int src, int dst, int base) {
        bilinearTaps(src, dst, taps, tapWeights);
        const int at = base + static_cast<int>(params.size());
        for (int i = 0; i < dst; ++i) {
            params.push_back(taps[i]);
            params.push_back(tapWeights[i]);
        }
        return at;
    }

    // Returns false if this configuration should go to the CPU.
    bool run(const FrameView& frame, FrameFeatures& out);
};

bool GpuFeatureExtractor::Impl::run(const FrameView& frame, FrameFeatures& out) {
    const int bpp = frame.format == FrameView::Format::Gray8 ? 1 : (frame.format == FrameView::Format::RGB8 ? 3 : 4);
    const int fw = frame.width, fh = frame.height;

    // Processed size, exactly as FeatureExtractor::extract.
    int w = fw, h = fh;
    out.processScale = 1.0f;
    const int longSide = std::max(fw, fh);
    const bool resizeInput = processSize > 0 && longSide > processSize;
    if (resizeInput) {
        const double s = double(longSide) / processSize;
        w = std::max(1, static_cast<int>(std::lround(fw / s)));
        h = std::max(1, static_cast<int>(std::lround(fh / s)));
        out.processScale = float(fw) / w;
    }
    out.width = w;
    out.height = h;
    if (std::max(w, h) > kMaxGpuSide) return false;

    // Pyramid levels that hold any pixels (coarser ones are all empty too).
    const std::vector<float>& scales = plan.scales();
    const std::vector<int>& budgets = plan.budgets();
    int n = 0;
    while (n < static_cast<int>(scales.size())) {
        int lw = w, lh = h;
        if (n > 0) OrbExtractor::levelSize(w, h, scales[n], lw, lh);
        if (lw == 0) break;
        ++n;
    }
    if (n > kMaxGpuLevels) return false;

    // Layout: images (words), score maps and cell flags per detected level.
    levels.assign(static_cast<std::size_t>(n), Level{});
    int imageWords = resizeInput ? strideWords(fw) * fh : 0;  // full-size gray first, if resized
    const int grayOffset = 0;
    int scoreFloats = 0, cellWords = 0, totalBudget = 0;
    for (int l = 0; l < n; ++l) {
        Level& L = levels[l];
        L.width = w;
        L.height = h;
        if (l > 0) OrbExtractor::levelSize(w, h, scales[l], L.width, L.height);
        L.stride = strideWords(L.width);
        L.image = imageWords;
        imageWords += L.stride * L.height;
        L.budget = budgets[l];
        L.detect = L.width > 2 * kOrbBorder && L.height > 2 * kOrbBorder && L.budget > 0;
        if (!L.detect) continue;
        L.blurred = imageWords;
        imageWords += L.stride * L.height;
        L.grid = orbGrid(L.width, L.height, L.budget);
        L.score = scoreFloats;
        scoreFloats += L.width * L.height;
        L.cells = cellWords;
        cellWords += L.grid.cols * L.grid.rows;
        totalBudget += L.budget;
    }

    if (candidateCapacity == 0) {
        candidateCapacity = static_cast<std::uint32_t>(std::max(1 << 18, 32 * totalBudget));
        maximaCapacity = static_cast<std::uint32_t>(std::max(1 << 16, 8 * totalBudget));
    }

    // Params table (taps after the level table, then the BRIEF pattern).
    params.assign(16 + static_cast<std::size_t>(n) * kLevelWords, 0);
    params[kParamLevels] = 16;
    const int head = 0;
    if (resizeInput) {
        // Input -> level 0.
        const int tx = addTaps(fw, w, head), ty = addTaps(fh, h, head);
        levels[0].tapsX = tx;
        levels[0].tapsY = ty;
    }
    for (int l = 1; l < n; ++l) {
        levels[l].tapsX = addTaps(levels[l - 1].width, levels[l].width, head);
        levels[l].tapsY = addTaps(levels[l - 1].height, levels[l].height, head);
    }
    params[kParamPattern] = static_cast<int>(params.size());
    for (const BriefTest& t : learnedBriefPattern()) {
        params.push_back(int(std::uint8_t(t.x1)) | (int(std::uint8_t(t.y1)) << 8) | (int(std::uint8_t(t.x2)) << 16) |
                         static_cast<int>(static_cast<std::uint32_t>(std::uint8_t(t.y2)) << 24));
    }
    for (int l = 0; l < n; ++l) {
        const Level& L = levels[l];
        int* e = params.data() + 16 + l * kLevelWords;
        e[kLvWidth] = L.width;
        e[kLvHeight] = L.height;
        e[kLvImage] = L.image;
        e[kLvStride] = L.stride;
        e[kLvBlurred] = L.blurred;
        e[kLvScore] = L.score;
        e[kLvCells] = L.cells;
        e[kLvX0] = L.grid.x0;
        e[kLvY0] = L.grid.y0;
        e[kLvX1] = L.grid.x1;
        e[kLvY1] = L.grid.y1;
        e[kLvCell] = L.grid.cell;
        e[kLvCols] = L.grid.cols;
        e[kLvThreshold] = orb.fastThreshold;
        e[kLvMinThreshold] = orb.minFastThreshold;
    }

    // Buffers (grown as needed) and the frame upload.
    const std::size_t frameBytes = static_cast<std::size_t>(fw) * fh * bpp;
    ensure(kFrame, frameBytes + 4, Buffer::Kind::Upload);
    for (int y = 0; y < fh; ++y) {
        std::memcpy(static_cast<std::uint8_t*>(buffers[kFrame].mapped()) + static_cast<std::size_t>(y) * fw * bpp,
                    frame.data + static_cast<std::size_t>(y) * frame.stride, static_cast<std::size_t>(fw) * bpp);
    }
    buffers[kFrame].flush();
    ensure(kImages, static_cast<std::size_t>(imageWords) * 4, Buffer::Kind::Device);
    ensure(kTemp, static_cast<std::size_t>(w) * h * 4, Buffer::Kind::Device);
    ensure(kScores, static_cast<std::size_t>(scoreFloats) * 4, Buffer::Kind::Device);
    ensure(kCells, static_cast<std::size_t>(cellWords) * 4, Buffer::Kind::Device);

    for (int attempt = 0;; ++attempt) {
        params[kParamCandidateCapacity] = static_cast<int>(candidateCapacity);
        params[kParamMaximaCapacity] = static_cast<int>(maximaCapacity);
        ensure(kParams, params.size() * 4, Buffer::Kind::Upload);
        std::memcpy(buffers[kParams].mapped(), params.data(), params.size() * 4);
        buffers[kParams].flush();
        ensure(kCandidates, static_cast<std::size_t>(candidateCapacity) * 12, Buffer::Kind::Device);
        ensure(kResults, (kResHeader + static_cast<std::size_t>(maximaCapacity) * kRecordWords) * 4,
               Buffer::Kind::Readback);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdFillBuffer(cmd, buffers[kResults].handle(), 0, kResHeader * 4, 0);
        if (scoreFloats > 0) vkCmdFillBuffer(cmd, buffers[kScores].handle(), 0, static_cast<VkDeviceSize>(scoreFloats) * 4, 0);
        if (cellWords > 0) vkCmdFillBuffer(cmd, buffers[kCells].handle(), 0, static_cast<VkDeviceSize>(cellWords) * 4, 0);
        barrier();

        // Gray, then the pyramid (each level from the previous one).
        {
            const int target = resizeInput ? grayOffset : levels[0].image;
            struct { int width, height, bpp, dst, stride; } p{fw, fh, bpp, target, strideWords(fw)};
            dispatch(kGray, p, groups(static_cast<std::int64_t>(strideWords(fw)) * fh, 64));
            barrier();
        }
        for (int l = resizeInput ? 0 : 1; l < n; ++l) {
            const bool fromInput = l == 0;
            const int srcOff = fromInput ? grayOffset : levels[l - 1].image;
            const int srcStride = fromInput ? strideWords(fw) : levels[l - 1].stride;
            const int srcW = fromInput ? fw : levels[l - 1].width, srcH = fromInput ? fh : levels[l - 1].height;
            const Level& L = levels[l];
            struct { int so, ss, sw, sh, d, ds, dw, dh, tx, ty; } p{srcOff, srcStride, srcW, srcH, L.image, L.stride,
                                                                    L.width, L.height, L.tapsX, L.tapsY};
            dispatch(kResize, p, groups(static_cast<std::int64_t>(L.stride) * L.height, 64));
            barrier();
        }

        // Per detected level: blur (for descriptors), then FAST in two passes.
        for (int l = 0; l < n; ++l) {
            const Level& L = levels[l];
            if (!L.detect) continue;
            struct { int off, stride, width, height; } pb{L.image, L.stride, L.width, L.height};
            dispatch(kBlurH, pb, groups(static_cast<std::int64_t>(L.width) * L.height, 64));
            barrier();
            pb.off = L.blurred;
            dispatch(kBlurV, pb, groups(static_cast<std::int64_t>(L.stride) * L.height, 64));
            barrier();
            struct { int lv; } pl{l};
            const std::uint32_t gx = groups(L.grid.x1 - L.grid.x0, 16), gy = groups(L.grid.y1 - L.grid.y0, 16);
            dispatch(kFastCells, pl, gx, gy);
            barrier();
            dispatch(kFastCollect, pl, gx, gy);
            barrier();
        }

        struct { int unused; } pd{0};
        dispatch(kDescribe, pd, groups(candidateCapacity, 64));
        VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        hb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        ctx->submitAndWait(cmd, fence);
        buffers[kResults].invalidate();

        const auto* res = static_cast<const std::uint32_t*>(buffers[kResults].mapped());
        const std::uint32_t candidates = res[kResCandidates], maxima = res[kResMaxima];
        if (candidates <= candidateCapacity && maxima <= maximaCapacity) break;
        // Not enough room this time (e.g. a FAST threshold of 0): grow and redo.
        if (attempt >= 3) throw std::runtime_error("GPU feature buffers keep overflowing");
        candidateCapacity = std::max(candidateCapacity, candidates + candidates / 4);
        maximaCapacity = std::max(maximaCapacity, maxima + maxima / 4);
    }

    // Host: per level, the local maxima in detection order, spread over the
    // grid by the CPU's own code.
    const auto* res = static_cast<const std::uint32_t*>(buffers[kResults].mapped());
    const std::uint32_t maxima = res[kResMaxima];
    struct Record {
        int cellIndex, x, y;
        std::uint32_t index;
    };
    std::vector<std::vector<Record>> perLevel(static_cast<std::size_t>(n));
    for (std::uint32_t m = 0; m < maxima; ++m) {
        const std::uint32_t* r = res + kResHeader + static_cast<std::size_t>(m) * kRecordWords;
        const int x = static_cast<int>(r[0] & 0xFFFFu), y = static_cast<int>(r[0] >> 16);
        const int l = static_cast<int>(r[1]);
        perLevel[l].push_back({levels[l].grid.cellOf(x, y), x, y, m});
    }

    out.keypoints.clear();
    out.descriptors.clear();
    std::vector<OrbCorner> corners, chosen;
    for (int l = 0; l < n; ++l) {
        std::vector<Record>& recs = perLevel[l];
        if (recs.empty()) continue;
        std::sort(recs.begin(), recs.end(), [](const Record& a, const Record& b) {
            return a.cellIndex != b.cellIndex ? a.cellIndex < b.cellIndex : (a.y != b.y ? a.y < b.y : a.x < b.x);
        });
        corners.clear();
        for (const Record& rc : recs) {
            float score;
            std::memcpy(&score, res + kResHeader + static_cast<std::size_t>(rc.index) * kRecordWords + 2, 4);
            corners.push_back({rc.x, rc.y, score});
        }
        chosen.clear();
        distributeOrbCorners(levels[l].grid, corners, levels[l].budget, orb.centerBoost, chosen);

        const float scale = scales[l];
        for (const OrbCorner& c : chosen) {
            const auto it = std::lower_bound(recs.begin(), recs.end(), c, [&](const Record& a, const OrbCorner& k) {
                const int kc = levels[l].grid.cellOf(k.x, k.y);
                return a.cellIndex != kc ? a.cellIndex < kc : (a.y != k.y ? a.y < k.y : a.x < k.x);
            });
            const std::uint32_t* r = res + kResHeader + static_cast<std::size_t>(it->index) * kRecordWords;
            Keypoint kp;
            kp.angle = std::atan2(float(static_cast<std::int32_t>(r[3])), float(static_cast<std::int32_t>(r[4])));
            kp.x = (c.x + 0.5f) * scale - 0.5f;
            kp.y = (c.y + 0.5f) * scale - 0.5f;
            kp.response = c.score;
            kp.level = l;
            kp.scale = scale;
            out.keypoints.push_back(kp);
            Descriptor d;
            std::memcpy(d.data(), r + 6, sizeof(d));
            out.descriptors.push_back(d);
        }
    }

    // Colors from the full-resolution frame (as FeatureExtractor::extract).
    out.colors.resize(out.keypoints.size());
    for (std::size_t i = 0; i < out.keypoints.size(); ++i) {
        const Keypoint& k = out.keypoints[i];
        const int x = std::clamp(static_cast<int>(std::lround((k.x + 0.5f) * out.processScale - 0.5f)), 0, fw - 1);
        const int y = std::clamp(static_cast<int>(std::lround((k.y + 0.5f) * out.processScale - 0.5f)), 0, fh - 1);
        const std::uint8_t* p = frame.data + static_cast<std::size_t>(y) * frame.stride + static_cast<std::size_t>(x) * bpp;
        out.colors[i] = bpp == 1 ? std::array<std::uint8_t, 3>{p[0], p[0], p[0]}
                                 : std::array<std::uint8_t, 3>{p[0], p[1], p[2]};
    }
    return true;
}

GpuFeatureExtractor::GpuFeatureExtractor(std::shared_ptr<VulkanContext> context, int processSize, const OrbParams& orb)
    : impl_(std::make_unique<Impl>(std::move(context), processSize, orb)) {}

GpuFeatureExtractor::~GpuFeatureExtractor() = default;

void GpuFeatureExtractor::extract(const FrameView& frame, long index, FrameFeatures& out) {
    out.index = index;
    out.timestamp = frame.timestamp;
    if (!usedCpu_ && frame.width > 0 && frame.height > 0 && impl_->run(frame, out)) return;
    usedCpu_ = true;
    if (!impl_->cpu) impl_->cpu = std::make_unique<FeatureExtractor>(impl_->processSize, impl_->orb);
    impl_->cpu->extract(frame, index, out);
}

FrameExtractorFactory makeGpuExtractorFactory(int processSize, const OrbParams& orb, std::string& status) {
    std::string error;
    std::shared_ptr<VulkanContext> ctx = VulkanContext::create(error);
    if (!ctx) {
        status = "CPU (no GPU: " + error + ")";
        return {};
    }
    status = "GPU (" + ctx->deviceName() + ", Vulkan)";
    return [ctx, processSize, orb]() -> std::unique_ptr<FrameExtractor> {
        return std::make_unique<GpuFeatureExtractor>(ctx, processSize, orb);
    };
}

}  // namespace vg::gpu
