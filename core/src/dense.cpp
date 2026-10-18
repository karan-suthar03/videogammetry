#include "videogammetry/dense.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "videogammetry/image.h"
#include "videogammetry/linalg.h"
#include "videogammetry/reconstructor.h"
#include "videogammetry/thread_pool.h"
#include "videogammetry/two_view.h"

namespace vg {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int kPyramid = 4;  // image levels: imageSize, 1/2, 1/4, 1/8
constexpr double kPi = 3.14159265358979323846;
constexpr int kPatch = 5;                          // samples per side
constexpr int kPatchSpacing = 2;                   // pixels between samples
constexpr int kPatchRadius = (kPatch / 2) * kPatchSpacing;
constexpr int kPatchSize = kPatch * kPatch;

// Quality tiers: a cell starts at 0 (fast) and climbs while the solver is idle.
struct Tier {
    int step;        // pixels between solved pixels
    int depths;      // plane sweep samples
    int neighbours;  // views matched against the reference
    int fineSteps;   // finer search around the best sample
    int references;  // reference cameras per cell
    int regionScale; // times DenseParams::maxRegionPx (finer image levels)
};
constexpr Tier kTiers[] = {
    {4, 16, 3, 4, 1, 1}, {2, 24, 4, 6, 1, 1}, {2, 32, 5, 8, 2, 1},
    {1, 48, 6, 8, 2, 2}, {1, 64, 8, 12, 3, 2}, {1, 96, 12, 16, 4, 2},
};
constexpr int kMaxTier = static_cast<int>(sizeof(kTiers) / sizeof(kTiers[0])) - 1;
enum Mode { kFirst, kNewViews, kUpgrade, kRotate };

// A keyframe image kept for matching: gray pyramid and color at half size.
struct KeyImage {
    Image gray[kPyramid];
    std::vector<std::uint8_t> rgb;  // level 1 size
    int rgbWidth = 0, rgbHeight = 0;
    double fromInput = 1.0;  // level-0 pixels per input-frame pixel
    std::vector<std::uint8_t> sky;  // level 2: 1 = sky
    std::vector<float> surface;     // level 2: nearest dense depth measured from here (0 = none)
    bool isSky(int x2, int y2) const {  // with its 8 neighbours
        const Image& g = gray[2];
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int x = std::clamp(x2 + dx, 0, g.width - 1), y = std::clamp(y2 + dy, 0, g.height - 1);
                if (!sky[static_cast<std::size_t>(y) * g.width + x]) return false;
            }
        return true;
    }
};

// Sky: smooth, bright pixels connected to the top of the image through
// smooth bright pixels (level 2).
void computeSky(KeyImage& k) {
    const Image& g = k.gray[2];
    const int w = g.width, h = g.height;
    std::vector<int> hist(256, 0);
    for (std::uint8_t v : g.data) ++hist[v];
    int p90 = 255;
    for (int acc = 0, v = 0; v < 256; ++v) {
        acc += hist[v];
        if (acc >= 0.9 * g.data.size()) {
            p90 = v;
            break;
        }
    }
    const int bright = std::max(60, p90 * 6 / 10);
    std::vector<std::uint8_t> candidate(g.data.size(), std::uint8_t{0});
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int c = g.at(x, y);
            const int gx = std::abs(g.at(std::min(x + 1, w - 1), y) - g.at(std::max(x - 1, 0), y));
            const int gy = std::abs(g.at(x, std::min(y + 1, h - 1)) - g.at(x, std::max(y - 1, 0)));
            candidate[static_cast<std::size_t>(y) * w + x] = c >= bright && gx + gy <= 6;
        }
    k.sky.assign(g.data.size(), std::uint8_t{0});
    std::vector<int> stack;
    for (int x = 0; x < w; ++x)
        if (candidate[x]) {
            k.sky[x] = 1;
            stack.push_back(x);
        }
    while (!stack.empty()) {
        const int i = stack.back();
        stack.pop_back();
        const int x = i % w, y = i / w;
        const int next[4] = {x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1};
        for (int j : next)
            if (j >= 0 && candidate[j] && !k.sky[j]) {
                k.sky[j] = 1;
                stack.push_back(j);
            }
    }
    k.surface.assign(g.data.size(), 0.0f);
}

struct Pending {
    long frame = 0;
    std::vector<std::uint8_t> pixels;
    FrameView view;
};

// What the solver needs from the map, copied on the tracking thread.
struct Snapshot {
    CameraIntrinsics K;  // processed image
    float processScale = 1.0f;
    struct Keyframe {
        long frame = 0;
        Pose pose;
        Vec3 center;
        std::vector<int> points;  // into `points`
    };
    std::vector<Vec3> points;
    std::vector<int> mapId;  // per point: index in Reconstructor::mapPoints()
    std::size_t mapPoints = 0;
    std::vector<Keyframe> keyframes;
};

// Dense points of one cell measured from one reference keyframe: raw
// (distorted) processed-image pixel and depth along the reference camera's z.
struct DensePoint {
    float u = 0, v = 0, z = 0;
    std::uint8_t r = 0, g = 0, b = 0;
};

struct Batch {
    long ref = 0;  // keyframe frame index
    int tier = 0;
    Clock::time_point at;
    std::vector<DensePoint> points;
};

// A cell of the volume hierarchy.
struct Node {
    int level = 0, x = 0, y = 0, z = 0;
    // From the latest snapshot.
    int sparse = 0;
    std::vector<std::pair<int, int>> observers;  // snapshot keyframe, its map points in the cell
    int newest = -1;                             // newest observing snapshot keyframe
    // Solving.
    bool solved = false;      // solved at least once (with or without points)
    bool fine = false;        // matched at full image size: no children needed
    int triedObservers = 0;   // observer count at the last attempt
    int solvedObservers = 0;  // ... at the last successful solve
    int triedSparse = 0;      // map points in it at the last attempt
    int tier = -1;            // quality tier of the last solve
    std::vector<long> refsUsed;  // reference cameras tried at the top tier
    bool exhausted = false;   // every camera seeing it has been a reference
    std::vector<Batch> batches;
    bool hasPoints() const {
        for (const Batch& b : batches)
            if (!b.points.empty()) return true;
        return false;
    }
};

std::uint64_t nodeKey(int level, int x, int y, int z) {
    constexpr int kOffset = 1 << 19;
    return (static_cast<std::uint64_t>(level) << 60) | (static_cast<std::uint64_t>(x + kOffset) & 0xFFFFF) << 40 |
           (static_cast<std::uint64_t>(y + kOffset) & 0xFFFFF) << 20 | (static_cast<std::uint64_t>(z + kOffset) & 0xFFFFF);
}

int cellIndex(double coord, double cell) { return static_cast<int>(std::floor(coord / cell)); }

// Halves an image by averaging 2x2 blocks.
void halve(const Image& src, Image& dst) {
    dst.resize(std::max(1, src.width / 2), std::max(1, src.height / 2));
    for (int y = 0; y < dst.height; ++y) {
        const std::uint8_t* a = src.row(std::min(2 * y, src.height - 1));
        const std::uint8_t* b = src.row(std::min(2 * y + 1, src.height - 1));
        std::uint8_t* out = dst.row(y);
        for (int x = 0; x < dst.width; ++x) {
            const int x0 = std::min(2 * x, src.width - 1), x1 = std::min(2 * x + 1, src.width - 1);
            out[x] = static_cast<std::uint8_t>((a[x0] + a[x1] + b[x0] + b[x1] + 2) / 4);
        }
    }
}

// Gray pyramid and half-size color of one keyframe image.
KeyImage makeKeyImage(const FrameView& view, int imageSize) {
    KeyImage k;
    Image full;
    toGray(view, full);
    const int longSide = std::max(full.width, full.height);
    const double s = imageSize > 0 && longSide > imageSize ? double(imageSize) / longSide : 1.0;
    const int w = std::max(1, static_cast<int>(std::lround(full.width * s)));
    const int h = std::max(1, static_cast<int>(std::lround(full.height * s)));
    // Halve while that stays above the target (no aliasing), then resample.
    Image src = std::move(full), tmp;
    while (src.width >= 2 * w && src.height >= 2 * h) {
        halve(src, tmp);
        std::swap(src, tmp);
    }
    if (src.width != w || src.height != h) {
        resizeBilinear(src, w, h, k.gray[0]);
    } else {
        k.gray[0] = std::move(src);
    }
    k.fromInput = double(w) / view.width;
    for (int l = 1; l < kPyramid; ++l) halve(k.gray[l - 1], k.gray[l]);
    computeSky(k);

    // Color at level 1: box average of the input pixels each one covers.
    k.rgbWidth = k.gray[1].width;
    k.rgbHeight = k.gray[1].height;
    k.rgb.resize(static_cast<std::size_t>(k.rgbWidth) * k.rgbHeight * 3);
    const int bpp = view.format == FrameView::Format::Gray8 ? 1 : (view.format == FrameView::Format::RGB8 ? 3 : 4);
    const double sx = double(view.width) / k.rgbWidth, sy = double(view.height) / k.rgbHeight;
    for (int y = 0; y < k.rgbHeight; ++y) {
        const int y0 = static_cast<int>(y * sy), y1 = std::max(y0 + 1, std::min(view.height, static_cast<int>((y + 1) * sy)));
        for (int x = 0; x < k.rgbWidth; ++x) {
            const int x0 = static_cast<int>(x * sx), x1 = std::max(x0 + 1, std::min(view.width, static_cast<int>((x + 1) * sx)));
            int sum[3] = {0, 0, 0}, n = 0;
            for (int yy = y0; yy < y1; yy += 2) {  // every other row/column: enough for a color
                const std::uint8_t* row = view.data + static_cast<std::size_t>(yy) * view.stride;
                for (int xx = x0; xx < x1; xx += 2) {
                    const std::uint8_t* p = row + static_cast<std::size_t>(xx) * bpp;
                    for (int c = 0; c < 3; ++c) sum[c] += p[bpp == 1 ? 0 : c];
                    ++n;
                }
            }
            std::uint8_t* out = k.rgb.data() + (static_cast<std::size_t>(y) * k.rgbWidth + x) * 3;
            for (int c = 0; c < 3; ++c) out[c] = static_cast<std::uint8_t>(sum[c] / std::max(n, 1));
        }
    }
    return k;
}

// Pinhole camera with radial distortion at one pyramid level of the images.
struct LevelCamera {
    double fx = 1, fy = 1, cx = 0, cy = 0, k1 = 0;

    // Undistorted normalized ray (z = 1) of a pixel.
    Vec3 ray(double u, double v) const {
        const double xd = (u - cx) / fx, yd = (v - cy) / fy;
        double x = xd, y = yd;
        if (k1 != 0) {
            for (int i = 0; i < 8; ++i) {
                const double d = 1 + k1 * (x * x + y * y);
                x = xd / d;
                y = yd / d;
            }
        }
        return {x, y, 1.0};
    }
    // Pixel of a camera-space point (z > 0).
    void project(const Vec3& p, double& u, double& v) const {
        const double x = p.x / p.z, y = p.y / p.z;
        const double d = 1 + k1 * (x * x + y * y);
        u = fx * x * d + cx;
        v = fy * y * d + cy;
    }
};

// Camera of the processed image scaled to level-0 image pixels, then to `level`.
LevelCamera levelCamera(const CameraIntrinsics& K, double processedToLevel0, int level) {
    const double s = processedToLevel0 / double(1 << level);
    LevelCamera c;
    c.fx = K.fx * s;
    c.fy = K.fy * s;
    c.cx = (K.cx + 0.5) * s - 0.5;
    c.cy = (K.cy + 0.5) * s - 0.5;
    c.k1 = K.k1;
    return c;
}

float sampleBilinear(const Image& img, double u, double v) {
    const int x0 = static_cast<int>(std::floor(u)), y0 = static_cast<int>(std::floor(v));
    const float fx = static_cast<float>(u - x0), fy = static_cast<float>(v - y0);
    const std::uint8_t* r0 = img.row(y0);
    const std::uint8_t* r1 = img.row(y0 + 1);
    const float top = r0[x0] + fx * (r0[x0 + 1] - r0[x0]);
    const float bottom = r1[x0] + fx * (r1[x0 + 1] - r1[x0]);
    return top + fy * (bottom - top);
}

}  // namespace

struct DenseMapper::Impl {
    DenseParams params;
    std::unique_ptr<ThreadPool> pool;

    // Shared with the calling threads.
    mutable std::mutex mutex;
    std::condition_variable wake;
    bool stop = false;
    bool fullSpeed = false;
    std::deque<Pending> pending;
    std::shared_ptr<const Snapshot> nextSnapshot;
    std::shared_ptr<const std::vector<Point>> cloud = std::make_shared<std::vector<Point>>();
    std::shared_ptr<const std::vector<std::uint8_t>> hidden;
    Stats stats;
    Clock::time_point lastUpdate{};

    // Solver thread only.
    std::shared_ptr<const Snapshot> snap;
    std::map<long, KeyImage> images;
    std::vector<Node> nodes;
    std::unordered_map<std::uint64_t, int> nodeIndex;
    double baseCell = 0;     // size of a level-0 cell (map units); 0 = not set yet
    long mapFirstFrame = -1; // first keyframe of the map the nodes belong to
    std::unordered_map<long, int> keyframeOf;  // frame -> snapshot keyframe
    bool cloudDirty = false;
    Clock::time_point lastPublish{};
    int solves = 0;
    bool idle = false;
    double lastSolveMs = 0;

    // Display: the solve in progress and the recently finished ones.
    DenseMapper::Activity current;
    std::deque<std::pair<DenseMapper::Activity::Box, Clock::time_point>> recent;
    void publishActivity() {
        auto a = std::make_shared<DenseMapper::Activity>(current);
        const Clock::time_point now = Clock::now();
        for (const auto& [box, at] : recent) {
            a->recent.push_back(box);
            a->recent.back().age = std::chrono::duration<double>(now - at).count();
        }
        std::lock_guard<std::mutex> lock(mutex);
        activity = std::move(a);
    }
    std::shared_ptr<const DenseMapper::Activity> activity = std::make_shared<DenseMapper::Activity>();

    std::thread thread;

    int findNode(int level, int x, int y, int z) const {
        const auto it = nodeIndex.find(nodeKey(level, x, y, z));
        return it == nodeIndex.end() ? -1 : it->second;
    }
    int node(int level, int x, int y, int z) {
        const auto [it, inserted] = nodeIndex.try_emplace(nodeKey(level, x, y, z), static_cast<int>(nodes.size()));
        if (inserted) {
            Node n;
            n.level = level;
            n.x = x;
            n.y = y;
            n.z = z;
            nodes.push_back(std::move(n));
        }
        return it->second;
    }
    double cellSize(int level) const { return baseCell * double(1 << level); }
    double processedToLevel0(const KeyImage& img) const { return snap->processScale * img.fromInput; }

    void run();
    void takeSnapshot(std::shared_ptr<const Snapshot> s);
    int chooseNode(int& mode) const;
    void solve(int nodeId, int mode);
    void publish();
    Vec3 worldPoint(const DensePoint& p, const Pose& refPose) const;
};

DenseMapper::DenseMapper(const DenseParams& params) : impl_(std::make_unique<Impl>()) {
    impl_->params = params;
    const int threads = params.threads > 0 ? params.threads : std::max(1, ThreadPool::hardwareThreads() / 2);
    if (threads > 1) impl_->pool = std::make_unique<ThreadPool>(threads, true);
    impl_->thread = std::thread([this] { impl_->run(); });
}

DenseMapper::~DenseMapper() {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stop = true;
    }
    impl_->wake.notify_all();
    impl_->thread.join();
}

void DenseMapper::addKeyframeImage(long frameIndex, std::vector<std::uint8_t>&& pixels, const FrameView& view) {
    Pending p;
    p.frame = frameIndex;
    p.pixels = std::move(pixels);  // the buffer (and `view.data` into it) moves along
    p.view = view;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->pending.push_back(std::move(p));
    }
    impl_->wake.notify_all();
}

void DenseMapper::updateMap(const Reconstructor& r, bool force) {
    const Clock::time_point now = Clock::now();
    if (!force && now - impl_->lastUpdate < std::chrono::milliseconds(200)) return;
    if (r.keyframes().size() < 2 || r.intrinsics().fx <= 0) return;
    impl_->lastUpdate = now;

    auto s = std::make_shared<Snapshot>();
    s->K = r.intrinsics();
    s->processScale = r.processScale();
    const std::vector<MapPoint>& mapPoints = r.mapPoints();
    std::vector<int> compact(mapPoints.size(), -1);
    s->points.reserve(mapPoints.size());
    for (std::size_t i = 0; i < mapPoints.size(); ++i) {
        if (mapPoints[i].bad || mapPoints[i].observations.size() < 2) continue;
        compact[i] = static_cast<int>(s->points.size());
        s->points.push_back(mapPoints[i].position);
        s->mapId.push_back(static_cast<int>(i));
    }
    s->mapPoints = mapPoints.size();
    s->keyframes.reserve(r.keyframes().size());
    for (const Keyframe& k : r.keyframes()) {
        Snapshot::Keyframe sk;
        sk.frame = k.frameIndex;
        sk.pose = k.pose;
        sk.center = k.pose.center();
        sk.points.reserve(k.pointIds.size());
        for (int id : k.pointIds) {
            if (id >= 0 && compact[id] >= 0) sk.points.push_back(compact[id]);
        }
        s->keyframes.push_back(std::move(sk));
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->nextSnapshot = std::move(s);
    }
    impl_->wake.notify_all();
}

void DenseMapper::setFullSpeed(bool full) {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->fullSpeed = full;
    }
    impl_->wake.notify_all();
}

std::shared_ptr<const std::vector<Point>> DenseMapper::cloud() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->cloud;
}

std::shared_ptr<const std::vector<std::uint8_t>> DenseMapper::hiddenSparse() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->hidden;
}

DenseMapper::Stats DenseMapper::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->stats;
}

void DenseMapper::Impl::run() {
    lowerCurrentThreadPriority();
    while (true) {
        std::deque<Pending> newImages;
        std::shared_ptr<const Snapshot> newSnap;
        bool full;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stop) return;
            full = fullSpeed;
            newImages.swap(pending);
            newSnap = std::move(nextSnapshot);
        }
        for (Pending& p : newImages) images[p.frame] = makeKeyImage(p.view, params.imageSize);
        while (static_cast<int>(images.size()) > std::max(params.maxImages, 3)) images.erase(images.begin());
        if (newSnap) takeSnapshot(std::move(newSnap));
        // Only this thread uses the pool, so it can be swapped here.
        const int threads = full ? ThreadPool::hardwareThreads()
                                 : (params.threads > 0 ? params.threads : std::max(1, ThreadPool::hardwareThreads() / 2));
        if ((pool ? pool->size() : 1) != threads) pool = threads > 1 ? std::make_unique<ThreadPool>(threads, true) : nullptr;

        int mode = kFirst;
        const int n = snap ? chooseNode(mode) : -1;
        if ((n < 0) != idle) {
            idle = n < 0;
            std::lock_guard<std::mutex> lock(mutex);
            stats.idle = idle;
        }
        if (n >= 0) {
            const Clock::time_point start = Clock::now();
            solve(n, mode);
            if (current.solving) {  // gave up part way
                current.solving = false;
                publishActivity();
            }
            lastSolveMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            ++solves;
        }
        if (cloudDirty && (n < 0 || Clock::now() - lastPublish > std::chrono::milliseconds(500))) publish();
        if (n < 0) {
            std::unique_lock<std::mutex> lock(mutex);
            wake.wait_for(lock, std::chrono::milliseconds(200),
                          [&] { return stop || !pending.empty() || nextSnapshot != nullptr; });
        }
    }
}

void DenseMapper::Impl::takeSnapshot(std::shared_ptr<const Snapshot> s) {
    snap = std::move(s);
    const Snapshot& S = *snap;
    if (S.keyframes.front().frame != mapFirstFrame) {
        // A new map (tracking was lost): its coordinates have nothing to do
        // with the old one.
        mapFirstFrame = S.keyframes.front().frame;
        nodes.clear();
        nodeIndex.clear();
        baseCell = 0;
        cloudDirty = true;
        while (!images.empty() && images.begin()->first < mapFirstFrame) images.erase(images.begin());
    }
    keyframeOf.clear();
    for (int k = 0; k < static_cast<int>(S.keyframes.size()); ++k) keyframeOf[S.keyframes[k].frame] = k;
    cloudDirty = true;  // poses moved
    if (S.points.empty()) return;

    if (baseCell == 0) {
        std::vector<double> d;
        const Vec3 c = S.keyframes.front().center;
        for (std::size_t i = 0; i < S.points.size(); i += std::max<std::size_t>(1, S.points.size() / 1000)) {
            d.push_back(norm(S.points[i] - c));
        }
        std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
        baseCell = std::max(d[d.size() / 2], 1e-9) / 16.0;
    }

    for (Node& n : nodes) {
        n.sparse = 0;
        n.observers.clear();
        n.newest = -1;
    }
    const int levels = std::clamp(params.levels, 1, 12);
    std::vector<int> nodeOf(S.points.size() * levels);
    for (std::size_t i = 0; i < S.points.size(); ++i) {
        const Vec3& p = S.points[i];
        for (int l = 0; l < levels; ++l) {
            const double cell = cellSize(l);
            const int id = node(l, cellIndex(p.x, cell), cellIndex(p.y, cell), cellIndex(p.z, cell));
            nodeOf[i * levels + l] = id;
            ++nodes[id].sparse;
        }
    }
    std::vector<int> ids;
    for (int k = 0; k < static_cast<int>(S.keyframes.size()); ++k) {
        ids.clear();
        for (int p : S.keyframes[k].points) {
            for (int l = 0; l < levels; ++l) ids.push_back(nodeOf[static_cast<std::size_t>(p) * levels + l]);
        }
        std::sort(ids.begin(), ids.end());
        for (std::size_t i = 0; i < ids.size();) {
            std::size_t j = i;
            while (j < ids.size() && ids[j] == ids[i]) ++j;
            Node& n = nodes[ids[i]];
            n.observers.push_back({k, static_cast<int>(j - i)});
            n.newest = k;
            i = j;
        }
    }
}

// Most urgent cell: never solved ones first (coarse before fine, recently
// seen before old), then cells with many new observing keyframes.
int DenseMapper::Impl::chooseNode(int& mode) const {
    const Snapshot& S = *snap;
    const int latest = static_cast<int>(S.keyframes.size()) - 1;
    const int top = std::clamp(params.levels, 1, 12) - 1;
    int best = -1;
    double bestScore = 0;
    int bestMode = kFirst;
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        const Node& n = nodes[i];
        if (n.sparse < params.minSparsePoints || n.observers.size() < 3) continue;
        const int observers = static_cast<int>(n.observers.size());
        const bool fresh = observers > n.triedObservers;  // new observing keyframes since the last attempt
        if (n.level < top) {
            const int parent = findNode(n.level + 1, n.x >> 1, n.y >> 1, n.z >> 1);
            if (parent < 0 || !nodes[parent].solved || nodes[parent].fine) continue;
        }
        const double recency = std::exp(-(latest - n.newest) / 15.0);
        double score;
        int m = kFirst;
        const int added = observers - n.solvedObservers;
        const double gain = double(added) / std::max(n.solvedObservers, 1);
        if (!n.solved) {
            if (!fresh) continue;
            score = 3 + 2 * recency + 0.1 * n.level;
        } else if (fresh && added >= 2 && gain >= params.resolveGain) {
            score = 1 + recency + 0.5 * std::min(gain, 2.0);
            m = kNewViews;
        } else if (n.sparse > n.triedSparse) {  // new map points in an already solved cell
            score = 0.9 + recency;
            m = kNewViews;
        } else if (!params.refine) {
            continue;
        } else if (n.tier < kMaxTier) {  // idle: the weakest cells one tier better
            score = 0.6 - 0.08 * n.tier + 0.2 * recency + 0.01 * n.level;
            m = kUpgrade;
        } else if (!n.exhausted) {  // top tier: references not used yet
            score = 0.1 + 0.1 * recency - 0.001 * n.refsUsed.size();
            m = kRotate;
        } else {
            continue;
        }
        if (score > bestScore) {
            bestScore = score;
            best = i;
            bestMode = m;
        }
    }
    mode = bestMode;
    return best;
}

Vec3 DenseMapper::Impl::worldPoint(const DensePoint& p, const Pose& refPose) const {
    const Vec2 u = snap->K.undistortPixel(p.u, p.v);
    const Vec3 c{(u.x - snap->K.cx) / snap->K.fx * p.z, (u.y - snap->K.cy) / snap->K.fy * p.z, p.z};
    return refPose.R.transposed() * (c - refPose.t);
}

void DenseMapper::Impl::solve(int nodeId, int mode) {
    const Snapshot& S = *snap;
    Node& n = nodes[nodeId];
    n.triedObservers = static_cast<int>(n.observers.size());
    n.triedSparse = n.sparse;
    n.solved = true;  // even if this attempt fails: its children may still work
    const int tier = mode == kFirst ? 0 : mode == kNewViews ? std::max(n.tier, 0)
                   : mode == kUpgrade ? std::min(n.tier + 1, kMaxTier) : kMaxTier;
    n.tier = std::max(n.tier, tier);  // also when this attempt fails: no retry loop
    const Tier& T = kTiers[tier];
    const bool newReference = mode == kRotate || (mode == kUpgrade && static_cast<int>(n.batches.size()) < T.references);

    const double cell = cellSize(n.level);
    const Vec3 lo{n.x * cell, n.y * cell, n.z * cell};
    const Vec3 hi = lo + Vec3{cell, cell, cell};
    const Vec3 center = (lo + hi) * 0.5;

    // Observers with an image, most shared map points first.
    std::vector<std::pair<int, int>> obs;
    for (const auto& [k, count] : n.observers) {
        if (images.count(S.keyframes[k].frame)) obs.push_back({count, k});
    }
    std::sort(obs.rbegin(), obs.rend());
    if (obs.size() < 3) return;

    // Reference: a new one (as different a direction from the existing ones
    // as possible among the well-observing ones), or the best existing one
    // again with the views added since.
    int ref = -1;
    const auto dirOf = [&](int k) { return normalized(center - S.keyframes[k].center); };
    if (n.batches.empty() || newReference) {
        double bestAngle = -1;
        const int candidates = mode == kRotate ? static_cast<int>(obs.size()) : std::max(1, static_cast<int>(obs.size()) / 2);
        for (int i = 0; i < candidates; ++i) {
            const int k = obs[i].second;
            const long frame = S.keyframes[k].frame;
            if (std::any_of(n.batches.begin(), n.batches.end(), [&](const Batch& b) { return b.ref == frame; })) continue;
            if (std::find(n.refsUsed.begin(), n.refsUsed.end(), frame) != n.refsUsed.end()) continue;
            double minAngle = kPi;
            for (const Batch& b : n.batches) {
                const auto it = keyframeOf.find(b.ref);
                if (it != keyframeOf.end()) minAngle = std::min(minAngle, std::acos(std::clamp(dot(dirOf(k), dirOf(it->second)), -1.0, 1.0)));
            }
            if (minAngle > bestAngle + 1e-9) {
                bestAngle = minAngle;
                ref = k;
            }
            if (n.batches.empty()) break;  // the best observer
        }
    }
    if (ref < 0 && mode == kRotate) {
        n.exhausted = true;
        return;
    }
    if (ref < 0) {  // again: the weakest result whose camera image is still there
        int weakest = std::numeric_limits<int>::max();
        for (const Batch& b : n.batches) {
            const auto it = keyframeOf.find(b.ref);
            if (it != keyframeOf.end() && images.count(b.ref) && b.tier < weakest) {
                weakest = b.tier;
                ref = it->second;
            }
        }
    }
    if (ref < 0) return;
    n.refsUsed.push_back(S.keyframes[ref].frame);
    if (mode == kRotate && static_cast<int>(n.batches.size()) >= T.references) {  // make room: drop the oldest
        n.batches.erase(std::min_element(n.batches.begin(), n.batches.end(),
                                         [](const Batch& a, const Batch& b) { return a.at < b.at; }));
    }
    const Snapshot::Keyframe& refKf = S.keyframes[ref];
    KeyImage& refImg = images.at(refKf.frame);
    const double refDist = norm(center - refKf.center);

    // Neighbours: seen at a usable angle and a similar distance.
    struct View {
        const KeyImage* img;
        Mat3 R;  // reference camera -> this camera
        Vec3 t;
    };
    std::vector<View> views;
    const double minA = params.minAngleDeg * kPi / 180, maxA = params.maxAngleDeg * kPi / 180;
    for (const auto& [count, k] : obs) {
        if (k == ref) continue;
        const double angle = std::acos(std::clamp(dot(dirOf(k), dirOf(ref)), -1.0, 1.0));
        const double ratio = norm(center - S.keyframes[k].center) / std::max(refDist, 1e-12);
        if (angle < minA || angle > maxA || ratio < 0.5 || ratio > 2.0) continue;
        View v;
        v.img = &images.at(S.keyframes[k].frame);
        v.R = S.keyframes[k].pose.R * refKf.pose.R.transposed();
        v.t = S.keyframes[k].pose.t - v.R * refKf.pose.t;
        views.push_back(v);
        if (static_cast<int>(views.size()) >= std::min(T.neighbours, 16)) break;
    }
    if (views.size() < 2) return;
    current.solving = true;
    current.current = {lo, hi, n.level, 0, 0};
    current.reference = refKf.center;
    current.views.clear();
    for (const auto& [count, k] : obs) {
        if (k == ref) continue;
        for (const View& v : views)
            if (v.img == &images.at(S.keyframes[k].frame)) current.views.push_back(S.keyframes[k].center);
    }
    publishActivity();

    // Image region and depth range of the cell in the reference.
    const double toLevel0 = processedToLevel0(refImg);
    const LevelCamera cam0 = levelCamera(S.K, toLevel0, 0);
    const int w0 = refImg.gray[0].width, h0 = refImg.gray[0].height;
    const double nearZ = 0.05 * refDist;
    double zMin = std::numeric_limits<double>::max(), zMax = 0;
    double u0 = w0, u1 = 0, v0 = h0, v1 = 0;
    bool behind = false;
    for (int c = 0; c < 8; ++c) {
        const Vec3 corner{(c & 1) ? hi.x : lo.x, (c & 2) ? hi.y : lo.y, (c & 4) ? hi.z : lo.z};
        const Vec3 pc = refKf.pose.transform(corner);
        zMin = std::min(zMin, pc.z);
        zMax = std::max(zMax, pc.z);
        if (pc.z <= nearZ) {
            behind = true;
            continue;
        }
        double u, v;
        cam0.project(pc, u, v);
        u0 = std::min(u0, u);
        u1 = std::max(u1, u);
        v0 = std::min(v0, v);
        v1 = std::max(v1, v);
    }
    if (zMax <= nearZ) return;
    zMin = std::max(zMin, nearZ);
    if (behind) {
        u0 = 0;
        v0 = 0;
        u1 = w0;
        v1 = h0;
    }
    u0 = std::max(u0, 0.0);
    v0 = std::max(v0, 0.0);
    u1 = std::min(u1, double(w0 - 1));
    v1 = std::min(v1, double(h0 - 1));
    if (u1 <= u0 || v1 <= v0) return;

    // Pyramid level: the cell's region at most maxRegionPx across.
    int level = 0;
    while (level < kPyramid - 1 && std::max(u1 - u0, v1 - v0) / double(1 << level) > params.maxRegionPx * T.regionScale) ++level;
    const double scale = 1.0 / double(1 << level);
    const LevelCamera cam = levelCamera(S.K, toLevel0, level);
    const Image& refGray = refImg.gray[level];
    const int margin = kPatchRadius + 1;
    const int x0 = std::max(margin, static_cast<int>(std::floor((u0 + 0.5) * scale - 0.5)));
    const int x1 = std::min(refGray.width - 1 - margin, static_cast<int>(std::ceil((u1 + 0.5) * scale - 0.5)));
    const int y0 = std::max(margin, static_cast<int>(std::floor((v0 + 0.5) * scale - 0.5)));
    const int y1 = std::min(refGray.height - 1 - margin, static_cast<int>(std::ceil((v1 + 0.5) * scale - 0.5)));

    n.solved = true;
    n.fine = level == 0;
    n.solvedObservers = static_cast<int>(n.observers.size());
    if (x1 <= x0 || y1 <= y0) return;

    const int step = T.step;
    const int gw = (x1 - x0) / step + 1, gh = (y1 - y0) / step + 1;
    const int D = T.depths;
    const double invNear = 1.0 / zMin, invFar = 1.0 / zMax;
    std::vector<float> depth(static_cast<std::size_t>(gw) * gh, 0.0f);  // 0 = rejected
    const float minNcc = static_cast<float>(params.minNcc);
    const float minStd = static_cast<float>(params.minPatchStd);
    const double margin2 = 0.1 * cell;
    const int needAgree = std::min(std::max(params.minAgreeing, 2), static_cast<int>(views.size()));

    parallelFor(pool.get(), 0, gh, 1, [&](int rowBegin, int rowEnd) {
        std::vector<float> ncc(views.size());
        std::vector<float> agg(D);
        for (int gy = rowBegin; gy < rowEnd; ++gy) {
            const int y = y0 + gy * step;
            for (int gx = 0; gx < gw; ++gx) {
                const int x = x0 + gx * step;
                if (refImg.isSky(static_cast<int>((x + 0.5) * (1 << level) / 4), static_cast<int>((y + 0.5) * (1 << level) / 4))) continue;
                // Reference patch, normalized.
                float patch[kPatchSize];
                float mean = 0;
                for (int i = 0; i < kPatchSize; ++i) {
                    const int dx = (i % kPatch - kPatch / 2) * kPatchSpacing, dy = (i / kPatch - kPatch / 2) * kPatchSpacing;
                    patch[i] = refGray.at(x + dx, y + dy);
                    mean += patch[i];
                }
                mean /= kPatchSize;
                float var = 0;
                for (float& p : patch) {
                    p -= mean;
                    var += p * p;
                }
                if (var < minStd * minStd * kPatchSize) continue;
                const float invNorm = 1.0f / std::sqrt(var);
                for (float& p : patch) p *= invNorm;

                const Vec3 r0 = cam.ray(x, y), rx = cam.ray(x + 1, y), ry = cam.ray(x, y + 1);
                Vec3 a0[16], ax[16], ay[16];
                for (std::size_t j = 0; j < views.size(); ++j) {
                    a0[j] = views[j].R * r0;
                    ax[j] = views[j].R * rx;
                    ay[j] = views[j].R * ry;
                }
                // NCC of every view at inverse depth `inv` into `out`; returns
                // the mean of the two best (occlusion in the others).
                const auto evaluate = [&](double inv, float* out) {
                    const double z = 1.0 / inv;
                    for (std::size_t j = 0; j < views.size(); ++j) {
                        const View& view = views[j];
                        const Image& img = view.img->gray[level];
                        out[j] = -1;
                        const Vec3 p0 = a0[j] * z + view.t, px = ax[j] * z + view.t, py = ay[j] * z + view.t;
                        if (p0.z <= 0 || px.z <= 0 || py.z <= 0) continue;
                        double qu, qv, qxu, qxv, qyu, qyv;
                        cam.project(p0, qu, qv);
                        cam.project(px, qxu, qxv);
                        cam.project(py, qyu, qyv);
                        const double axu = qxu - qu, axv = qxv - qv, ayu = qyu - qu, ayv = qyv - qv;
                        const double ex = kPatchRadius * (std::abs(axu) + std::abs(ayu));
                        const double ey = kPatchRadius * (std::abs(axv) + std::abs(ayv));
                        if (qu - ex < 0 || qv - ey < 0 || qu + ex >= img.width - 1 || qv + ey >= img.height - 1) continue;
                        float smp[kPatchSize];
                        float sm = 0;
                        for (int i = 0; i < kPatchSize; ++i) {
                            const double dx = (i % kPatch - kPatch / 2) * kPatchSpacing;
                            const double dy = (i / kPatch - kPatch / 2) * kPatchSpacing;
                            smp[i] = sampleBilinear(img, qu + axu * dx + ayu * dy, qv + axv * dx + ayv * dy);
                            sm += smp[i];
                        }
                        sm /= kPatchSize;
                        float dotp = 0, sv = 0;
                        for (int i = 0; i < kPatchSize; ++i) {
                            const float d = smp[i] - sm;
                            dotp += d * patch[i];
                            sv += d * d;
                        }
                        if (sv > 1e-6f) out[j] = dotp / std::sqrt(sv);
                    }
                    float b1 = -1, b2 = -1;
                    for (std::size_t j = 0; j < views.size(); ++j) {
                        if (out[j] > b1) {
                            b2 = b1;
                            b1 = out[j];
                        } else if (out[j] > b2) {
                            b2 = out[j];
                        }
                    }
                    return 0.5f * (b1 + b2);
                };
                const auto invAt = [&](double k) { return invFar + (invNear - invFar) * k / (D - 1); };
                int bestK = -1;
                float bestScore = -2;
                for (int k = 0; k < D; ++k) {
                    agg[k] = evaluate(invAt(k), ncc.data());
                    if (agg[k] > bestScore) {
                        bestScore = agg[k];
                        bestK = k;
                    }
                }
                if (bestK <= 0 || bestK >= D - 1 || bestScore < minNcc - 0.1f) continue;
                // Finer search between the neighbouring samples.
                const int kFine = T.fineSteps;
                float fine[18];
                double fineInv[18];
                int bestF = 0;
                for (int f = 0; f < kFine + 2; ++f) {
                    fineInv[f] = invAt(bestK - 1 + 2.0 * f / (kFine + 1));
                    fine[f] = f == 0 ? agg[bestK - 1] : (f == kFine + 1 ? agg[bestK + 1] : evaluate(fineInv[f], ncc.data()));
                    if (fine[f] > fine[bestF]) bestF = f;
                }
                if (bestF == 0 || bestF == kFine + 1) continue;
                const float am = fine[bestF - 1], ac = fine[bestF], ap = fine[bestF + 1];
                const float denom = am - 2 * ac + ap;
                const double offset = denom < 0 ? std::clamp(0.5f * (am - ap) / denom, -0.5f, 0.5f) : 0.0;
                const double inv = fineInv[bestF] + offset * (fineInv[bestF + 1] - fineInv[bestF]);
                // Agreement at the chosen depth.
                if (evaluate(inv, ncc.data()) < minNcc) continue;
                int agreeing = 0;
                for (std::size_t j = 0; j < views.size(); ++j) agreeing += ncc[j] >= minNcc - 0.1f;
                if (agreeing < needAgree) continue;
                const double z = 1.0 / inv;
                const Vec3 world = refKf.pose.R.transposed() * (r0 * z - refKf.pose.t);
                if (world.x < lo.x - margin2 || world.y < lo.y - margin2 || world.z < lo.z - margin2 ||
                    world.x > hi.x + margin2 || world.y > hi.y + margin2 || world.z > hi.z + margin2) {
                    continue;
                }
                depth[static_cast<std::size_t>(gy) * gw + gx] = static_cast<float>(z);
            }
        }
    });

    // Keep depths that agree with at least two of their grid neighbours
    // (isolated ones are mostly wrong matches), and that no solved child
    // cell covers (its finer result replaces this one there).
    Batch batch;
    batch.ref = refKf.frame;
    batch.tier = tier;
    batch.at = Clock::now();
    const double toProcessed = 1.0 / toLevel0;
    const double cellBelow = n.level > 0 ? cellSize(n.level - 1) : 0;
    for (int gy = 0; gy < gh; ++gy) {
        for (int gx = 0; gx < gw; ++gx) {
            const float z = depth[static_cast<std::size_t>(gy) * gw + gx];
            if (z <= 0) continue;
            int agree = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if ((dx || dy) && gx + dx >= 0 && gx + dx < gw && gy + dy >= 0 && gy + dy < gh) {
                        const float zn = depth[static_cast<std::size_t>(gy + dy) * gw + gx + dx];
                        if (zn > 0 && std::abs(zn - z) < 0.03f * z) ++agree;
                    }
                }
            }
            if (agree < 2) continue;
            const int x = x0 + gx * step, y = y0 + gy * step;
            const Vec3 world = refKf.pose.R.transposed() * (cam.ray(x, y) * z - refKf.pose.t);
            if (n.level > 0) {
                const int c = findNode(n.level - 1, cellIndex(world.x, cellBelow), cellIndex(world.y, cellBelow),
                                       cellIndex(world.z, cellBelow));
                if (c >= 0 && nodes[c].solved && nodes[c].hasPoints()) continue;
            }
            DensePoint p;
            const double lu = (x + 0.5) / scale - 0.5, lv = (y + 0.5) / scale - 0.5;  // level-0 pixel
            p.u = static_cast<float>((lu + 0.5) * toProcessed - 0.5);
            p.v = static_cast<float>((lv + 0.5) * toProcessed - 0.5);
            p.z = z;
            {  // free space in front of it, seen from the reference
                const int sx = std::clamp(static_cast<int>((lu + 0.5) / 4), 0, refImg.gray[2].width - 1);
                const int sy = std::clamp(static_cast<int>((lv + 0.5) / 4), 0, refImg.gray[2].height - 1);
                float& d = refImg.surface[static_cast<std::size_t>(sy) * refImg.gray[2].width + sx];
                d = d > 0 ? std::min(d, z) : z;
            }
            const int cu = std::clamp(static_cast<int>(std::lround((lu + 0.5) / 2 - 0.5)), 0, refImg.rgbWidth - 1);
            const int cv = std::clamp(static_cast<int>(std::lround((lv + 0.5) / 2 - 0.5)), 0, refImg.rgbHeight - 1);
            const std::uint8_t* c = refImg.rgb.data() + (static_cast<std::size_t>(cv) * refImg.rgbWidth + cu) * 3;
            p.r = c[0];
            p.g = c[1];
            p.b = c[2];
            batch.points.push_back(p);
        }
    }

    // Replace this reference's earlier result; the coarser cells above give
    // up their points inside this one.
    std::erase_if(n.batches, [&](const Batch& b) { return b.ref == batch.ref; });
    const bool any = !batch.points.empty();
    n.batches.push_back(std::move(batch));
    if (any) {
        const int top = std::clamp(params.levels, 1, 12) - 1;
        for (int l = n.level + 1, x = n.x >> 1, y = n.y >> 1, z = n.z >> 1; l <= top; ++l, x >>= 1, y >>= 1, z >>= 1) {
            const int a = findNode(l, x, y, z);
            if (a < 0) continue;
            for (Batch& b : nodes[a].batches) {
                const auto it = keyframeOf.find(b.ref);
                if (it == keyframeOf.end()) continue;
                const Pose& pose = S.keyframes[it->second].pose;
                std::erase_if(b.points, [&](const DensePoint& p) {
                    const Vec3 w = worldPoint(p, pose);
                    return w.x >= lo.x && w.y >= lo.y && w.z >= lo.z && w.x < hi.x && w.y < hi.y && w.z < hi.z;
                });
            }
        }
    }
    cloudDirty = true;
    DenseMapper::Activity::Box done = current.current;
    done.points = static_cast<int>(n.batches.back().points.size());
    recent.push_front({done, Clock::now()});
    if (recent.size() > 40) recent.pop_back();
    current.solving = false;
    publishActivity();
}

std::shared_ptr<const DenseMapper::Activity> DenseMapper::activity() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->activity;
}

void DenseMapper::Impl::publish() {
    cloudDirty = false;
    lastPublish = Clock::now();
    auto out = std::make_shared<std::vector<Point>>();
    Stats st;
    const int topLevel = std::clamp(params.levels, 1, 12) - 1;
    for (const Node& n : nodes) {
        bool needed = n.sparse >= params.minSparsePoints && n.observers.size() >= 3;
        for (int l = n.level + 1, x = n.x >> 1, y = n.y >> 1, z = n.z >> 1; needed && l <= topLevel;
             ++l, x >>= 1, y >>= 1, z >>= 1) {
            const int a = findNode(l, x, y, z);
            needed = !(a >= 0 && nodes[a].solved && nodes[a].fine);  // matched at full resolution above
        }
        if (needed) {
            ++st.regions;
            if (n.solved) ++st.solvedRegions;
        }
        for (const Batch& b : n.batches) {
            const auto it = keyframeOf.find(b.ref);
            if (it == keyframeOf.end()) continue;
            const Pose& pose = snap->keyframes[it->second].pose;
            for (const DensePoint& p : b.points) {
                const Vec3 w = worldPoint(p, pose);
                out->push_back({static_cast<float>(w.x), static_cast<float>(w.y), static_cast<float>(w.z), p.r, p.g, p.b});
            }
        }
    }
    st.points = static_cast<long>(out->size());

    // Map points the dense cloud replaces or removes.
    const Snapshot& S = *snap;
    auto hide = std::make_shared<std::vector<std::uint8_t>>(S.mapPoints, std::uint8_t{0});
    std::vector<int> nodePoints(nodes.size(), 0);
    for (std::size_t i = 0; i < nodes.size(); ++i)
        for (const Batch& b : nodes[i].batches) nodePoints[i] += static_cast<int>(b.points.size());
    struct Surface {
        const KeyImage* img;
        const Pose* pose;
        LevelCamera cam;
    };
    std::vector<Surface> surfaces;
    std::vector<int> skyVotes(S.points.size(), 0), groundVotes(S.points.size(), 0);
    for (const Snapshot::Keyframe& k : S.keyframes) {
        const auto it = images.find(k.frame);
        if (it == images.end()) continue;
        const KeyImage& img = it->second;
        const LevelCamera cam = levelCamera(S.K, processedToLevel0(img), 2);
        if (std::any_of(img.surface.begin(), img.surface.end(), [](float d) { return d > 0; })) {
            surfaces.push_back({&img, &k.pose, cam});
        }
        for (int p : k.points) {
            const Vec3 c = k.pose.transform(S.points[p]);
            if (c.z <= 0) continue;
            double u, v;
            cam.project(c, u, v);
            const int x = static_cast<int>(std::lround(u)), y = static_cast<int>(std::lround(v));
            if (x < 0 || y < 0 || x >= img.gray[2].width || y >= img.gray[2].height) continue;
            ++(img.isSky(x, y) ? skyVotes[p] : groundVotes[p]);
        }
    }
    const int top = std::clamp(params.levels, 1, 12) - 1;
    const float keep = static_cast<float>(1.0 - params.freeSpaceMargin);
    for (std::size_t i = 0; i < S.points.size() && params.hideSparse; ++i) {
        const Vec3& X = S.points[i];
        bool remove = skyVotes[i] > groundVotes[i];
        for (int l = 0; l <= top && !remove; ++l) {  // the deepest solved cell holding it
            const double cell = cellSize(l);
            const int id = findNode(l, cellIndex(X.x, cell), cellIndex(X.y, cell), cellIndex(X.z, cell));
            if (id < 0 || !nodes[id].solved) continue;
            remove = nodePoints[id] >= params.coverPoints;
            break;
        }
        for (std::size_t s2 = 0; s2 < surfaces.size() && !remove; ++s2) {
            const Surface& f = surfaces[s2];
            const Vec3 c = f.pose->transform(X);
            if (c.z <= 0) continue;
            double u, v;
            f.cam.project(c, u, v);
            const int x = static_cast<int>(std::lround(u)), y = static_cast<int>(std::lround(v));
            if (x < 0 || y < 0 || x >= f.img->gray[2].width || y >= f.img->gray[2].height) continue;
            const float d = f.img->surface[static_cast<std::size_t>(y) * f.img->gray[2].width + x];
            remove = d > 0 && c.z < keep * d;
        }
        if (remove) {
            (*hide)[S.mapId[i]] = 1;
            ++st.hiddenSparse;
        }
    }
    st.solves = solves;
    st.images = static_cast<int>(images.size());
    st.lastSolveMs = lastSolveMs;
    st.idle = idle;
    std::lock_guard<std::mutex> lock(mutex);
    cloud = std::move(out);
    hidden = std::move(hide);
    stats = st;
}

}  // namespace vg
