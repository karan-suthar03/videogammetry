#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "videogammetry/frame.h"
#include "videogammetry/image.h"
#include "videogammetry/linalg.h"
#include "videogammetry/point_cloud.h"

namespace vg {

class Reconstructor;
class ThreadPool;

struct DenseParams {
    // Long side of the images matched (pixels), 0 = the size they are given
    // at (the video's own size). Lowered when the images and depth maps of
    // all keyframes would not fit in `maxMemoryMB`.
    int imageSize = 0;
    int maxMemoryMB = 4096;
    // Views matched against each keyframe: the keyframes sharing the most
    // map points with it, weighted by the triangulation angle of those
    // points (best between the bounds).
    int sourceViews = 8;
    double minAngleDeg = 3.0, maxAngleDeg = 40.0;
    // PatchMatch: a depth and a surface normal per pixel, refined coarse to
    // fine over `scales` image levels (each half the size of the next),
    // `iterations` rounds at the coarsest level and `fineIterations` at the
    // others. A hypothesis costs 1 - NCC of an 11x11 patch (every second
    // pixel, weighted by color similarity to the center), averaged over its
    // `bestViews` best matching views.
    int scales = 3;
    int iterations = 6;
    int fineIterations = 3;
    int bestViews = 3;
    // A pixel's depth is kept when its NCC reaches `minNcc` and its patch has
    // texture (weighted standard deviation in gray levels).
    double minNcc = 0.5;
    double minPatchStd = 2.0;
    // Fusion: a pixel becomes a point when at least `minConsistentViews`
    // source views measured the same surface there: depth within
    // `maxDepthDiff` (relative), the round trip back within `maxReprojPx`,
    // normals within `maxNormalDeg`. The point is the average of them all.
    int minConsistentViews = 2;
    double maxDepthDiff = 0.01;
    double maxReprojPx = 2.0;
    double maxNormalDeg = 15.0;
    // Keyframe selection, before anything else: of keyframes that see the
    // scene from nearly the same place (centers closer than
    // `keyframeSpacing` times their median scene depth) and direction
    // (within `keyframeAngleDeg`), only the one seeing the most map points
    // is kept. The others are not decoded or used at all.
    bool selectKeyframes = true;
    double keyframeSpacing = 0.05;
    double keyframeAngleDeg = 10.0;
    int threads = 0;  // 0 = all hardware threads
};

// Pinhole camera (no distortion) with pixel centers at integer coordinates.
struct PinholeCamera {
    double fx = 1, fy = 1, cx = 0, cy = 0;
    // The same camera on an image `factor` times smaller.
    PinholeCamera scaled(double factor) const {
        return {fx / factor, fy / factor, (cx + 0.5) / factor - 0.5, (cy + 0.5) / factor - 0.5};
    }
};

// One camera image for PatchMatch.
struct MvsView {
    Pose pose;  // world -> camera
    Image gray;
};

// Per-pixel result of PatchMatch on one reference view; depth 0 = none.
struct DepthMap {
    int width = 0, height = 0;
    std::vector<float> depth;   // along the camera z axis
    std::vector<Vec3> normal;   // camera frame, facing the camera
    std::vector<float> cost;    // 1 - NCC
};

// Depth and normal of every pixel of `ref` from the `sources` (PatchMatch
// stereo, coarse to fine). `seeds` (world points, e.g. map points the
// reference sees) start the search; depths stay within [minDepth, maxDepth].
// Pixels that fail the NCC or texture tests have depth 0.
DepthMap patchMatch(const MvsView& ref, const std::vector<const MvsView*>& sources, const PinholeCamera& camera,
                    double minDepth, double maxDepth, const std::vector<Vec3>& seeds, const DenseParams& params,
                    ThreadPool* pool, const std::atomic<bool>* cancel = nullptr);

// Dense chunks of points, one per fused keyframe (shared, never changed).
using DenseCloud = std::vector<std::shared_ptr<const std::vector<Point>>>;

// Dense reconstruction of a finished map: a depth map for every keyframe
// (PatchMatch against its best views), then the depth maps that agree are
// fused into one cloud. Keyframes are worked in video order and each one is
// fused as soon as the depth maps of all its views exist, so the cloud grows
// along the camera path while it runs.
//
// Usage: construct from the finished Reconstructor, call imageLongSide()
// with the video size, decode the video at that size and give addImage() the
// frames listed by frames(), then run().
class DenseReconstructor {
public:
    DenseReconstructor(const Reconstructor& reconstructor, const DenseParams& params = {});
    ~DenseReconstructor();
    DenseReconstructor(const DenseReconstructor&) = delete;
    DenseReconstructor& operator=(const DenseReconstructor&) = delete;

    // Video frame indices of the keyframe images needed, ascending.
    const std::vector<long>& frames() const { return frames_; }
    // Long side to decode the frames at, for a video of this size: the
    // params' image size, lowered to fit the memory limit.
    int imageLongSide(int videoWidth, int videoHeight) const;
    // The image of keyframe frame `frameIndex` (RGB8, any size with the
    // aspect of the processed image). It is undistorted and kept.
    void addImage(long frameIndex, const FrameView& rgb);

    // Computes and fuses everything; `onUpdate` is called (on this thread)
    // after every depth map and fusion step. Returns false when cancelled.
    bool run(const std::atomic<bool>& cancel, const std::function<void()>& onUpdate = {});

    struct Progress {
        int keyframes = 0;   // with an image
        int totalKeyframes = 0;  // of the map, before the selection
        int selected = 0;        // kept by the selection
        int references = 0;      // with enough views for a depth map (known once run() starts)
        int depthMaps = 0;   // computed
        int fused = 0;
        long points = 0;
        int width = 0, height = 0;  // matched image size
        double lastDepthMs = 0;
        bool finished = false;
    };
    // Any thread.
    Progress progress() const;
    std::shared_ptr<const DenseCloud> cloud() const;
    // The cloud so far with normals, for saving.
    PointCloud pointCloud() const;

private:
    struct View;
    void computeDepth(int index, ThreadPool& pool, const std::atomic<bool>& cancel);
    void fuse(int index);
    void checkViews();

    DenseParams params_;
    std::vector<View> views_;
    std::vector<long> frames_;
    std::vector<Vec3> mapPoints_;
    double processedWidth_ = 0, processedHeight_ = 0;
    double fxP_ = 0, fyP_ = 0, cxP_ = 0, cyP_ = 0, k1_ = 0;  // processed-image camera
    PinholeCamera camera_;  // of the matched images (undistorted)
    int width_ = 0, height_ = 0;
    std::vector<float> mapX_, mapY_;  // undistortion: source position of each pixel

    mutable std::mutex mutex_;
    Progress progress_;
    std::shared_ptr<const DenseCloud> cloud_;
    std::vector<std::shared_ptr<const std::vector<Vec3>>> normals_;  // per chunk
};

}  // namespace vg
