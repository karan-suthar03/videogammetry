#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "videogammetry/frame.h"
#include "videogammetry/point_cloud.h"

namespace vg {

class Reconstructor;

struct DenseParams {
    // Keyframe images are kept at this long side (pixels) for matching.
    int imageSize = 640;
    // Worker threads of one region solve (they and the solver thread run at
    // low OS priority, on CPU time tracking leaves over).
    int threads = 2;
    // Volume hierarchy: cubic cells, the smallest 1/16 of the median scene
    // depth of the first map snapshot, each level twice the size of the one
    // below. A cell is solved once at least `minSparsePoints` map points lie
    // in it; the coarse levels first (a quick rough cloud), then their
    // children, down to cells that fit `maxRegionPx` pixels of a full-size
    // keyframe image.
    int levels = 6;
    int minSparsePoints = 8;
    int maxRegionPx = 160;
    // Plane sweep: depths tried per pixel (evenly in inverse depth over the
    // cell), every `pixelStep` pixels of the cell's image region.
    int depthSamples = 64;
    int pixelStep = 2;
    // Views matched against the reference: the keyframes sharing the most
    // map points with the cell, seen at an angle within these bounds.
    int neighbours = 5;
    double minAngleDeg = 2.0, maxAngleDeg = 40.0;
    // A depth is kept when the two best views reach this normalized cross
    // correlation (5x5 patch) there; texture-less patches are skipped.
    double minNcc = 0.65;
    double minPatchStd = 3.0;  // gray levels
    // Per cell, solved from up to this many reference keyframes over time
    // (covers what one view cannot see); revisits with enough new views
    // solve a reference again with them.
    int maxReferences = 2;
    double resolveGain = 0.5;  // new observing keyframes, relative to those at the last solve
    // Keyframe images kept at most (the oldest go first).
    int maxImages = 600;
};

// Dense reconstruction in the background: grows a dense colored cloud from
// the keyframes of a running Reconstructor, a region of space at a time,
// independently of tracking (its own threads; tracking never waits on it).
// Regions are re-solved when later keyframes see them again, and every dense
// point is stored relative to the keyframe it was measured in, so it follows
// when bundle adjustment moves that keyframe.
class DenseMapper {
public:
    explicit DenseMapper(const DenseParams& params = {});
    ~DenseMapper();  // stops and joins
    DenseMapper(const DenseMapper&) = delete;
    DenseMapper& operator=(const DenseMapper&) = delete;

    // Tracking thread, both cheap:
    // The image of a new keyframe (`frameIndex` = Keyframe::frameIndex), at
    // the size it was given to the Reconstructor. Takes the pixel buffer
    // `view` points into, so nothing is copied.
    void addKeyframeImage(long frameIndex, std::vector<std::uint8_t>&& pixels, const FrameView& view);
    // Copies the keyframe poses and map points it needs (about a millisecond
    // for a large map). Call after frames that added a keyframe; it skips
    // calls closer than 200 ms apart unless `force`.
    void updateMap(const Reconstructor& reconstructor, bool force = false);

    struct Stats {
        long points = 0;
        int regions = 0;        // cells with enough map points
        int solvedRegions = 0;
        int solves = 0;         // region solves run so far
        int images = 0;
        double lastSolveMs = 0;
    };
    // Any thread: the dense cloud with the latest keyframe poses.
    std::shared_ptr<const std::vector<Point>> cloud() const;
    Stats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vg
