#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "videogammetry/frame.h"
#include "videogammetry/linalg.h"
#include "videogammetry/point_cloud.h"

namespace vg {

class Reconstructor;

struct DenseParams {
    // Keyframe images are kept at this long side (pixels) for matching.
    int imageSize = 640;
    // Worker threads of one region solve, 0 = half the hardware threads
    // (they and the solver thread run at low OS priority, on CPU time
    // tracking leaves over).
    int threads = 0;
    // Volume hierarchy: cubic cells, the smallest 1/16 of the median scene
    // depth of the first map snapshot, each level twice the size of the one
    // below. A cell is solved once at least `minSparsePoints` map points lie
    // in it; the coarse levels first (a quick rough cloud), then their
    // children, down to cells that fit `maxRegionPx` pixels of a full-size
    // keyframe image.
    int levels = 6;
    int minSparsePoints = 8;
    int maxRegionPx = 160;
    // Quality rises over time: every cell is first solved by a very fast pass
    // (every 4th pixel, 16 depths, 3 views, one reference camera); whenever
    // nothing new waits, the weakest cells are solved again one tier better,
    // up to every pixel, 96 depths, 12 views and 4 references (finer images
    // from tier 3). At the top tier, cells keep being solved from reference
    // cameras not used yet until every camera that sees them has been one.
    // Views matched against the reference: the keyframes sharing the most
    // map points with the cell, seen at an angle within these bounds.
    double minAngleDeg = 2.0, maxAngleDeg = 40.0;
    // A depth is kept when the two best views reach this normalized cross
    // correlation (5x5 patch) there and `minAgreeing` views (fewer if the
    // cell has fewer) come within 0.1 of it; texture-less patches and sky
    // are skipped.
    double minNcc = 0.65;
    double minPatchStd = 3.0;  // gray levels
    int minAgreeing = 3;
    // Map points it replaces: those in cells it has solved (with at least
    // `coverPoints` dense points), those in front of a measured dense surface
    // (in free space: floating outliers), and those that most of their
    // keyframes see on sky (smooth bright area connected to the image top).
    int coverPoints = 50;
    double freeSpaceMargin = 0.15;  // relative to the surface depth
    // Revisits with enough new observing keyframes solve a cell again at
    // its tier.
    double resolveGain = 0.5;  // new observing keyframes, relative to those at the last solve
    // Keyframe images kept at most (the oldest go first; about 0.4 MB each at
    // the default size). Regions seen only by dropped ones cannot be solved again.
    int maxImages = 2000;
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

    // All hardware threads from the next solve on (e.g. once tracking has
    // ended and nothing else needs the CPU); still at low OS priority.
    void setFullSpeed(bool full);

    struct Stats {
        long points = 0;
        int regions = 0;        // cells to solve: enough map points and views, not inside a cell solved at full resolution
        int solvedRegions = 0;
        int solves = 0;         // region solves run so far
        int images = 0;
        long hiddenSparse = 0;  // map points it replaces or removes
        bool idle = false;      // nothing left to solve until new keyframes arrive
        double lastSolveMs = 0;
    };
    // Any thread: the dense cloud with the latest keyframe poses.
    std::shared_ptr<const std::vector<Point>> cloud() const;
    Stats stats() const;
    // Per map point (index into Reconstructor::mapPoints() at the last
    // update; later points are not covered): 1 = replaced or removed by the
    // dense solver. May be null.
    std::shared_ptr<const std::vector<std::uint8_t>> hiddenSparse() const;

    // What the solver is doing, for display: the cell being solved, the
    // cameras it matches (reference first), and the cells solved recently.
    struct Activity {
        struct Box {
            Vec3 lo, hi;
            int level = 0;
            int points = 0;    // dense points the solve kept
            double age = 0;    // seconds since it finished
        };
        bool solving = false;
        Box current;
        Vec3 reference;
        std::vector<Vec3> views;
        std::vector<Box> recent;  // newest first
    };
    std::shared_ptr<const Activity> activity() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vg
