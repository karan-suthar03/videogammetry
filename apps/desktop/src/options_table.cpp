#include "options_table.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string_view>

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

std::vector<SettingEntry> optionEntries(vg::Reconstructor::Options& o) {
    vg::OrbParams& orb = o.orb;
    vg::MappingParams& m = o.mapping;
    // clang-format off
    return {
        {"Processing", "processSize", "Frame size (long side)", "px, 0 = full size", &o.processSize, 0, kInf, 80},
        {"Processing", "threads", "Worker threads", "0 = all cores", &o.threads, 0, kInf, 1},
        {"Processing", "fovDeg", "Field of view (start guess)", "deg", &o.fovDeg, 0.001, 179.999, 5},

        {"Features", "maxFeatures", "Features per frame", "", &orb.maxFeatures, 1, kInf, 250},
        {"Features", "levels", "Pyramid levels", "", &orb.levels, 1, kInf, 1},
        {"Features", "scaleFactor", "Pyramid scale factor", "", &orb.scaleFactor, 1.001, kInf, 0.05},
        {"Features", "fastThreshold", "FAST threshold", "0..255", &orb.fastThreshold, 0, 255, 1},
        {"Features", "minFastThreshold", "FAST threshold (retry)", "0..255", &orb.minFastThreshold, 0, 255, 1},
        {"Features", "centerBoost", "Center boost", "0 = spread evenly", &orb.centerBoost, 0, kInf, 0.25},

        {"Initialization", "minInitParallaxDeg", "Min parallax", "deg", &m.minInitParallaxDeg, 0, 180, 0.25},
        {"Initialization", "minInitialPoints", "Min initial points", "", &m.minInitialPoints, 0, kInf, 10},
        {"Initialization", "maxReferenceAge", "Max reference age", "frames", &m.maxReferenceAge, 1, kInf, 5},
        {"Initialization", "twoViewThreshold", "Two-view inlier threshold", "px", &o.twoView.inlierThresholdPx, 0, kInf, 0.25},
        {"Initialization", "twoViewIterations", "Two-view RANSAC iterations", "", &o.twoView.maxIterations, 1, kInf, 100},

        {"Tracking", "localKeyframes", "Local map keyframes", "", &m.localKeyframes, 1, kInf, 5},
        {"Tracking", "covisibleNeighbours", "Covisible neighbours", "", &m.covisibleNeighbours, 0, kInf, 1},
        {"Tracking", "projectionRadiusPx", "Search radius", "px", &m.projectionRadiusPx, 0, kInf, 1},
        {"Tracking", "wideRadiusPx", "Wide search radius", "px", &m.wideRadiusPx, 0, kInf, 5},
        {"Tracking", "minTrackedPoints", "Min tracked points", "", &m.minTrackedPoints, 0, kInf, 5},
        {"Tracking", "maxLostFrames", "Lost frames before a new map", "", &m.maxLostFrames, 0, kInf, 5},
        {"Tracking", "pnpThreshold", "PnP inlier threshold", "px", &o.pnp.inlierThresholdPx, 0, kInf, 0.5},
        {"Tracking", "pnpIterations", "PnP RANSAC iterations", "", &o.pnp.maxIterations, 1, kInf, 50},

        {"Keyframes", "keyframeTrackedRatio", "New keyframe below tracked ratio", "", &m.keyframeTrackedRatio, 0, kInf, 0.05},
        {"Keyframes", "minKeyframeGap", "Min keyframe gap", "frames", &m.minKeyframeGap, 0, kInf, 1},
        {"Keyframes", "maxKeyframeGap", "Max keyframe gap", "frames", &m.maxKeyframeGap, 1, kInf, 5},

        {"Triangulation", "minTriangulationAngleDeg", "Min triangulation angle", "deg", &m.minTriangulationAngleDeg, 0, 180, 0.25},
        {"Triangulation", "maxReprojectionErrorPx", "Max reprojection error", "px", &m.maxReprojectionErrorPx, 0, kInf, 0.5},
        {"Triangulation", "triangulationKeyframes", "Keyframes to triangulate with", "", &m.triangulationKeyframes, 1, kInf, 1},
        {"Triangulation", "triangulationSearch", "Keyframes searched", "", &m.triangulationSearch, 1, kInf, 1},
        {"Triangulation", "minBaselineRatio", "Min baseline / depth", "", &m.minBaselineRatio, 0, kInf, 0.01},
        {"Triangulation", "epipolarDistancePx", "Epipolar distance", "px", &m.epipolarDistancePx, 0, kInf, 0.5},
        {"Triangulation", "probationKeyframes", "Point probation", "keyframes", &m.probationKeyframes, 0, kInf, 1},
        {"Triangulation", "minConfirmingKeyframes", "Keyframes to confirm a point", "", &m.minConfirmingKeyframes, 0, kInf, 1},

        {"Fusion", "fuseNeighbours", "Fusion neighbours", "0 = no fusion", &m.fuseNeighbours, 0, kInf, 1},
        {"Fusion", "fuseSecondOrder", "Second-order neighbours", "", &m.fuseSecondOrder, 0, kInf, 1},
        {"Fusion", "fuseRadiusPx", "Fusion radius", "px", &m.fuseRadiusPx, 0, kInf, 0.5},
        {"Fusion", "fuseMerge", "Merge duplicate points", "", &m.fuseMerge, 0, 1, 1},

        {"Bundle adjustment", "localBAKeyframes", "Local BA keyframes", "0 = off", &m.localBAKeyframes, 0, kInf, 1},
        {"Bundle adjustment", "localBAIterations", "Local BA iterations", "", &m.localBAIterations, 0, kInf, 1},
        {"Bundle adjustment", "globalBAIterations", "Final global BA iterations", "0 = off", &m.globalBAIterations, 0, kInf, 5},
        {"Bundle adjustment", "backgroundBA", "Background global BA", "", &m.backgroundBA, 0, 1, 1},
        {"Bundle adjustment", "backgroundBAIterations", "Background BA iterations", "", &m.backgroundBAIterations, 0, kInf, 1},
        {"Bundle adjustment", "backgroundBAThreads", "Background BA threads", "0 = half the cores", &m.backgroundBAThreads, 0, kInf, 1},

        {"Calibration", "refineFocal", "Refine focal length", "", &m.refineFocal, 0, 1, 1},
        {"Calibration", "refineDistortion", "Refine lens distortion", "", &m.refineDistortion, 0, 1, 1},
        {"Calibration", "focalEveryKeyframes", "Focal refinement every", "keyframes", &m.focalEveryKeyframes, 1, kInf, 1},
        {"Calibration", "focalBAKeyframes", "Focal refinement keyframes", "", &m.focalBAKeyframes, 1, kInf, 1},
    };
    // clang-format on
}

const char* settingDetails(const char* key) {
    struct Details { const char* key; const char* text; };
    // clang-format off
    static constexpr Details kDetails[] = {
        {"realTime",
         "Paces playback to the video's own frame rate, as if it were a live camera: frames the reconstruction "
         "cannot keep up with are processed late, never skipped. Off processes frames as fast as the CPU allows, "
         "which is how the benchmark runs. Can also be toggled while playing with M."},
        {"thumbnailSize",
         "Long side, in pixels, of the video picture shown in the top-right corner with the tracked keypoints drawn "
         "on it. Only the display changes; the reconstruction always sees the frame at its processing size. A value "
         "larger than the video shows the frame at its full size. Very large thumbnails cost time on every frame "
         "for the resize."},
        {"pointSize",
         "Size of each map point on screen, in pixels (tracked points are drawn one pixel larger). Bigger points "
         "make a sparse cloud easier to read but hide detail. Drawing time grows with the square of the size, so "
         "it stops at 100. Also + and - while playing."},
        {"highlightTracked",
         "Draws the map points that were matched in the current frame in green instead of their own color, which "
         "shows what the camera is locked onto right now. T while playing."},
        {"showKeyframes",
         "Draws a small blue camera frustum at every keyframe (the frames the map is built from). Turning it off "
         "uncovers the points behind dense camera paths. K while playing."},

        {"processSize",
         "Frames are shrunk so that their long side is at most this many pixels before features are extracted; 0 "
         "uses the full resolution. Smaller is much faster (cost grows with the pixel count) but sees less detail "
         "and localizes features less precisely; larger finds more, finer features at a much higher cost. Values "
         "under about 40 px leave no room for any feature, so nothing is tracked. Larger than the video changes "
         "nothing."},
        {"threads",
         "Worker threads for feature extraction and matching. 0 uses every hardware thread. 1 runs single-threaded "
         "(slowest, but the easiest to profile). More threads than the CPU has only adds switching overhead, and "
         "every thread costs memory, so tens of thousands use gigabytes."},
        {"fovDeg",
         "The camera is uncalibrated, so the reconstruction starts from this guess of the horizontal field of view "
         "(across the long side of the image) and refines it from the motion when focal refinement is on. A good "
         "guess helps the first frames initialize: phones are about 65-75 deg, wide/action cameras 90-120 deg. "
         "Extreme values (near 0 or 180) describe impossible lenses and initialization will usually fail."},

        {"maxFeatures",
         "How many ORB features (corners with a binary descriptor) are kept per frame, spread over the image and "
         "the pyramid levels. More features give denser clouds and steadier tracking in low texture, but every "
         "stage gets slower (matching, triangulation, bundle adjustment). Very few (under ~100) make tracking "
         "fragile; huge values are limited by how many corners the image actually has."},
        {"levels",
         "Number of image pyramid levels features are detected on; each level is the previous one shrunk by the "
         "scale factor. More levels find the same corner at more distances, so tracking survives moving towards "
         "or away from things. 1 = only the full image. Levels too small to hold a feature are skipped, so very "
         "large values only cost memory and time up to that point."},
        {"scaleFactor",
         "Size ratio between neighbouring pyramid levels (1.2 = each level is 1/1.2 of the one above). Close to 1 "
         "means many nearly identical levels: slower, with features duplicated across levels. Large values leave "
         "gaps in scale where a corner is not found again. Together with the level count it sets the range of "
         "scales covered."},
        {"fastThreshold",
         "FAST corner test: how much brighter or darker than the center (0-255 gray levels) the surrounding pixels "
         "must be. Lower finds more, weaker corners (more noise, slower); higher keeps only strong, repeatable "
         "corners. 0 accepts almost any pixel."},
        {"minFastThreshold",
         "Second, usually lower, FAST threshold used only in grid cells where the main threshold found nothing, so "
         "smooth areas (walls, sky) still get some features. Set it equal to the main threshold to disable the "
         "retry."},
        {"centerBoost",
         "Shifts features towards the middle of the image: a grid cell at the center gets (1 + boost) times the "
         "features of a cell at the edge, falling off smoothly. 0 spreads them evenly. Helps when the edges are "
         "blurry or distorted; very large values put almost everything in the center."},

        {"minInitParallaxDeg",
         "Before there is a map, the first two frames used must see the scene from different enough positions: "
         "this is the minimum median angle between the two viewing rays of a point. Lower starts the map sooner "
         "but with poorly determined depths; higher waits for more camera movement. 0 accepts a camera that only "
         "rotated, which gives a broken map."},
        {"minInitialPoints",
         "Minimum number of points the two-view initialization must triangulate before a map is started. Lower "
         "starts sooner (risking a weak first map); higher waits for a well-textured, well-moved pair."},
        {"maxReferenceAge",
         "During initialization, the first frame of the pair is replaced after this many frames without a "
         "successful start (or when too few of its features are still matched). Small values keep trying fresh "
         "pairs; large values keep waiting for parallax against an old frame."},
        {"twoViewThreshold",
         "Initialization estimates the relative motion of two frames (essential matrix, RANSAC). A match counts as "
         "consistent if its Sampson error is below this many pixels. Lower is stricter (fewer, cleaner matches); "
         "higher tolerates noise and wrong matches."},
        {"twoViewIterations",
         "Maximum RANSAC samples tried when estimating the initial two-view motion. More finds a good solution in "
         "heavily mismatched data at more cost; it stops earlier once confident."},

        {"localKeyframes",
         "Each frame is tracked by projecting map points from nearby keyframes into it. This caps how many "
         "keyframes (chosen by shared points, not time) contribute. More finds the camera again after quick turns "
         "and reuses older parts of the map; fewer is faster."},
        {"covisibleNeighbours",
         "For each keyframe that sees the currently tracked points, up to this many of its best-connected "
         "keyframes are added to the local map. 0 uses only the keyframes seeing the last frame's points."},
        {"projectionRadiusPx",
         "Map points are projected into the new frame where the motion model predicts them; a matching feature "
         "is searched within this radius (pixels in the processed frame). Larger tolerates faster, jerkier motion "
         "but risks wrong matches and costs time; smaller needs smooth motion."},
        {"wideRadiusPx",
         "If the normal radius tracks too few points (fast motion), the search is retried with this wider radius "
         "before falling back to matching against the last keyframe."},
        {"minTrackedPoints",
         "A frame counts as tracked only with at least this many matched map points after pose estimation; below "
         "it the frame is lost. Also drives keyframe insertion (a new keyframe is forced under twice this). Lower "
         "keeps tracking through poor views (possibly drifting); 0 never declares the camera lost."},
        {"maxLostFrames",
         "How many lost frames in a row are tolerated before the current map is abandoned and a new one started "
         "(maps are not merged). 0 restarts at the first lost frame; large values keep trying to relocalize in the "
         "existing map."},
        {"pnpThreshold",
         "Camera pose from 2D-3D matches (PnP, RANSAC): a match is an inlier if it reprojects within this many "
         "pixels. Lower rejects more matches as outliers; higher accepts sloppier ones."},
        {"pnpIterations",
         "Maximum RANSAC iterations for the pose. It stops earlier once confident, so raising it only helps frames "
         "with many wrong matches."},

        {"keyframeTrackedRatio",
         "A new keyframe is taken when the frame tracks fewer than this share of the points the last keyframe "
         "tracked, i.e. when the view has changed enough. Higher (towards 1 and above) makes keyframes very often: "
         "denser map, much more mapping work. Lower makes them rarely (only the gap limits apply)."},
        {"minKeyframeGap",
         "At least this many frames between keyframes, whatever the view change. Large values make the map sparse "
         "and can lose tracking because the map does not grow fast enough."},
        {"maxKeyframeGap",
         "A keyframe is forced after this many frames even if the view barely changed. 1 makes every frame a "
         "keyframe (very slow, very dense)."},

        {"minTriangulationAngleDeg",
         "New points are triangulated between keyframes only if the two viewing rays meet at least at this angle. "
         "Small angles give depths with big errors (points smeared along the ray); larger angles keep only "
         "well-determined points, so the cloud gets sparser."},
        {"maxReprojectionErrorPx",
         "A newly triangulated point is kept only if it reprojects within this many pixels in both keyframes. "
         "Lower gives a cleaner, sparser cloud; higher keeps more, noisier points."},
        {"triangulationKeyframes",
         "Each new keyframe triangulates new points against up to this many earlier keyframes (those sharing the "
         "most points that have enough baseline). More gives more points per keyframe at more mapping cost."},
        {"triangulationSearch",
         "How many of the best-connected earlier keyframes are considered when picking the triangulation partners "
         "above."},
        {"minBaselineRatio",
         "A triangulation partner must be at least this far away, as a share of the median scene depth "
         "(0.02 = 2 cm per metre of depth). Short baselines give poor depth."},
        {"epipolarDistancePx",
         "Matching for triangulation uses the known poses: a candidate must lie within this distance of the "
         "epipolar line (scaled by its pyramid level). Wider finds more matches when poses are slightly off, but "
         "more of them are wrong."},
        {"probationKeyframes",
         "A new point is checked after this many more keyframes have been added: if too few keyframes see it by "
         "then, it is removed as a likely mismatch. 0 checks at once."},
        {"minConfirmingKeyframes",
         "Keyframes that must see a point by the end of its probation to keep it. 2 keeps every successful "
         "triangulation (denser, noisier); 3 or more keeps only points seen again (cleaner)."},

        {"fuseNeighbours",
         "After triangulation, the new keyframe's points are projected into this many best-connected keyframes "
         "(and theirs into it). A projection landing on a matching feature links the two, so new keyframes attach "
         "to old points, duplicates are merged and scale drift shrinks. 0 turns fusion off (faster, more "
         "duplicate points)."},
        {"fuseSecondOrder",
         "Fusion also reaches up to this many best neighbours of each neighbour above, linking further back in the "
         "map."},
        {"fuseRadiusPx",
         "Search radius for fusion, in pixels at the predicted pyramid scale. Larger links more points but risks "
         "merging different ones."},
        {"fuseMerge",
         "When a fused projection lands on a feature that already has its own point, the two points are merged "
         "into one. Off only links points to features that have no point yet."},

        {"localBAKeyframes",
         "After each new keyframe, it and its most connected keyframes (this many in all) are refined together "
         "with their points by bundle adjustment, which keeps the local map consistent. 0 turns local refinement "
         "off: faster, but errors accumulate. Larger windows are more accurate and slower."},
        {"localBAIterations",
         "Gauss-Newton/Levenberg-Marquardt iterations for each local refinement. More iterations converge "
         "further at more mapping time per keyframe; 0 leaves the local map unrefined."},
        {"globalBAIterations",
         "When the video ends, the whole map is refined at once for this many iterations before the cloud is "
         "final (and savable). 0 skips it. Large maps make each iteration slow."},
        {"backgroundBA",
         "While frames keep being tracked, the whole map is re-optimized on a separate low-priority thread and the "
         "result merged in, repairing older parts and refining the focal length and distortion jointly. Results "
         "then depend on timing; off is fully deterministic."},
        {"backgroundBAIterations",
         "Iterations per background pass. More converges each pass further; passes then take longer and are "
         "merged less often."},
        {"backgroundBAThreads",
         "Worker threads for the background passes; 0 uses half the hardware threads. More makes passes faster but "
         "competes with tracking for the CPU."},

        {"refineFocal",
         "Self-calibration: every few keyframes the focal length (field of view) is estimated together with "
         "recent keyframes. The result is kept only if the motion determines it well; straight-ahead walking "
         "often cannot. Off keeps the starting field of view throughout."},
        {"refineDistortion",
         "Estimates radial lens distortion (k1) together with the focal length. Needs focal refinement on. Helps "
         "wide lenses whose straight lines bend near the edges."},
        {"focalEveryKeyframes",
         "Focal refinement runs every this many keyframes. 1 refines at every keyframe (more mapping cost)."},
        {"focalBAKeyframes",
         "Keyframes optimized together in each focal refinement. More gives a better-determined focal length at "
         "more cost."},
    };
    // clang-format on
    for (const Details& d : kDetails) {
        if (std::string_view(d.key) == key) return d.text;
    }
    return "";
}

bool isBool(const SettingEntry& e) { return std::holds_alternative<bool*>(e.value); }

double settingValue(const SettingEntry& e) {
    return std::visit([](auto* p) { return static_cast<double>(*p); }, e.value);
}

double setSettingValue(const SettingEntry& e, double v) {
    if (std::isnan(v)) return settingValue(e);
    v = std::clamp(v, e.min, e.max);
    std::visit(
        [&](auto* p) {
            using T = std::remove_pointer_t<decltype(p)>;
            if constexpr (std::is_same_v<T, bool>) {
                *p = v >= 0.5;
            } else if constexpr (std::is_same_v<T, int>) {
                constexpr double lo = std::numeric_limits<int>::min(), hi = std::numeric_limits<int>::max();
                *p = static_cast<int>(std::clamp(std::round(v), lo, hi));
            } else if constexpr (std::is_same_v<T, float>) {
                *p = static_cast<float>(std::min(v, double(std::numeric_limits<float>::max())));
            } else {
                *p = v;
            }
        },
        e.value);
    return settingValue(e);
}

std::string formatSettingValue(const SettingEntry& e) {
    if (isBool(e)) return *std::get<bool*>(e.value) ? "on" : "off";
    char buf[64];
    if (std::holds_alternative<int*>(e.value)) {
        std::snprintf(buf, sizeof(buf), "%d", *std::get<int*>(e.value));
    } else {
        std::snprintf(buf, sizeof(buf), "%.6g", settingValue(e));
    }
    return buf;
}

bool parseSettingValue(const std::string& text, double& v) {
    const char* begin = text.c_str();
    char* end = nullptr;
    v = std::strtod(begin, &end);
    if (end == begin) return false;
    while (*end == ' ') ++end;
    return *end == '\0';
}

bool applySettingLine(const std::string& line, const std::vector<SettingEntry>& entries, std::string* key,
                      std::string* value) {
    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) {
        if (key) key->clear();
        if (value) value->clear();
        return false;
    }
    std::string k = line.substr(0, eq), v = line.substr(eq + 1);
    if (!v.empty() && v.back() == '\r') v.pop_back();
    if (key) *key = k;
    if (value) *value = v;
    for (const SettingEntry& e : entries) {
        double x;
        if (k != e.key) continue;
        if (!parseSettingValue(v, x)) return false;
        setSettingValue(e, x);
        return true;
    }
    return false;
}

bool loadOptionsFile(const std::filesystem::path& file, vg::Reconstructor::Options& options) {
    std::ifstream in(file);
    if (!in) return false;
    const std::vector<SettingEntry> entries = optionEntries(options);
    std::string line;
    while (std::getline(in, line)) applySettingLine(line, entries, nullptr, nullptr);
    return true;
}
