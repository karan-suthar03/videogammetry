#include "settings_table.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

std::vector<SettingEntry> settingEntries(ViewerSettings& s) {
    vg::Reconstructor::Options& o = s.session.options;
    vg::OrbParams& orb = o.orb;
    vg::MappingParams& m = o.mapping;
    // clang-format off
    return {
        {"Playback", "realTime", "Real-time playback", "off = as fast as possible", &s.session.realTime, 0, 1, 1},
        {"Playback", "thumbnailSize", "Video thumbnail size", "px, up to the full frame", &s.session.thumbnailSize, 1, kInf, 20},
        {"Playback", "pointSize", "Point size", "px, 1..100", &s.render.pointSize, 1, 100, 1},
        {"Playback", "highlightTracked", "Highlight tracked points", "", &s.render.highlightTracked, 0, 1, 1},
        {"Playback", "showKeyframes", "Show keyframes", "", &s.render.showKeyframes, 0, 1, 1},

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

void loadSettings(const std::filesystem::path& file, ViewerSettings& s, std::vector<std::string>& recent) {
    std::ifstream in(file);
    if (!in) return;
    const std::vector<SettingEntry> entries = settingEntries(s);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq), value = line.substr(eq + 1);
        if (key == "recent") {
            if (!value.empty()) recent.push_back(value);
            continue;
        }
        for (const SettingEntry& e : entries) {
            double v;
            if (key == e.key && parseSettingValue(value, v)) setSettingValue(e, v);
        }
    }
}

void saveSettings(const std::filesystem::path& file, ViewerSettings& s, const std::vector<std::string>& recent) {
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    std::ofstream out(file, std::ios::trunc);
    if (!out) return;
    for (const SettingEntry& e : settingEntries(s)) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", settingValue(e));  // exact round trip
        out << e.key << '=' << buf << '\n';
    }
    for (const std::string& r : recent) out << "recent=" << r << '\n';
}
