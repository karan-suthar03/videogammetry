#include "settings_table.h"

#include <cstdio>
#include <fstream>
#include <limits>

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

std::vector<SettingEntry> settingEntries(ViewerSettings& s) {
    // clang-format off
    std::vector<SettingEntry> entries = {
        {"Playback", "realTime", "Real-time playback", "off = as fast as possible", &s.session.realTime, 0, 1, 1},
        {"Playback", "thumbnailSize", "Video thumbnail size", "px, up to the full frame", &s.session.thumbnailSize, 1, kInf, 20},
        {"Playback", "pointSize", "Point size", "px, 1..100", &s.render.pointSize, 1, 100, 1},
        {"Playback", "highlightTracked", "Highlight tracked points", "", &s.render.highlightTracked, 0, 1, 1},
        {"Playback", "colorByNoise", "Color points by noise", "green = precise, red = noisy", &s.render.colorByNoise, 0, 1, 1},
        {"Playback", "showKeyframes", "Show keyframes", "", &s.render.showKeyframes, 0, 1, 1},
        {"Playback", "showSparse", "Show sparse points", "the map points", &s.render.showSparse, 0, 1, 1},
        {"Playback", "showDense", "Show dense cloud", "", &s.render.showDense, 0, 1, 1},
        {"Processing", "gpu", "Extract features on the GPU", "Vulkan; CPU if none", &s.session.gpu, 0, 1, 1},
        {"Dense", "dense", "Dense cloud after the video", "", &s.session.dense, 0, 1, 1},
        {"Dense", "denseGpu", "Dense depth maps on the GPU", "Vulkan; CPU if none", &s.session.denseGpu, 0, 1, 1},
        {"Dense", "denseImageSize", "Dense image size", "px, long side; 0 = the video's", &s.session.denseParams.imageSize, 0, kInf, 160},
        {"Dense", "denseMaxMemoryMB", "Dense memory limit", "MB; lowers the image size", &s.session.denseParams.maxMemoryMB, 256, kInf, 512},
        {"Dense", "denseSourceViews", "Dense views per keyframe", "2..16", &s.session.denseParams.sourceViews, 2, 16, 1},
        {"Dense", "denseScales", "Dense image levels", "coarse to fine", &s.session.denseParams.scales, 1, 6, 1},
        {"Dense", "denseIterations", "Dense rounds, coarsest level", "", &s.session.denseParams.iterations, 1, kInf, 1},
        {"Dense", "denseFineIterations", "Dense rounds, finer levels", "", &s.session.denseParams.fineIterations, 0, kInf, 1},
        {"Dense", "denseMinNcc", "Dense match threshold", "NCC, -1..1", &s.session.denseParams.minNcc, -1, 1, 0.05},
        {"Dense", "denseMinViews", "Dense agreeing views", "besides the keyframe", &s.session.denseParams.minConsistentViews, 1, 16, 1},
        {"Dense", "denseSelect", "Dense: thin out close keyframes", "keep the best of each spot", &s.session.denseParams.selectKeyframes, 0, 1, 1},
        {"Dense", "denseSpacing", "Dense keyframe spacing", "x scene depth", &s.session.denseParams.keyframeSpacing, 0, kInf, 0.01},
        {"Dense", "denseKeyframeAngle", "Dense keyframe angle", "deg; turned more = kept", &s.session.denseParams.keyframeAngleDeg, 0, 180, 1},
        {"Dense", "denseThreads", "Dense threads", "0 = all", &s.session.denseParams.threads, 0, kInf, 1},
    };
    // clang-format on
    const std::vector<SettingEntry> options = optionEntries(s.session.options);
    entries.insert(entries.end(), options.begin(), options.end());
    return entries;
}

bool appliesLive(const SettingEntry& e, const ViewerSettings& s) {
    const void* p = std::visit([](auto* v) { return static_cast<const void*>(v); }, e.value);
    const auto* begin = reinterpret_cast<const char*>(&s.render);
    return p >= begin && p < begin + sizeof(s.render);
}

void loadSettings(const std::filesystem::path& file, ViewerSettings& s, std::vector<std::string>& recent) {
    std::ifstream in(file);
    if (!in) return;
    const std::vector<SettingEntry> entries = settingEntries(s);
    std::string line, key, value;
    while (std::getline(in, line)) {
        if (!applySettingLine(line, entries, &key, &value) && key == "recent" && !value.empty()) {
            recent.push_back(value);
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
