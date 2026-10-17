#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "frame_prefetcher.h"
#include "image_io.h"
#include "options_table.h"
#include "power.h"
#include "video_reader.h"
#include "videogammetry/map_quality.h"
#include "videogammetry/reconstructor.h"
#include "videogammetry/version.h"
#ifdef VG_HAS_GPU
#include "videogammetry/gpu/gpu_orb.h"
#endif

namespace {

void printUsage() {
    std::printf(
        "Usage: videogammetry <input-video> [options]\n"
        "       videogammetry --version\n"
        "\n"
        "Options:\n"
        "  -o <file.ply>               output point cloud (default: cloud.ply)\n"
        "  --config <file.ini>         reconstruction options from a settings file, e.g. the viewer's\n"
        "                              %%APPDATA%%/videogammetry/viewer.ini (options after it override it)\n"
        "  --trajectory <file.ply>     also write the keyframe camera positions\n"
        "  --size <px>                 long side of the frame is downscaled to this, 0 = full resolution (default 0)\n"
        "  --threads <n>               worker threads for tracking/mapping, 0 = all hardware threads (default 0)\n"
        "  --extract-workers <n>       frames whose features are extracted in parallel, 0 = all hardware threads\n"
        "  --serial                    extract features inside the reconstructor, one frame at a time\n"
        "  --gpu                       extract features on the GPU (Vulkan; falls back to the CPU)\n"
        "  --center-boost <x>          extra share of features for the middle of the image, 0 = even (default 0)\n"
        "  --confirm <n>               keyframes that must see a new point to keep it, 2 = no culling (default 3)\n"
        "  --fixed-focal               keep the --fov focal length (no self-calibration)\n"
        "  --no-background-ba          no continuous global BA on a background thread (deterministic results)\n"
        "  --no-mapping-thread         solve local BA inline on the keyframe's frame (deterministic results)\n"
        "  --fuse <n>                  covisible keyframes to fuse points with, 0 = no fusion (default 10)\n"
        "  --no-merge                  fusion only links points to free keypoints, never merges duplicates\n"
        "  --fov <deg>                 assumed field of view across the long side of the frame (default 65)\n"
        "  --dump <dir>                write debug images: <dir>/tracking (every --dump-every frames)\n"
        "                              and <dir>/map.bmp (top and side view of the result)\n"
        "  --dump-every <n>            frame interval for --dump (default 25)\n"
        "  --save-frame <n> <file>     write decoded frame n (0-based) as .bmp or .ppm\n");
}

bool ensureDir(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "error: cannot create %s: %s\n", dir.string().c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

std::string numbered(const std::filesystem::path& dir, const char* prefix, long index) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s_%05ld.bmp", prefix, index);
    return (dir / name).string();
}

const char* stateName(vg::Reconstructor::State s) {
    switch (s) {
        case vg::Reconstructor::State::Initializing: return "initializing";
        case vg::Reconstructor::State::Tracking: return "tracking";
        case vg::Reconstructor::State::Lost: return "lost";
    }
    return "?";
}

}  // namespace

int main(int argc, char** argv) {
    std::string input;
    std::string output = "cloud.ply";
    std::string trajectoryPath;
    long saveFrameIndex = -1;
    std::string saveFramePath;
    vg::Reconstructor::Options options;
    std::string dumpDir;
    long dumpEvery = 25;
    bool serial = false;
    bool gpu = false;
    int extractWorkers = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            std::printf("videogammetry %s\n", vg::version());
            return 0;
        }
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            printUsage();
            return 0;
        }
        if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            if (!loadOptionsFile(argv[++i], options)) {
                std::fprintf(stderr, "error: cannot read %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--trajectory") == 0 && i + 1 < argc) {
            trajectoryPath = argv[++i];
        } else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            options.processSize = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            options.threads = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--gpu") == 0) {
            gpu = true;
        } else if (std::strcmp(argv[i], "--serial") == 0) {
            serial = true;
        } else if (std::strcmp(argv[i], "--extract-workers") == 0 && i + 1 < argc) {
            extractWorkers = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--center-boost") == 0 && i + 1 < argc) {
            options.orb.centerBoost = static_cast<float>(std::strtod(argv[++i], nullptr));
        } else if (std::strcmp(argv[i], "--no-merge") == 0) {
            options.mapping.fuseMerge = false;
        } else if (std::strcmp(argv[i], "--no-mapping-thread") == 0) {
            options.mapping.mappingThread = false;
        } else if (std::strcmp(argv[i], "--no-background-ba") == 0) {
            options.mapping.backgroundBA = false;
        } else if (std::strcmp(argv[i], "--fixed-focal") == 0) {
            options.mapping.refineFocal = false;
        } else if (std::strcmp(argv[i], "--fuse") == 0 && i + 1 < argc) {
            options.mapping.fuseNeighbours = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--confirm") == 0 && i + 1 < argc) {
            options.mapping.minConfirmingKeyframes = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--fov") == 0 && i + 1 < argc) {
            options.fovDeg = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dumpDir = argv[++i];
        } else if (std::strcmp(argv[i], "--dump-every") == 0 && i + 1 < argc) {
            dumpEvery = std::max(1L, std::strtol(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--save-frame") == 0 && i + 2 < argc) {
            saveFrameIndex = std::strtol(argv[++i], nullptr, 10);
            saveFramePath = argv[++i];
        } else if (input.empty()) {
            input = argv[i];
        } else {
            printUsage();
            return 1;
        }
    }

    if (input.empty()) {
        printUsage();
        return 1;
    }

    disablePowerThrottling();

    VideoReader reader;
    std::string error;
    if (!reader.open(input, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    reader.setMaxSize(options.processSize);  // downscale while converting to RGB
    const VideoReader::Info& info = reader.info();
    std::printf("input:   %s\n", input.c_str());
    std::printf("video:   %dx%d, %.2f fps, %.2f s, codec %s",
                info.width, info.height, info.fps, info.durationSec, info.codec.c_str());
    if (info.frameCountEstimate > 0) {
        std::printf(", ~%lld frames", static_cast<long long>(info.frameCountEstimate));
    }
    std::printf("\n");

    const int longSide = std::max(info.width, info.height);
    const bool downscaled = options.processSize > 0 && longSide > options.processSize;
    const double shrink = downscaled ? double(longSide) / options.processSize : 1.0;
    std::printf("process: %ldx%ld%s, assumed FOV %.0f deg across the long side\n", std::lround(info.width / shrink),
                std::lround(info.height / shrink), downscaled ? " (downscaled)" : " (full resolution)",
                options.fovDeg);

    const std::filesystem::path trackingDir = std::filesystem::path(dumpDir) / "tracking";
    if (!dumpDir.empty() && !ensureDir(trackingDir)) return 1;

    vg::Reconstructor reconstructor(options);
    std::printf("threads: %d of %d hardware threads (+ background decode)\n",
                reconstructor.threadCount(), vg::ThreadPool::hardwareThreads());

    long dumped = 0;
    vg::FrameView frame;
    long index = 0;
    double lastTimestamp = 0.0;
    double decodeWaitMs = 0.0;
    long long totalFeatures = 0, totalTracked = 0;
    long initializedAt = -1;

    // Per-frame log for diagnosing tracking: written with --dump.
    std::ofstream frameLog;
    if (!dumpDir.empty()) {
        frameLog.open(std::filesystem::path(dumpDir) / "frames.csv");
        frameLog << "frame,time,state,tracked,keyframe,keyframes,map_points,cx,cy,cz,orb_ms,track_ms,mapping_ms\n";
    }

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    // Decode thread -> feature pipeline (many frames extracted in parallel) ->
    // this thread (tracking and mapping, in order). --serial instead extracts
    // inside the reconstructor, one frame at a time.
    FramePrefetcher frames(reader);
    std::unique_ptr<vg::FeaturePipeline> pipeline;
    std::thread feeder;
    if (!serial) {
        vg::FrameExtractorFactory makeExtractor;
        std::string extractorName = "CPU";
        if (gpu) {
#ifdef VG_HAS_GPU
            makeExtractor = vg::gpu::makeGpuExtractorFactory(options.processSize, options.orb, extractorName);
            if (makeExtractor && extractWorkers <= 0) extractWorkers = 3;  // overlap upload, GPU and host work
#else
            extractorName = "CPU (built without the GPU backend)";
#endif
        }
        pipeline = std::make_unique<vg::FeaturePipeline>(options.processSize, options.orb, extractWorkers,
                                                         std::move(makeExtractor));
        std::printf("extract: %d frames in parallel on %s\n", pipeline->workers(), extractorName.c_str());
        feeder = std::thread([&] {
            vg::FrameView f;
            while (frames.next(f)) pipeline->push(f);
            pipeline->close();
        });
    }
    vg::FrameFeatures features;
    while (true) {
        const auto waitStart = Clock::now();
        if (serial ? !frames.next(frame) : !pipeline->pop(features, &frame)) break;
        decodeWaitMs += std::chrono::duration<double, std::milli>(Clock::now() - waitStart).count();

        if (index == saveFrameIndex) {
            if (writeImage(frame, saveFramePath)) {
                std::printf("saved frame %ld (t=%.3f s) to %s\n", index, frame.timestamp, saveFramePath.c_str());
            } else {
                std::fprintf(stderr, "warning: could not write %s\n", saveFramePath.c_str());
            }
        }

        // Per-stage time of this frame = growth of each stage's total.
        const auto stageTotal = [&](const char* name) {
            for (const vg::Profiler::Stage& s : reconstructor.profiler().stages()) {
                if (s.name == name) return s.totalMs;
            }
            return 0.0;
        };
        const double orbBefore = stageTotal("orb"), trackBefore = stageTotal("track"),
                     mappingBefore = stageTotal("mapping");
        if (serial) {
            reconstructor.addFrame(frame);
        } else {
            reconstructor.addFeatures(std::move(features));
        }
        const double orbMs = stageTotal("orb") - orbBefore, trackMs = stageTotal("track") - trackBefore,
                     mappingMs = stageTotal("mapping") - mappingBefore;
        totalFeatures += static_cast<long long>(reconstructor.lastKeypoints().size());
        totalTracked += reconstructor.lastTrackedCount();
        if (initializedAt < 0 && reconstructor.state() == vg::Reconstructor::State::Tracking) initializedAt = index;

        if (frameLog.is_open()) {
            const vg::Vec3 c = reconstructor.lastPose().center();
            char line[256];
            std::snprintf(line, sizeof(line), "%ld,%.3f,%s,%d,%d,%zu,%zu,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\n", index,
                          frame.timestamp, stateName(reconstructor.state()), reconstructor.lastTrackedCount(),
                          reconstructor.lastWasKeyframe() ? 1 : 0, reconstructor.keyframes().size(),
                          reconstructor.mapPoints().size(), c.x, c.y, c.z, orbMs, trackMs, mappingMs);
            frameLog << line;
        }

        if (!dumpDir.empty() && index % dumpEvery == 0) {
            const RgbImage img = drawTracking(frame, reconstructor.lastKeypoints(), reconstructor.lastTrackedPoints(),
                                              reconstructor.processScale());
            if (writeImage(img.view(), numbered(trackingDir, "tracking", index))) ++dumped;
        }

        lastTimestamp = frame.timestamp;
        ++index;
    }
    if (feeder.joinable()) feeder.join();
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    frameLog.close();

    if (!frames.error().empty()) {
        std::fprintf(stderr, "error: %s (after %ld frames)\n", frames.error().c_str(), index);
        return 1;
    }

    reconstructor.finish();  // global bundle adjustment
    const vg::Reconstructor::Stats& st = reconstructor.stats();
    const vg::PointCloud cloud = reconstructor.pointCloud();
    const std::vector<vg::Vec3> path = reconstructor.trajectory();

    // Final keyframe poses (after all refinement), same columns as frames.csv.
    if (!dumpDir.empty()) {
        std::ofstream kfLog(std::filesystem::path(dumpDir) / "keyframes.csv");
        kfLog << "frame,time,state,cx,cy,cz\n";
        for (const vg::Keyframe& kf : reconstructor.keyframes()) {
            const vg::Vec3 c = kf.pose.center();
            char line[160];
            std::snprintf(line, sizeof(line), "%ld,%.3f,tracking,%.5f,%.5f,%.5f\n", kf.frameIndex,
                          info.fps > 0 ? kf.frameIndex / info.fps : 0.0, c.x, c.y, c.z);
            kfLog << line;
        }
    }

    std::printf("frames:  %ld (last t=%.3f s), %.0f ORB features/frame avg\n",
                index, lastTimestamp, index ? double(totalFeatures) / index : 0.0);
    if (initializedAt >= 0) {
        std::printf("map:     initialized at frame %ld; tracked %d frames (%.0f map points/frame avg), "
                    "lost %d frames, %d map(s)\n",
                    initializedAt, st.trackedFrames, st.trackedFrames ? double(totalTracked) / st.trackedFrames : 0.0,
                    st.lostFrames, st.maps);
        std::printf("         %zu keyframes, %zu map points (in output); wide searches %d, keyframe re-matches %d; "
                    "final state: %s\n",
                    path.size(), cloud.points.size(), st.wideSearches, st.keyframeSearches,
                    stateName(reconstructor.state()));
        if (st.localBAs > 0) {
            std::printf("ba:      local x%d: reprojection %.2f -> %.2f px avg, %d observations rejected\n", st.localBAs,
                        st.localBARmsBefore / st.localBAs, st.localBARmsAfter / st.localBAs, st.localBAOutliers);
        }
        if (st.globalBA.ok) {
            std::printf("         global (%d keyframes, %d points): reprojection %.2f -> %.2f px, %d iterations, "
                        "%d observations rejected\n",
                        st.globalBACameras, st.globalBAPoints, st.globalBA.initialRmsPx, st.globalBA.finalRmsPx,
                        st.globalBA.iterations, st.globalBA.outliers);
        }
        std::printf("camera:  field of view %.1f deg (started from %.1f), radial distortion k1 %.3f; refined %d "
                    "times, %d times not determined by the motion\n",
                    reconstructor.fovDeg(), options.fovDeg, reconstructor.distortion(), st.focalUpdates,
                    st.focalRejected);
        std::printf("bg ba:   %d passes merged, %d discarded (intrinsics changed meanwhile); last took %.0f ms on "
                    "its own thread; tracking thread spent %.0f ms copying/merging\n",
                    st.backgroundPasses, st.backgroundDiscarded, st.backgroundLastMs, st.backgroundSnapshotMs);
        std::printf("points:  %d triangulated, %d culled as unconfirmed; fusion: %d merged, %d observations added\n",
                    st.pointsCreated, st.pointsCulled, st.pointsMerged, st.observationsFused);
        const vg::MapQuality q =vg::evaluateMap(reconstructor.mapPoints(), reconstructor.keyframes());
        std::printf("quality: %d points: %.1f keyframes/point avg, %.0f%% seen by only 2; parallax median %.1f deg, "
                    "%.0f%% below 2 deg; surface variation median %.3f, %.0f%% scattered\n",
                    q.points, q.meanObservations, 100 * q.twoViewShare, q.medianParallaxDeg,
                    100 * q.lowParallaxShare, q.medianSurfaceVariation, 100 * q.scatteredShare);
    } else {
        std::printf("map:     not initialized (not enough parallax/matches between frames)\n");
    }

    std::printf("\n  %-14s %10s %10s %10s\n", "stage", "total ms", "avg ms", "max ms");
    // Decoding runs on its own thread; this is only the time spent waiting for it.
    std::printf("  %-14s %10.1f %10.2f %10s\n", "decode*", decodeWaitMs, index ? decodeWaitMs / index : 0.0, "-");
    double coreMs = 0.0;
    for (const vg::Profiler::Stage& s : reconstructor.profiler().stages()) {
        std::printf("  %-14s %10.1f %10.2f %10.2f\n", s.name.c_str(), s.totalMs, s.avgMs(), s.maxMs);
        if (s.name.find('.') == std::string::npos) coreMs += s.totalMs;  // "a.b" are parts of stage "a"
    }
    std::printf("  %-14s %10.1f %10.2f\n", "all", seconds * 1000.0, index ? seconds * 1000.0 / index : 0.0);
    std::printf("\nthroughput: %.1f fps overall, %.1f fps core only (video is %.2f fps)\n",
                seconds > 0 ? index / seconds : 0.0,
                coreMs > 0 ? index / (coreMs / 1000.0) : 0.0, info.fps);

    if (!vg::writePly(cloud, output)) {
        std::fprintf(stderr, "error: could not write %s\n", output.c_str());
        return 1;
    }
    std::printf("output:  %s (%zu points)\n", output.c_str(), cloud.points.size());

    if (!trajectoryPath.empty()) {
        vg::PointCloud traj;
        for (const vg::Vec3& c : path) {
            vg::Point p;
            p.x = static_cast<float>(c.x);
            p.y = static_cast<float>(c.y);
            p.z = static_cast<float>(c.z);
            p.r = 255;
            p.g = 40;
            p.b = 40;
            traj.points.push_back(p);
        }
        if (vg::writePly(traj, trajectoryPath)) {
            std::printf("output:  %s (%zu camera positions)\n", trajectoryPath.c_str(), traj.points.size());
        }
    }

    if (!dumpDir.empty()) {
        const RgbImage views = renderMapViews(cloud.points, path, 700);
        const std::string mapPath = (std::filesystem::path(dumpDir) / "map.bmp").string();
        if (writeImage(views.view(), mapPath)) ++dumped;
        std::printf("dumped:  %ld images to %s (map views: %s)\n", dumped, dumpDir.c_str(), mapPath.c_str());
    }
    return 0;
}
