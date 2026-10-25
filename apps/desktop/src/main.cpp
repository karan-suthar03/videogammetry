#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "frame_prefetcher.h"
#include "image_io.h"
#include "options_table.h"
#include "power.h"
#include "video_reader.h"
#include "videogammetry/map_quality.h"
#include "videogammetry/mvs.h"
#include "videogammetry/reconstructor.h"
#include "videogammetry/version.h"
#ifdef VG_HAS_GPU
#include "videogammetry/gpu/gpu_mvs.h"
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
        "  --export-colmap <dir>       also write the keyframes as a COLMAP text model (poses + images\n"
        "                              undistorted, at the --size resolution; map points), e.g. for 3DGS\n"
        "  --export-max <n>            keyframes in that export, evenly spaced, 0 = all (default 300)\n"
        "  --dense                     after tracking, also build the dense cloud (as the viewer does) and write\n"
        "                              <output>_dense.ply; with --export-colmap its points3D.txt holds the dense\n"
        "                              points (better splat initialization) instead of the sparse ones\n"
        "  --dense-size <px>           long side of the images the dense stage matches, 0 = video size (default 0)\n"
        "  --dense-cpu                 dense depth maps on the CPU even when there is a GPU\n"
        "  --export-points <file.ply>  with --export-colmap: points3D.txt holds the points of this cloud (e.g. a\n"
        "                              dense cloud saved by the viewer) instead of the map points\n"
        "  --align-to <sparse.ply>     the sparse cloud saved together with that file: it is aligned onto this\n"
        "                              run's map (scale, rotation, translation; ICP) and the same transform moves\n"
        "                              the --export-points cloud into this run's frame. Use the viewer's settings\n"
        "                              (--config) so both runs track alike\n"
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

// Rotation matrix -> unit quaternion (w, x, y, z).
std::array<double, 4> toQuaternion(const vg::Mat3& R) {
    std::array<double, 4> q;
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);
    if (tr > 0) {
        const double s = 2 * std::sqrt(tr + 1);
        q = {s / 4, (R(2, 1) - R(1, 2)) / s, (R(0, 2) - R(2, 0)) / s, (R(1, 0) - R(0, 1)) / s};
    } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
        const double s = 2 * std::sqrt(1 + R(0, 0) - R(1, 1) - R(2, 2));
        q = {(R(2, 1) - R(1, 2)) / s, s / 4, (R(0, 1) + R(1, 0)) / s, (R(0, 2) + R(2, 0)) / s};
    } else if (R(1, 1) > R(2, 2)) {
        const double s = 2 * std::sqrt(1 + R(1, 1) - R(0, 0) - R(2, 2));
        q = {(R(0, 2) - R(2, 0)) / s, (R(0, 1) + R(1, 0)) / s, s / 4, (R(1, 2) + R(2, 1)) / s};
    } else {
        const double s = 2 * std::sqrt(1 + R(2, 2) - R(0, 0) - R(1, 1));
        q = {(R(1, 0) - R(0, 1)) / s, (R(0, 2) + R(2, 0)) / s, (R(1, 2) + R(2, 1)) / s, s / 4};
    }
    if (q[0] < 0) for (double& v : q) v = -v;
    return q;
}

// Writes the keyframes as a COLMAP text model for 3D Gaussian Splatting:
// <dir>/images/*.bmp (frames at the processing size, undistorted to pinhole),
// <dir>/sparse/0/{cameras,images,points3D}.txt with one PINHOLE camera, the
// map points and their keypoints per image. Decodes the video a second time to get the keyframe images.
// At most `maxImages` keyframes, evenly spaced in video order (0 = all),
// with the long side of the images at most `maxSize` (0 = video size).
// `dense` given: points3D.txt holds those points instead of the map points
// (and the images list no 2D points, since dense points have no tracks).
bool exportColmap(const std::string& video, const vg::Reconstructor& rec, const std::filesystem::path& dir,
                  std::size_t maxImages, int maxSize, const vg::PointCloud* dense = nullptr) {
    std::vector<const vg::Keyframe*> all;
    for (const vg::Keyframe& kf : rec.keyframes()) all.push_back(&kf);
    std::sort(all.begin(), all.end(), [](auto* a, auto* b) { return a->frameIndex < b->frameIndex; });
    std::vector<const vg::Keyframe*> kfs = all;
    if (maxImages > 0 && all.size() > maxImages) {
        kfs.clear();
        for (std::size_t i = 0; i < maxImages; ++i)
            kfs.push_back(all[maxImages == 1 ? 0 : i * (all.size() - 1) / (maxImages - 1)]);
    }
    if (kfs.empty()) {
        std::fprintf(stderr, "error: no keyframes to export\n");
        return false;
    }
    const std::filesystem::path imageDir = dir / "images", sparseDir = dir / "sparse" / "0";
    if (!ensureDir(imageDir) || !ensureDir(sparseDir)) return false;

    VideoReader reader;
    std::string error;
    if (!reader.open(video, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return false;
    }
    reader.setMaxSize(maxSize);

    // Intrinsics are in processed pixels with integer pixel centers; COLMAP
    // puts pixel centers at +0.5, so a processed x maps to (x + 0.5) * s,
    // s = exported pixels per processed pixel. The frames are decoded with
    // the same maximum size as the ones the reconstructor was given, so s is
    // its processScale (pixels of the frame it received per processed pixel,
    // not of the video's native size).
    const vg::CameraIntrinsics& K = rec.intrinsics();
    int width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0, scale = 1;
    const double k1 = K.k1;
    std::vector<float> mapX, mapY;  // source position (frame pixel index space) of each undistorted pixel

    const std::vector<vg::MapPoint>& points = rec.mapPoints();
    std::ofstream images(sparseDir / "images.txt");
    images << "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME, then POINTS2D[] as (X Y POINT3D_ID)\n";
    RgbImage out;
    std::size_t next = 0;
    vg::FrameView frame;
    for (long index = 0; next < kfs.size() && reader.read(frame); ++index) {
        if (mapX.empty()) {
            width = frame.width;
            height = frame.height;
            const double s = scale = rec.processScale();
            fx = K.fx * s, fy = K.fy * s, cx = (K.cx + 0.5) * s, cy = (K.cy + 0.5) * s;
            mapX.resize(std::size_t(width) * height);
            mapY.resize(mapX.size());
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    const double nx = (x + 0.5 - cx) / fx, ny = (y + 0.5 - cy) / fy;
                    const double d = 1 + k1 * (nx * nx + ny * ny);
                    mapX[std::size_t(y) * width + x] = float(fx * nx * d + cx - 0.5);
                    mapY[std::size_t(y) * width + x] = float(fy * ny * d + cy - 0.5);
                }
            out.width = width;
            out.height = height;
            out.pixels.resize(std::size_t(width) * height * 3);
        }
        if (index != kfs[next]->frameIndex) continue;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const float sx = std::clamp(mapX[std::size_t(y) * width + x], 0.0f, float(width - 1));
                const float sy = std::clamp(mapY[std::size_t(y) * width + x], 0.0f, float(height - 1));
                const int x0 = std::min(int(sx), width - 2), y0 = std::min(int(sy), height - 2);
                const float ax = sx - x0, ay = sy - y0;
                const std::uint8_t* p = frame.data + std::size_t(y0) * frame.stride + x0 * 3;
                std::uint8_t* o = out.pixels.data() + (std::size_t(y) * width + x) * 3;
                for (int c = 0; c < 3; ++c) {
                    const float top = p[c] * (1 - ax) + p[3 + c] * ax;
                    const float bottom = p[frame.stride + c] * (1 - ax) + p[frame.stride + 3 + c] * ax;
                    o[c] = std::uint8_t(top * (1 - ay) + bottom * ay + 0.5f);
                }
            }
        char name[32];
        std::snprintf(name, sizeof(name), "frame_%05ld.bmp", index);
        if (!writeImage(out.view(), (imageDir / name).string())) {
            std::fprintf(stderr, "error: could not write %s\n", (imageDir / name).string().c_str());
            return false;
        }
        const vg::Keyframe& kf = *kfs[next];
        const std::array<double, 4> q = toQuaternion(kf.pose.R);
        char line[256];
        std::snprintf(line, sizeof(line), "%zu %.9f %.9f %.9f %.9f %.9f %.9f %.9f 1 %s\n", next + 1, q[0], q[1], q[2],
                      q[3], kf.pose.t.x, kf.pose.t.y, kf.pose.t.z, name);
        images << line;
        // Observed (undistorted) keypoints of map points, in exported pixels.
        for (std::size_t i = 0; !dense && i < kf.keypoints.size(); ++i) {
            const int id = kf.pointIds[i];
            if (id < 0 || std::size_t(id) >= points.size() || points[std::size_t(id)].bad) continue;
            std::snprintf(line, sizeof(line), "%.2f %.2f %d ", (kf.keypoints[i].x + 0.5) * scale,
                          (kf.keypoints[i].y + 0.5) * scale, id + 1);
            images << line;
        }
        images << "\n";
        ++next;
    }
    if (next < kfs.size()) {
        std::fprintf(stderr, "error: video ended before keyframe frame %ld\n", kfs[next]->frameIndex);
        return false;
    }

    std::ofstream cameras(sparseDir / "cameras.txt");
    char line[256];
    std::snprintf(line, sizeof(line), "1 PINHOLE %d %d %.6f %.6f %.6f %.6f\n", width, height, fx, fy, cx, cy);
    cameras << "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n" << line;
    std::ofstream pointsFile(sparseDir / "points3D.txt");
    pointsFile << "# POINT3D_ID X Y Z R G B ERROR (no tracks)\n";
    for (std::size_t i = 0; dense && i < dense->points.size(); ++i) {
        const vg::Point& p = dense->points[i];
        std::snprintf(line, sizeof(line), "%zu %.6f %.6f %.6f %d %d %d 0\n", i + 1, p.x, p.y, p.z, p.r, p.g, p.b);
        pointsFile << line;
    }
    for (std::size_t i = 0; !dense && i < points.size(); ++i) {
        const vg::MapPoint& p = points[i];
        if (p.bad) continue;
        std::snprintf(line, sizeof(line), "%zu %.6f %.6f %.6f %d %d %d 0\n", i + 1, p.position.x, p.position.y,
                      p.position.z, p.color[0], p.color[1], p.color[2]);
        pointsFile << line;
    }
    std::printf("export:  %zu keyframes to %s (PINHOLE %dx%d, f %.1f px, undistorted k1 %.3f), %zu %s points\n",
                kfs.size(), dir.string().c_str(), width, height, fx, k1,
                dense ? dense->points.size() : points.size(), dense ? "dense" : "map");
    return true;
}

// The dense stage of the viewer (vg::DenseReconstructor): decodes the
// keyframe images again, computes and fuses the depth maps. Progress on one
// line; the result is empty if it could not run.
vg::PointCloud buildDense(const std::string& video, const vg::Reconstructor& rec, const vg::DenseParams& params,
                          bool gpu) {
    if (rec.keyframes().size() < 3) {
        std::fprintf(stderr, "warning: too few keyframes for the dense cloud\n");
        return {};
    }
    vg::DenseReconstructor dense(rec, params);
#ifdef VG_HAS_GPU
    if (gpu) {
        std::string device;
        vg::DepthMapSolver solver = vg::gpu::makeGpuDepthSolver(device);
        dense.setDepthSolver(std::move(solver), device);
    }
#else
    (void)gpu;
#endif
    VideoReader reader;
    std::string error;
    if (!reader.open(video, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return {};
    }
    reader.setMaxSize(dense.imageLongSide(reader.info().width, reader.info().height));
    const std::vector<long>& needed = dense.frames();
    std::size_t next = 0;
    vg::FrameView frame;
    for (long index = 0; next < needed.size() && reader.read(frame); ++index) {
        if (index != needed[next]) continue;
        dense.addImage(index, frame);
        ++next;
    }
    const vg::DenseReconstructor::Progress start = dense.progress();
    std::printf("dense:   kept %d of %d keyframes, images %dx%d, depth maps on %s\n", start.selected,
                start.totalKeyframes, start.width, start.height, start.solver.c_str());
    const auto t0 = std::chrono::steady_clock::now();
    auto last = t0;
    const std::atomic<bool> cancel{false};
    dense.run(cancel, [&]() {
        const auto now = std::chrono::steady_clock::now();
        const vg::DenseReconstructor::Progress p = dense.progress();
        if (!p.finished && now - last < std::chrono::milliseconds(500)) return;
        last = now;
        std::printf("\rdense:   depth maps %d / %d, %.2f M points, %.0f s   ", p.depthMaps, p.references, p.points / 1e6,
                    std::chrono::duration<double>(now - t0).count());
        std::fflush(stdout);
    });
    const vg::DenseReconstructor::Progress p = dense.progress();
    std::printf("\ndense:   %.2f M points from %d keyframes in %.0f s (%s)\n", p.points / 1e6, p.references,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), p.solver.c_str());
    return dense.pointCloud();
}

// Reads a binary little-endian PLY vertex list (as vg::writePly writes it,
// with or without normals): positions and colors.
bool readPly(const std::string& path, vg::PointCloud& out, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "could not open " + path;
        return false;
    }
    std::string line;
    std::getline(in, line);
    if (line.rfind("ply", 0) != 0) {
        error = path + " is not a PLY file";
        return false;
    }
    std::size_t count = 0, stride = 0;
    int ox = -1, oy = -1, oz = -1, ored = -1, ogreen = -1, oblue = -1;
    bool binary = false, inVertex = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "end_header") break;
        std::istringstream s(line);
        std::string word, type, name;
        s >> word;
        if (word == "format") {
            s >> type;
            binary = type == "binary_little_endian";
        } else if (word == "element") {
            s >> name >> count;
            inVertex = name == "vertex";
            if (!inVertex) count = 0;
        } else if (word == "property" && inVertex) {
            s >> type >> name;
            const int offset = static_cast<int>(stride);
            stride += type == "double" ? 8 : (type == "float" || type == "int" || type == "uint") ? 4
                    : (type == "short" || type == "ushort") ? 2 : 1;
            if (name == "x") ox = offset;
            if (name == "y") oy = offset;
            if (name == "z") oz = offset;
            if (name == "red") ored = offset;
            if (name == "green") ogreen = offset;
            if (name == "blue") oblue = offset;
            if ((name == "x" || name == "y" || name == "z") && type != "float") {
                error = path + ": positions must be float";
                return false;
            }
        }
    }
    if (!binary || ox < 0 || oy < 0 || oz < 0) {
        error = path + ": needs binary little-endian float x, y, z";
        return false;
    }
    std::vector<char> data(count * stride);
    in.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (static_cast<std::size_t>(in.gcount()) != data.size()) {
        error = path + " is truncated";
        return false;
    }
    out.points.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const char* v = data.data() + i * stride;
        vg::Point& p = out.points[i];
        std::memcpy(&p.x, v + ox, 4);
        std::memcpy(&p.y, v + oy, 4);
        std::memcpy(&p.z, v + oz, 4);
        if (ored >= 0) p.r = static_cast<std::uint8_t>(v[ored]);
        if (ogreen >= 0) p.g = static_cast<std::uint8_t>(v[ogreen]);
        if (oblue >= 0) p.b = static_cast<std::uint8_t>(v[oblue]);
    }
    return true;
}

// Similarity transform x -> s R x + t.
struct Similarity {
    double s = 1;
    vg::Mat3 R = vg::Mat3::identity();
    vg::Vec3 t;
    vg::Vec3 apply(const vg::Vec3& x) const { return R * x * s + t; }
};

// Aligns `src` onto `dst` (two reconstructions of the same scene that start
// at the same camera, so they agree up to scale at first): scale from the
// median distance to the origin, then ICP with a similarity transform
// (Umeyama) and a shrinking match radius. `fit` receives the inlier RMS
// relative to the size of `dst`, `inliers` the fraction of `src` matched.
Similarity alignClouds(std::vector<vg::Vec3> src, const std::vector<vg::Vec3>& dst, double& fit, double& inliers) {
    Similarity T;
    fit = 1;
    inliers = 0;
    if (src.size() < 10 || dst.size() < 10) return T;
    if (src.size() > 50000) {  // evenly spaced subset
        std::vector<vg::Vec3> sub;
        for (std::size_t i = 0; i < 50000; ++i) sub.push_back(src[i * src.size() / 50000]);
        src = std::move(sub);
    }
    const auto medianNorm = [](const std::vector<vg::Vec3>& v, const vg::Vec3& c) {
        std::vector<double> d;
        for (const vg::Vec3& p : v) d.push_back(vg::norm(p - c));
        std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
        return d[d.size() / 2];
    };
    const double dstSize = medianNorm(dst, {});
    T.s = dstSize / std::max(1e-12, medianNorm(src, {}));

    for (double radius = 0.2 * dstSize; radius > 0.004 * dstSize; radius *= 0.5) {
        // Grid of `dst` with cells of the match radius.
        std::unordered_map<long long, std::vector<int>> grid;
        const auto key = [&](long long x, long long y, long long z) { return (x * 73856093LL) ^ (y * 19349663LL) ^ (z * 83492791LL); };
        const auto cell = [&](double v) { return static_cast<long long>(std::floor(v / radius)); };
        for (std::size_t i = 0; i < dst.size(); ++i)
            grid[key(cell(dst[i].x), cell(dst[i].y), cell(dst[i].z))].push_back(static_cast<int>(i));
        for (int iter = 0; iter < 10; ++iter) {
            std::vector<std::pair<vg::Vec3, vg::Vec3>> pairs;  // (src, dst)
            double sq = 0;
            for (const vg::Vec3& p0 : src) {
                const vg::Vec3 p = T.apply(p0);
                const long long cx = cell(p.x), cy = cell(p.y), cz = cell(p.z);
                double best = radius * radius;
                int bi = -1;
                for (long long dx = -1; dx <= 1; ++dx)
                    for (long long dy = -1; dy <= 1; ++dy)
                        for (long long dz = -1; dz <= 1; ++dz) {
                            const auto it = grid.find(key(cx + dx, cy + dy, cz + dz));
                            if (it == grid.end()) continue;
                            for (int j : it->second) {
                                const vg::Vec3 d = dst[static_cast<std::size_t>(j)] - p;
                                const double dd = vg::dot(d, d);
                                if (dd < best) best = dd, bi = j;
                            }
                        }
                if (bi < 0) continue;
                pairs.push_back({p0, dst[static_cast<std::size_t>(bi)]});
                sq += best;
            }
            if (pairs.size() < 10) return T;
            fit = std::sqrt(sq / pairs.size()) / dstSize;
            inliers = double(pairs.size()) / src.size();
            // Umeyama: the similarity minimizing the squared pair distances.
            vg::Vec3 ma, mb;
            for (const auto& [a, b] : pairs) ma = ma + a, mb = mb + b;
            ma = ma * (1.0 / pairs.size());
            mb = mb * (1.0 / pairs.size());
            vg::Mat3 C;
            double va = 0;
            for (const auto& [a, b] : pairs) {
                const vg::Vec3 da = a - ma, db = b - mb;
                va += vg::dot(da, da);
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) C(r, c) += db[r] * da[c];
            }
            vg::Mat3 U, V;
            vg::Vec3 D;
            vg::svd3(C, U, D, V);
            vg::Mat3 S = vg::Mat3::identity();
            if ((U * V.transposed()).determinant() < 0) S(2, 2) = -1;
            T.R = U * S * V.transposed();
            T.s = (D.x * S(0, 0) + D.y * S(1, 1) + D.z * S(2, 2)) / std::max(1e-12, va);
            T.t = mb - T.R * ma * T.s;
        }
    }
    return T;
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
    std::string colmapDir;
    std::size_t exportMax = 300;
    bool dense = false, denseGpu = true;
    std::string exportPointsPath, alignToPath;
    vg::DenseParams denseParams;
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
        } else if (std::strcmp(argv[i], "--export-colmap") == 0 && i + 1 < argc) {
            colmapDir = argv[++i];
        } else if (std::strcmp(argv[i], "--dense") == 0) {
            dense = true;
        } else if (std::strcmp(argv[i], "--dense-size") == 0 && i + 1 < argc) {
            denseParams.imageSize = static_cast<int>(std::max(0L, std::strtol(argv[++i], nullptr, 10)));
        } else if (std::strcmp(argv[i], "--export-points") == 0 && i + 1 < argc) {
            exportPointsPath = argv[++i];
        } else if (std::strcmp(argv[i], "--align-to") == 0 && i + 1 < argc) {
            alignToPath = argv[++i];
        } else if (std::strcmp(argv[i], "--dense-cpu") == 0) {
            denseGpu = false;
        } else if (std::strcmp(argv[i], "--export-max") == 0 && i + 1 < argc) {
            exportMax = static_cast<std::size_t>(std::max(0L, std::strtol(argv[++i], nullptr, 10)));
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
        std::printf("noise:   depth uncertainty (1 px) median %.2f%%, 90th percentile %.2f%% of the distance\n",
                    q.medianNoisePct, q.p90NoisePct);
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

    vg::PointCloud denseCloud;
    if (dense) {
        denseCloud = buildDense(input, reconstructor, denseParams, denseGpu);
        std::string densePath = output;
        if (densePath.size() > 4 && densePath.ends_with(".ply")) densePath.resize(densePath.size() - 4);
        densePath += "_dense.ply";
        if (!vg::writePly(denseCloud, densePath)) {
            std::fprintf(stderr, "error: could not write %s\n", densePath.c_str());
            return 1;
        }
        std::printf("output:  %s (%zu points)\n", densePath.c_str(), denseCloud.points.size());
    }

    if (!exportPointsPath.empty()) {
        if (!denseCloud.points.empty()) std::printf("note: --export-points replaces the --dense points in the export\n");
        std::string err;
        if (!readPly(exportPointsPath, denseCloud, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        std::printf("points:  %s (%zu points)\n", exportPointsPath.c_str(), denseCloud.points.size());
        if (!alignToPath.empty()) {
            vg::PointCloud old;
            if (!readPly(alignToPath, old, err)) {
                std::fprintf(stderr, "error: %s\n", err.c_str());
                return 1;
            }
            std::vector<vg::Vec3> src, dst;
            for (const vg::Point& p : old.points) src.push_back({p.x, p.y, p.z});
            for (const vg::Point& p : cloud.points) dst.push_back({p.x, p.y, p.z});
            double fit = 1, inliers = 0;
            const Similarity T = alignClouds(src, dst, fit, inliers);
            std::printf("align:   %s onto this map: scale %.4f, rotation %.2f deg, %.0f%% matched, RMS %.2f%% of the "
                        "scene size\n", alignToPath.c_str(), T.s, vg::rotationAngle(T.R) * 180 / 3.14159265358979,
                        inliers * 100, fit * 100);
            if (inliers < 0.5 || fit > 0.02)
                std::fprintf(stderr, "warning: poor alignment; the runs may have tracked differently (same --config "
                                     "as the viewer?)\n");
            for (vg::Point& p : denseCloud.points) {
                const vg::Vec3 q = T.apply({p.x, p.y, p.z});
                p.x = static_cast<float>(q.x);
                p.y = static_cast<float>(q.y);
                p.z = static_cast<float>(q.z);
            }
        }
        denseCloud.normals.clear();
    }

    if (!colmapDir.empty()) {
        if (st.maps > 1 && reconstructor.trajectory().size() != reconstructor.keyframes().size())
            std::fprintf(stderr, "warning: tracking restarted; exporting the last map, not the largest\n");
        if (!exportColmap(input, reconstructor, colmapDir, exportMax, options.processSize,
                          denseCloud.points.empty() ? nullptr : &denseCloud))
            return 1;
    }

    if (!dumpDir.empty()) {
        const RgbImage views = renderMapViews(cloud.points, path, 700);
        const std::string mapPath = (std::filesystem::path(dumpDir) / "map.bmp").string();
        if (writeImage(views.view(), mapPath)) ++dumped;
        std::printf("dumped:  %ld images to %s (map views: %s)\n", dumped, dumpDir.c_str(), mapPath.c_str());
    }
    return 0;
}
