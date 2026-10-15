// JNI bridge between the Android app and the C++ core.
// A session wraps one Reconstructor; frames arrive as RGBA_8888 Bitmaps.
// All calls for one session must come from the same thread (the app's
// processing thread); the core runs its own worker and background threads.

#include <jni.h>

#include <android/bitmap.h>
#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "videogammetry/reconstructor.h"
#include "videogammetry/version.h"

namespace {

struct Session {
    explicit Session(const vg::Reconstructor::Options& options) : reconstructor(options) {}
    vg::Reconstructor reconstructor;
    long frames = 0;
    std::vector<float> depth;  // z-buffer of the live view
};

Session* session(jlong handle) { return reinterpret_cast<Session*>(handle); }

const char* stateName(vg::Reconstructor::State s) {
    switch (s) {
        case vg::Reconstructor::State::Initializing: return "initializing";
        case vg::Reconstructor::State::Tracking: return "tracking";
        case vg::Reconstructor::State::Lost: return "LOST";
    }
    return "?";
}

// Layout of the option arrays passed to create() (kept in sync with Settings.java).
enum IntOption { kProcessSize, kMaxFeatures, kThreads, kLocalBAKeyframes, kLocalBAIterations, kFlags, kIntCount };
enum FloatOption { kFovDeg, kFloatCount };
enum Flag {
    kBackgroundBA = 1 << 0,
    kRefineFocal = 1 << 1,
    kRefineDistortion = 1 << 2,
    kFusion = 1 << 3,
    kConfirmPoints = 1 << 4,
};

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL Java_com_videogammetry_NativeBridge_version(JNIEnv* env, jclass) {
    return env->NewStringUTF(vg::version());
}

// ints / floats: see IntOption / FloatOption.
JNIEXPORT jlong JNICALL Java_com_videogammetry_NativeBridge_create(JNIEnv* env, jclass, jintArray intsArray,
                                                                   jfloatArray floatsArray) {
    if (env->GetArrayLength(intsArray) < kIntCount || env->GetArrayLength(floatsArray) < kFloatCount) return 0;
    jint ints[kIntCount];
    jfloat floats[kFloatCount];
    env->GetIntArrayRegion(intsArray, 0, kIntCount, ints);
    env->GetFloatArrayRegion(floatsArray, 0, kFloatCount, floats);

    vg::Reconstructor::Options o;
    o.processSize = ints[kProcessSize];
    o.orb.maxFeatures = std::max(200, static_cast<int>(ints[kMaxFeatures]));
    o.threads = ints[kThreads];
    o.fovDeg = floats[kFovDeg];
    vg::MappingParams& m = o.mapping;
    m.localBAKeyframes = std::max(0, static_cast<int>(ints[kLocalBAKeyframes]));
    m.localBAIterations = std::max(1, static_cast<int>(ints[kLocalBAIterations]));
    const int flags = ints[kFlags];
    m.backgroundBA = (flags & kBackgroundBA) != 0;
    m.refineFocal = (flags & kRefineFocal) != 0;
    m.refineDistortion = (flags & kRefineDistortion) != 0;
    if (!(flags & kFusion)) m.fuseNeighbours = 0;
    m.minConfirmingKeyframes = (flags & kConfirmPoints) ? 3 : 2;
    __android_log_print(ANDROID_LOG_INFO, "videogammetry",
                        "create: size %d, features %d, threads %d, fov %.1f, local BA %d x%d, flags 0x%x",
                        o.processSize, o.orb.maxFeatures, o.threads, o.fovDeg, m.localBAKeyframes,
                        m.localBAIterations, flags);
    return reinterpret_cast<jlong>(new Session(o));
}

JNIEXPORT void JNICALL Java_com_videogammetry_NativeBridge_destroy(JNIEnv*, jclass, jlong handle) {
    delete session(handle);
}

// Adds one frame (Bitmap must be ARGB_8888, i.e. RGBA bytes in memory).
JNIEXPORT jboolean JNICALL Java_com_videogammetry_NativeBridge_addFrame(JNIEnv* env, jclass, jlong handle,
                                                                        jobject bitmap, jdouble timestamp) {
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return JNI_FALSE;
    }
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) return JNI_FALSE;

    vg::FrameView frame;
    frame.data = static_cast<const std::uint8_t*>(pixels);
    frame.width = static_cast<int>(info.width);
    frame.height = static_cast<int>(info.height);
    frame.stride = static_cast<int>(info.stride);
    frame.format = vg::FrameView::Format::RGBA8;
    frame.timestamp = timestamp;
    Session* s = session(handle);
    s->reconstructor.addFrame(frame);
    ++s->frames;

    AndroidBitmap_unlockPixels(env, bitmap);
    return JNI_TRUE;
}

// Draws the map as seen from the current camera (its pose and field of view,
// scaled to fit the bitmap) into an RGBA_8888 bitmap: z-buffered square
// splats on a dark background. highlightTracked: points tracked in the
// current frame are drawn green. Returns the number of points drawn.
JNIEXPORT jint JNICALL Java_com_videogammetry_NativeBridge_renderView(JNIEnv* env, jclass, jlong handle,
                                                                      jobject bitmap, jint pointSize,
                                                                      jboolean highlightTracked) {
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return -1;
    }
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) return -1;
    const int w = static_cast<int>(info.width), h = static_cast<int>(info.height);
    const int stride = static_cast<int>(info.stride / 4);
    auto* px = static_cast<std::uint32_t*>(pixels);
    constexpr std::uint32_t kBackground = 0xFF1D1816u;  // ABGR: dark blue-grey
    for (int y = 0; y < h; ++y) std::fill_n(px + static_cast<std::size_t>(y) * stride, w, kBackground);

    Session* s = session(handle);
    const vg::Reconstructor& r = s->reconstructor;
    int drawn = 0;
    const vg::CameraIntrinsics& k = r.intrinsics();
    if (!r.keyframes().empty() && k.fx > 0) {
        const vg::Pose pose = r.state() == vg::Reconstructor::State::Tracking ? r.lastPose() : r.keyframes().back().pose;
        const double imgW = 2 * k.cx, imgH = 2 * k.cy;
        const double fit = std::min(w / imgW, h / imgH);
        const float f = static_cast<float>(k.fx * fit);
        const float cx = 0.5f * w, cy = 0.5f * h;
        float R[9], t[3];
        for (int i = 0; i < 9; ++i) R[i] = static_cast<float>(pose.R.m[i]);
        t[0] = static_cast<float>(pose.t.x);
        t[1] = static_cast<float>(pose.t.y);
        t[2] = static_cast<float>(pose.t.z);

        const std::vector<vg::MapPoint>& points = r.mapPoints();
        std::vector<char> tracked;
        if (highlightTracked) {
            tracked.assign(points.size(), 0);
            for (int id : r.lastTrackedPoints()) {
                if (id >= 0 && static_cast<std::size_t>(id) < tracked.size()) tracked[id] = 1;
            }
        }
        s->depth.assign(static_cast<std::size_t>(w) * h, std::numeric_limits<float>::max());
        const int size = std::max(1, static_cast<int>(pointSize));
        for (std::size_t i = 0; i < points.size(); ++i) {
            const vg::MapPoint& p = points[i];
            if (p.bad) continue;
            const float X = static_cast<float>(p.position.x), Y = static_cast<float>(p.position.y),
                        Z = static_cast<float>(p.position.z);
            const float z = R[6] * X + R[7] * Y + R[8] * Z + t[2];
            if (z < 1e-3f) continue;
            const float x = R[0] * X + R[1] * Y + R[2] * Z + t[0];
            const float yv = R[3] * X + R[4] * Y + R[5] * Z + t[1];
            const int u0 = static_cast<int>(f * x / z + cx) - (size - 1) / 2;
            const int v0 = static_cast<int>(f * yv / z + cy) - (size - 1) / 2;
            if (u0 + size <= 0 || v0 + size <= 0 || u0 >= w || v0 >= h) continue;
            const bool isTracked = highlightTracked && tracked[i];
            const std::uint32_t color =
                isTracked ? 0xFF6EFF46u
                          : (0xFF000000u | (static_cast<std::uint32_t>(p.color[2]) << 16) |
                             (static_cast<std::uint32_t>(p.color[1]) << 8) | p.color[0]);
            for (int v = std::max(v0, 0); v < std::min(v0 + size, h); ++v) {
                for (int u = std::max(u0, 0); u < std::min(u0 + size, w); ++u) {
                    float& d = s->depth[static_cast<std::size_t>(v) * w + u];
                    if (z < d) {
                        d = z;
                        px[static_cast<std::size_t>(v) * stride + u] = color;
                    }
                }
            }
            ++drawn;
        }
    }
    AndroidBitmap_unlockPixels(env, bitmap);
    return drawn;
}

JNIEXPORT jstring JNICALL Java_com_videogammetry_NativeBridge_status(JNIEnv* env, jclass, jlong handle) {
    const Session* s = session(handle);
    const vg::Reconstructor& r = s->reconstructor;
    const vg::Reconstructor::Stats& st = r.stats();
    std::size_t points = 0;
    for (const vg::MapPoint& p : r.mapPoints()) points += p.bad ? 0 : 1;
    char text[384];
    std::snprintf(text, sizeof(text),
                  "%s | %zu keyframes | %zu points | tracked %d | lost %d\n"
                  "field of view %.1f deg, k1 %.3f (%d refinements) | background BA passes %d",
                  stateName(r.state()), r.keyframes().size(), points, r.lastTrackedCount(), st.lostFrames,
                  r.fovDeg(), r.distortion(), st.focalUpdates, st.backgroundPasses);
    return env->NewStringUTF(text);
}

JNIEXPORT void JNICALL Java_com_videogammetry_NativeBridge_finish(JNIEnv*, jclass, jlong handle) {
    session(handle)->reconstructor.finish();
}

// Writes the point cloud; returns the number of points, or -1 on failure.
JNIEXPORT jint JNICALL Java_com_videogammetry_NativeBridge_writePly(JNIEnv* env, jclass, jlong handle, jstring path) {
    const char* p = env->GetStringUTFChars(path, nullptr);
    const vg::PointCloud cloud = session(handle)->reconstructor.pointCloud();
    const bool ok = vg::writePly(cloud, p);
    env->ReleaseStringUTFChars(path, p);
    return ok ? static_cast<jint>(cloud.points.size()) : -1;
}

}  // extern "C"
