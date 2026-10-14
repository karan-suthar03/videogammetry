// JNI bridge between the Android app and the C++ core.
// A session wraps one Reconstructor; frames arrive as RGBA_8888 Bitmaps.

#include <jni.h>

#include <android/bitmap.h>
#include <android/log.h>

#include <cstdio>
#include <memory>
#include <string>

#include "videogammetry/reconstructor.h"
#include "videogammetry/version.h"

namespace {

struct Session {
    explicit Session(const vg::Reconstructor::Options& options) : reconstructor(options) {}
    vg::Reconstructor reconstructor;
    long frames = 0;
};

Session* session(jlong handle) { return reinterpret_cast<Session*>(handle); }

const char* stateName(vg::Reconstructor::State s) {
    switch (s) {
        case vg::Reconstructor::State::Initializing: return "initializing";
        case vg::Reconstructor::State::Tracking: return "tracking";
        case vg::Reconstructor::State::Lost: return "lost";
    }
    return "?";
}

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL Java_com_videogammetry_NativeBridge_version(JNIEnv* env, jclass) {
    return env->NewStringUTF(vg::version());
}

JNIEXPORT jlong JNICALL Java_com_videogammetry_NativeBridge_create(JNIEnv*, jclass, jint processSize, jdouble fovDeg) {
    vg::Reconstructor::Options options;
    options.processSize = processSize;
    options.fovDeg = fovDeg;
    return reinterpret_cast<jlong>(new Session(options));
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

JNIEXPORT jstring JNICALL Java_com_videogammetry_NativeBridge_status(JNIEnv* env, jclass, jlong handle) {
    const Session* s = session(handle);
    const vg::Reconstructor& r = s->reconstructor;
    char text[256];
    std::snprintf(text, sizeof(text), "%s | %zu keyframes | %zu map points | tracked %d | lost frames %d",
                  stateName(r.state()), r.keyframes().size(), r.mapPoints().size(), r.lastTrackedCount(),
                  r.stats().lostFrames);
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
