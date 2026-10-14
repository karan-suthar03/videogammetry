package com.videogammetry;

import android.graphics.Bitmap;

// Java side of the JNI bridge to the C++ core (see src/main/cpp/native_bridge.cpp).
public final class NativeBridge {

    static {
        System.loadLibrary("videogammetry_jni");
    }

    private NativeBridge() {}

    public static native String version();

    // A reconstruction session; release with destroy().
    public static native long create(int processSize, double fovDeg);

    public static native void destroy(long session);

    // Bitmap must be ARGB_8888.
    public static native boolean addFrame(long session, Bitmap frame, double timestampSec);

    public static native String status(long session);

    // Global bundle adjustment; call after the last frame.
    public static native void finish(long session);

    // Returns the number of points written, or -1 on failure.
    public static native int writePly(long session, String path);
}
