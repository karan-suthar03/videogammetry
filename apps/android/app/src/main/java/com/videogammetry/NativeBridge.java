package com.videogammetry;

import android.graphics.Bitmap;

// Java side of the JNI bridge to the C++ core (see src/main/cpp/native_bridge.cpp).
// All calls for one session must come from the same thread.
public final class NativeBridge {

    static {
        System.loadLibrary("videogammetry_jni");
    }

    private NativeBridge() {}

    public static native String version();

    // A reconstruction session; release with destroy(). ints / floats are
    // laid out as in Settings.toInts() / Settings.toFloats().
    public static native long create(int[] ints, float[] floats);

    public static native void destroy(long session);

    // Bitmap must be ARGB_8888.
    public static native boolean addFrame(long session, Bitmap frame, double timestampSec);

    // Camera frame as YUV_420_888 planes (direct buffers from android.media.Image);
    // subsampled natively toward processSize (long side).
    public static native boolean addFrameYuv(long session, java.nio.ByteBuffer y, java.nio.ByteBuffer u,
                                             java.nio.ByteBuffer v, int width, int height, int yRowStride,
                                             int uvRowStride, int uvPixelStride, int processSize,
                                             double timestampSec);

    // Draws the point cloud from the current camera into an ARGB_8888 bitmap;
    // returns the number of points drawn.
    public static native int renderView(long session, Bitmap target, int pointSize, boolean highlightTracked);

    public static native String status(long session);

    // Global bundle adjustment; call after the last frame.
    public static native void finish(long session);

    // Returns the number of points written, or -1 on failure.
    public static native int writePly(long session, String path);
}
