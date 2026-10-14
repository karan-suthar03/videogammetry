package com.videogammetry;

// Java side of the JNI bridge to the C++ core (see src/main/cpp/native_bridge.cpp).
public final class NativeBridge {

    static {
        System.loadLibrary("videogammetry_jni");
    }

    private NativeBridge() {}

    public static native String version();
}
