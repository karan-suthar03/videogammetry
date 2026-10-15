package com.videogammetry;

import android.content.SharedPreferences;

// Tuning options of a run, saved between app starts. toInts() / toFloats()
// must match IntOption / FloatOption / Flag in native_bridge.cpp.
final class Settings {
    // Processing
    int processSize = 960;       // long side in pixels; 0 = full resolution
    int maxFeatures = 2000;      // ORB features per frame
    int frameStep = 1;           // process every N-th frame
    int threads = 0;             // worker threads, 0 = all cores
    float fovDeg = 65f;          // starting guess, refined from the video when enabled
    // Mapping
    int localBAKeyframes = 10;
    int localBAIterations = 10;
    boolean backgroundBA = true;
    boolean refineFocal = true;
    boolean refineDistortion = true;
    boolean fusion = true;
    boolean confirmPoints = true;
    // Live view
    boolean liveView = true;
    int viewFps = 20;
    int pointSize = 3;
    boolean highlightTracked = true;

    int[] toInts() {
        int flags = (backgroundBA ? 1 : 0) | (refineFocal ? 2 : 0) | (refineDistortion ? 4 : 0) | (fusion ? 8 : 0)
                | (confirmPoints ? 16 : 0);
        return new int[] {processSize, maxFeatures, threads, localBAKeyframes, localBAIterations, flags};
    }

    float[] toFloats() {
        return new float[] {fovDeg};
    }

    void load(SharedPreferences p) {
        processSize = p.getInt("processSize", processSize);
        maxFeatures = p.getInt("maxFeatures", maxFeatures);
        frameStep = p.getInt("frameStep", frameStep);
        threads = p.getInt("threads", threads);
        fovDeg = p.getFloat("fovDeg", fovDeg);
        localBAKeyframes = p.getInt("localBAKeyframes", localBAKeyframes);
        localBAIterations = p.getInt("localBAIterations", localBAIterations);
        backgroundBA = p.getBoolean("backgroundBA", backgroundBA);
        refineFocal = p.getBoolean("refineFocal", refineFocal);
        refineDistortion = p.getBoolean("refineDistortion", refineDistortion);
        fusion = p.getBoolean("fusion", fusion);
        confirmPoints = p.getBoolean("confirmPoints", confirmPoints);
        liveView = p.getBoolean("liveView", liveView);
        viewFps = p.getInt("viewFps", viewFps);
        pointSize = p.getInt("pointSize", pointSize);
        highlightTracked = p.getBoolean("highlightTracked", highlightTracked);
    }

    void save(SharedPreferences p) {
        p.edit()
                .putInt("processSize", processSize)
                .putInt("maxFeatures", maxFeatures)
                .putInt("frameStep", frameStep)
                .putInt("threads", threads)
                .putFloat("fovDeg", fovDeg)
                .putInt("localBAKeyframes", localBAKeyframes)
                .putInt("localBAIterations", localBAIterations)
                .putBoolean("backgroundBA", backgroundBA)
                .putBoolean("refineFocal", refineFocal)
                .putBoolean("refineDistortion", refineDistortion)
                .putBoolean("fusion", fusion)
                .putBoolean("confirmPoints", confirmPoints)
                .putBoolean("liveView", liveView)
                .putInt("viewFps", viewFps)
                .putInt("pointSize", pointSize)
                .putBoolean("highlightTracked", highlightTracked)
                .apply();
    }

    void reset() {
        Settings d = new Settings();
        processSize = d.processSize;
        maxFeatures = d.maxFeatures;
        frameStep = d.frameStep;
        threads = d.threads;
        fovDeg = d.fovDeg;
        localBAKeyframes = d.localBAKeyframes;
        localBAIterations = d.localBAIterations;
        backgroundBA = d.backgroundBA;
        refineFocal = d.refineFocal;
        refineDistortion = d.refineDistortion;
        fusion = d.fusion;
        confirmPoints = d.confirmPoints;
        liveView = d.liveView;
        viewFps = d.viewFps;
        pointSize = d.pointSize;
        highlightTracked = d.highlightTracked;
    }
}
