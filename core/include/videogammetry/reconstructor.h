#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "videogammetry/features.h"
#include "videogammetry/bundle_adjustment.h"
#include "videogammetry/frame.h"
#include "videogammetry/image.h"
#include "videogammetry/map.h"
#include "videogammetry/matcher.h"
#include "videogammetry/orb.h"
#include "videogammetry/point_cloud.h"
#include "videogammetry/pnp.h"
#include "videogammetry/profiler.h"
#include "videogammetry/thread_pool.h"
#include "videogammetry/two_view.h"

namespace vg {

// Entry point of the pipeline. Apps push frames in order, then read the cloud.
//
// Monocular visual odometry that makes no assumption about how the camera
// moves:
//  - Initializing: match frames against a reference frame until a pair has
//    enough parallax, then build the first map from their relative pose.
//  - Tracking: predict the pose (constant velocity), project nearby map
//    points, match around the projections and solve the pose with PnP
//    (2D-3D, works for rotation, translation, or both). Falls back to wider
//    searches for fast motion.
//  - Keyframes are added when the view has changed enough; new points are
//    triangulated between keyframes with known poses.
//  - Lost: retry against the last keyframe, then start a new map.
class Reconstructor {
public:
    struct Options {
        int processSize = 0;     // long side of the frame is downscaled to this; 0 = full size
        int threads = 0;         // worker threads: 0 = all hardware threads, 1 = single-threaded
        double fovDeg = 65.0;    // assumed field of view across the long side (camera is uncalibrated)
        OrbParams orb;
        MatchParams match;       // used for keyframe-to-keyframe and initialization matching
        TwoViewParams twoView;
        PnPParams pnp;
        MappingParams mapping;
    };

    enum class State { Initializing, Tracking, Lost };

    struct Stats {
        int trackedFrames = 0;
        int lostFrames = 0;
        int maps = 0;            // maps started (> 1 means tracking was lost for good)
        int wideSearches = 0;    // frames that needed the wider projection search
        int keyframeSearches = 0;// frames that needed full matching against the last keyframe
        int localBAs = 0;
        double localBARmsBefore = 0, localBARmsAfter = 0;  // mean RMS reprojection error (px)
        int localBAOutliers = 0;                           // observations removed
        int pointsCreated = 0;
        int pointsCulled = 0;    // new points removed for lack of confirmation
        BAReport globalBA;                                 // set by finish()
        int globalBACameras = 0, globalBAPoints = 0;
    };

    Reconstructor() : Reconstructor(Options{}) {}
    explicit Reconstructor(const Options& options);

    // Processes the next frame: extracts features here, then tracks/maps.
    void addFrame(const FrameView& frame);
    // Same with features computed elsewhere (e.g. FeaturePipeline, which
    // extracts many frames in parallel). Frames must arrive in order.
    void addFeatures(FrameFeatures&& features);
    // Call after the last frame: global bundle adjustment over the whole map.
    void finish();

    State state() const { return state_; }
    const Stats& stats() const { return stats_; }

    // Current map (the most recent one if tracking had to restart).
    const std::vector<MapPoint>& mapPoints() const { return points_; }
    const std::vector<Keyframe>& keyframes() const { return keyframes_; }
    // Colored cloud of the largest map built so far (maps are not merged).
    PointCloud pointCloud() const;
    // Camera centers of the keyframes of that map.
    std::vector<Vec3> trajectory() const;

    std::size_t frameCount() const { return frameCount_; }
    const Profiler& profiler() const { return profiler_; }

    // Most recent frame: features (processed-image pixels), map point tracked
    // by each keypoint (-1 if none), pose, and whether it became a keyframe.
    const std::vector<Keypoint>& lastKeypoints() const { return keypoints_; }
    const std::vector<int>& lastTrackedPoints() const { return trackedPoints_; }
    int lastTrackedCount() const { return trackedCount_; }
    const Pose& lastPose() const { return pose_; }
    bool lastWasKeyframe() const { return lastWasKeyframe_; }

    // Camera model of the processed image (valid after the first frame).
    const CameraIntrinsics& intrinsics() const { return intrinsics_; }
    // Input-frame pixels per processed-image pixel (>= 1).
    float processScale() const { return processScale_; }
    // Threads used per frame (including the caller).
    int threadCount() const { return pool_ ? pool_->size() : 1; }

private:
    struct FrameData {
        long index = -1;
        std::vector<Keypoint> keypoints;
        std::vector<Descriptor> descriptors;
        std::vector<std::array<std::uint8_t, 3>> colors;
    };

    void tryInitialize();
    bool track();
    void addKeyframe();
    int searchByProjection(const Pose& pose, float radius, const std::vector<int>& localPoints,
                           std::vector<int>& pointOfKeypoint) const;
    int solveAndRefine(const Pose& prior, const std::vector<int>& localPoints, std::vector<int>& pointOfKeypoint,
                       Pose& pose);
    // Keyframes sharing points with keyframe k: (keyframe, shared points),
    // most shared first.
    std::vector<std::pair<int, int>> covisibleKeyframes(int k) const;
    // Local map for tracking: keyframes covisible with the last frame, and their points.
    std::vector<int> localKeyframes() const;
    std::vector<int> localPoints() const;
    int triangulateBetween(int older, int newer);  // keyframe indices
    // Map bookkeeping: keyframe pointIds and point observations stay in sync.
    int createPoint(MapPoint point, PointObservation a, PointObservation b);
    void addObservation(int pointId, PointObservation obs);
    void removeObservation(int pointId, PointObservation obs);  // the point goes bad below 2 views
    void setBad(int pointId);
    // Removes new points that later keyframes did not confirm.
    void cullRecentPoints();
    double medianSceneDepth(const Keyframe& kf) const;
    // Bundle-adjusts the given keyframes with all their points; other
    // keyframes that see those points are held fixed. Applies the result and
    // removes rejected observations.
    BAReport adjust(const std::vector<int>& keyframes, int iterations);
    void startNewMap();

    Options options_;
    std::unique_ptr<ThreadPool> pool_;  // null when single-threaded; declared before extractor_
    FeatureExtractor extractor_;        // used by addFrame()
    Profiler profiler_;
    CameraIntrinsics intrinsics_;
    float processScale_ = 1.0f;
    int imageWidth_ = 0, imageHeight_ = 0;  // processed image size

    State state_ = State::Initializing;
    Stats stats_;
    std::size_t frameCount_ = 0;

    // Current frame.
    std::vector<Keypoint> keypoints_;
    std::vector<Descriptor> descriptors_;
    std::vector<std::array<std::uint8_t, 3>> colors_;
    std::vector<int> trackedPoints_;
    std::vector<int> previousTracked_;  // map point ids tracked in the previous frame
    int trackedCount_ = 0;
    Pose pose_;
    bool lastWasKeyframe_ = false;

    // Motion model and tracking state.
    Pose previousPose_;
    Pose velocity_;          // previous camera -> current camera
    bool hasVelocity_ = false;
    long lastKeyframeFrame_ = -1;
    int lostCount_ = 0;

    // Initialization.
    FrameData reference_;

    // Current map, and the best (largest) finished map.
    std::vector<Keyframe> keyframes_;
    std::vector<MapPoint> points_;
    std::vector<int> recentPoints_;  // points still on probation
    std::vector<MapPoint> bestPoints_;
    std::vector<Vec3> bestTrajectory_;
};

}  // namespace vg
