#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

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
        int processWidth = 960;  // frames wider than this are downscaled first; 0 = full size
        int threads = 0;         // worker threads: 0 = all hardware threads, 1 = single-threaded
        double fovDeg = 65.0;    // assumed horizontal field of view (camera is uncalibrated)
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
    };

    Reconstructor() : Reconstructor(Options{}) {}
    explicit Reconstructor(const Options& options);

    void addFrame(const FrameView& frame);

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

    void tryInitialize(const FrameView& frame);
    bool track(const FrameView& frame);
    void addKeyframe(const FrameView& frame);
    int searchByProjection(const Pose& pose, float radius, const std::vector<int>& localPoints,
                           std::vector<int>& pointOfKeypoint) const;
    int solveAndRefine(const Pose& prior, const std::vector<int>& localPoints, std::vector<int>& pointOfKeypoint,
                       Pose& pose);
    std::vector<int> localPoints() const;
    int triangulateBetween(Keyframe& older, Keyframe& newer);
    double medianSceneDepth(const Keyframe& kf) const;
    void startNewMap();
    void sampleColors(const FrameView& frame, FrameData& data) const;

    Options options_;
    std::unique_ptr<ThreadPool> pool_;  // null when single-threaded; declared before orb_
    OrbExtractor orb_;
    Profiler profiler_;
    CameraIntrinsics intrinsics_;
    float processScale_ = 1.0f;
    int imageWidth_ = 0, imageHeight_ = 0;  // processed image size
    Image gray_, processed_;

    State state_ = State::Initializing;
    Stats stats_;
    std::size_t frameCount_ = 0;

    // Current frame.
    std::vector<Keypoint> keypoints_;
    std::vector<Descriptor> descriptors_;
    std::vector<int> trackedPoints_;
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
    std::vector<MapPoint> bestPoints_;
    std::vector<Vec3> bestTrajectory_;
};

}  // namespace vg
