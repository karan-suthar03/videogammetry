#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "image_io.h"
#include "videogammetry/linalg.h"
#include "videogammetry/point_cloud.h"
#include "videogammetry/reconstructor.h"

// What the viewer draws: a copy of the reconstruction state at one frame.
struct LiveSnapshot {
    long version = 0;  // increases with every publish

    std::vector<vg::Point> points;       // current map (rejected points left out)
    std::vector<std::uint8_t> tracked;   // per point: seen in the current frame
    std::vector<vg::Pose> keyframes;     // world -> camera
    vg::Pose current;                    // pose of the current frame
    bool currentValid = false;           // false while initializing or lost
    double sceneScale = 1.0;             // typical camera-to-point distance (map units)

    // Camera model of the processed image, for drawing frustums.
    double focal = 1.0;
    int imageWidth = 16, imageHeight = 9;

    RgbImage thumbnail;  // current video frame with tracked keypoints drawn

    // Status.
    long frame = 0, frameCount = 0;  // frameCount is an estimate, may be 0
    double timestamp = 0.0;
    vg::Reconstructor::State state = vg::Reconstructor::State::Initializing;
    int trackedCount = 0;
    int maps = 0;
    double processingFps = 0.0;
    bool finished = false;  // global bundle adjustment done
    std::string message;    // error or summary line
};

// Runs the reconstruction of one video on background threads (decode ->
// parallel feature extraction -> tracking and mapping) and publishes
// snapshots for the viewer. Playback can be paced to the video's own frame
// rate or run as fast as possible.
class LiveSession {
public:
    struct Settings {
        vg::Reconstructor::Options options;
        bool realTime = true;  // pace to the video's frame rate
        int thumbnailSize = 420;  // long side of the video thumbnail, pixels
    };

    LiveSession(std::string path, Settings settings);
    ~LiveSession();  // stops and joins
    LiveSession(const LiveSession&) = delete;
    LiveSession& operator=(const LiveSession&) = delete;

    const std::string& path() const { return path_; }

    // Latest published state (never null; compare `version` to spot updates).
    std::shared_ptr<const LiveSnapshot> latest() const;

    void setPaused(bool paused) { paused_ = paused; }
    bool paused() const { return paused_; }
    void setRealTime(bool realTime) { realTime_ = realTime; }
    bool realTime() const { return realTime_; }

    // True once the video is done and the final map is ready.
    bool finished() const { return done_; }
    // Writes the final point cloud; only valid when finished().
    bool savePly(const std::string& path, std::size_t& pointsWritten) const;

private:
    void run();
    void publish(const vg::FrameView* frame, double processingFps, bool finished, const std::string& message);

    std::string path_;
    Settings settings_;
    std::unique_ptr<vg::Reconstructor> reconstructor_;

    std::atomic<bool> stop_{false}, paused_{false}, realTime_{true}, done_{false};
    long frame_ = 0, frameCount_ = 0;
    double timestamp_ = 0.0;

    mutable std::mutex mutex_;
    std::shared_ptr<const LiveSnapshot> snapshot_;
    long version_ = 0;
    RgbImage thumbnail_;  // last frame's, kept for snapshots after the video ends

    std::thread thread_;
};
