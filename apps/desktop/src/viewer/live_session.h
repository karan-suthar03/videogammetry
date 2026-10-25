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

class CameraReader;

// What the viewer draws: a copy of the reconstruction state at one frame.
struct LiveSnapshot {
    long version = 0;  // increases with every publish

    std::vector<vg::Point> points;       // current map (rejected points left out)
    std::vector<std::uint8_t> tracked;   // per point: seen in the current frame
    std::vector<float> noise;            // per point: MapPoint::noise (< 0 = unknown)
    std::vector<vg::Pose> keyframes;     // world -> camera
    vg::Pose current;                    // pose of the current frame
    bool currentValid = false;           // false while initializing or lost
    double sceneScale = 1.0;             // typical camera-to-point distance (map units)

    // Camera model of the processed image, for drawing frustums.
    double focal = 1.0;
    int imageWidth = 16, imageHeight = 9;
    double fovDeg = 0, startFovDeg = 0;  // current estimate (self-calibrated) and the starting guess
    int focalUpdates = 0;
    double k1 = 0;  // radial distortion estimate

    std::shared_ptr<const std::vector<vg::Point>> dense;  // dense cloud (may be null)

    RgbImage thumbnail;  // current video frame with tracked keypoints drawn

    // Status.
    long frame = 0, frameCount = 0;  // frameCount is an estimate, may be 0
    double timestamp = 0.0;
    vg::Reconstructor::State state = vg::Reconstructor::State::Initializing;
    int trackedCount = 0;
    int maps = 0;
    double processingFps = 0.0;
    std::string extractor;  // where features are extracted, e.g. "GPU (device name, Vulkan)"
    bool finished = false;  // global bundle adjustment done
    std::string message;    // error or summary line
};

// Runs the reconstruction of one video (or a live camera) on background threads (decode ->
// parallel feature extraction -> tracking and mapping) and publishes
// snapshots for the viewer. Playback can be paced to the video's own frame
// rate or run as fast as possible.
class LiveSession {
public:
    struct Settings {
        vg::Reconstructor::Options options;
        bool realTime = false;  // pace to the video's frame rate (default: as fast as possible)
        int thumbnailSize = 420;  // long side of the video thumbnail, pixels
        bool gpu = true;          // extract features on the GPU when there is one (Vulkan)
    };

    // `cameraId` set: reads that camera (CameraReader) and `path` is just its
    // name; the capture runs until endCapture().
    LiveSession(std::string path, Settings settings, std::wstring cameraId = {});
    ~LiveSession();  // stops and joins
    LiveSession(const LiveSession&) = delete;
    LiveSession& operator=(const LiveSession&) = delete;

    const std::string& path() const { return path_; }

    // Latest published state (never null; compare `version` to spot updates).
    std::shared_ptr<const LiveSnapshot> latest() const;

    void setPaused(bool paused) { paused_ = paused; }
    bool paused() const { return paused_; }
    void setRealTime(bool realTime) { realTime_ = realTime && cameraId_.empty(); }
    bool realTime() const { return realTime_; }
    bool camera() const { return !cameraId_.empty(); }

    // Stops reading frames here (ends a camera capture); the map is then
    // refined and finished as at the end of a video.
    void endCapture();

    // True once the video is done and the final map is ready.
    bool finished() const { return done_; }
    // Writes the final point cloud; only valid when finished().
    bool savePly(const std::string& path, std::size_t& pointsWritten) const;

private:
    void run();
    void publish(const vg::FrameView* frame, double processingFps, bool finished, const std::string& message);

    std::string path_;
    Settings settings_;
    std::wstring cameraId_;
    std::mutex cameraMutex_;
    CameraReader* camera_ = nullptr;  // while capturing
    std::unique_ptr<vg::Reconstructor> reconstructor_;

    std::atomic<bool> stop_{false}, endCapture_{false}, paused_{false}, realTime_{false}, done_{false};
    long frame_ = 0, frameCount_ = 0;
    double timestamp_ = 0.0;

    mutable std::mutex mutex_;
    std::shared_ptr<const LiveSnapshot> snapshot_;
    long version_ = 0;
    RgbImage thumbnail_;  // last frame's, kept for snapshots after the video ends
    std::string extractor_;

    std::thread thread_;
};
