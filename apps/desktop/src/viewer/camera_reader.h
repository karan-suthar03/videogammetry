#pragma once

#include <memory>
#include <string>
#include <vector>

#include "videogammetry/frame.h"

// Live camera frames (Windows Media Foundation), converted to RGB8 and
// optionally downscaled. A capture thread keeps only the newest frame, so
// read() never hands out a stale one when processing is slower than the camera.
class CameraReader {
public:
    struct Device {
        std::string name;  // friendly name (UTF-8)
        std::wstring id;   // symbolic link
    };
    static std::vector<Device> list();

    CameraReader();
    ~CameraReader();
    CameraReader(const CameraReader&) = delete;
    CameraReader& operator=(const CameraReader&) = delete;

    // Picks the camera mode whose long side is closest above `longSide`
    // (0 = largest) and downscales frames to it. Returns false and fills `error`.
    bool open(const std::wstring& id, int longSide, std::string& error);

    // Waits for a frame newer than the last one returned; the view stays
    // valid until the next call. False once closed or on a capture error.
    bool read(vg::FrameView& frame);
    void close();  // wakes read()
    const std::string& lastError() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
