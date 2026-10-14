#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vg {

struct Point {
    float x = 0, y = 0, z = 0;
    std::uint8_t r = 255, g = 255, b = 255;
};

struct PointCloud {
    std::vector<Point> points;
};

// Writes a binary little-endian PLY file. Returns false on I/O failure.
bool writePly(const PointCloud& cloud, const std::string& path);

}  // namespace vg
