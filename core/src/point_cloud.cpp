#include "videogammetry/point_cloud.h"

#include <fstream>

namespace vg {

bool writePly(const PointCloud& cloud, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    out << "ply\n"
        << "format binary_little_endian 1.0\n"
        << "element vertex " << cloud.points.size() << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "end_header\n";

    // All supported targets (x86-64, ARM64) are little-endian.
    for (const Point& p : cloud.points) {
        out.write(reinterpret_cast<const char*>(&p.x), sizeof(float));
        out.write(reinterpret_cast<const char*>(&p.y), sizeof(float));
        out.write(reinterpret_cast<const char*>(&p.z), sizeof(float));
        out.put(static_cast<char>(p.r));
        out.put(static_cast<char>(p.g));
        out.put(static_cast<char>(p.b));
    }
    return static_cast<bool>(out);
}

}  // namespace vg
