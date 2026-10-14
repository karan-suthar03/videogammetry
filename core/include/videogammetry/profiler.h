#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace vg {

// Accumulates wall-clock time per named pipeline stage, in first-use order.
class Profiler {
public:
    struct Stage {
        std::string name;
        double totalMs = 0;
        double maxMs = 0;
        long count = 0;
        double avgMs() const { return count ? totalMs / count : 0.0; }
    };

    // Times the enclosing block: `auto t = profiler.scope("orb");`
    class Scope {
    public:
        Scope(Profiler& p, const char* name)
            : profiler_(p), name_(name), start_(std::chrono::steady_clock::now()) {}
        ~Scope() {
            const auto end = std::chrono::steady_clock::now();
            profiler_.add(name_, std::chrono::duration<double, std::milli>(end - start_).count());
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        Profiler& profiler_;
        const char* name_;
        std::chrono::steady_clock::time_point start_;
    };

    Scope scope(const char* name) { return Scope(*this, name); }
    void add(const char* name, double ms);

    const std::vector<Stage>& stages() const { return stages_; }

private:
    std::vector<Stage> stages_;
};

}  // namespace vg
