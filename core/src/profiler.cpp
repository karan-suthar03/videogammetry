#include "videogammetry/profiler.h"

#include <algorithm>

namespace vg {

void Profiler::add(const char* name, double ms) {
    auto it = std::find_if(stages_.begin(), stages_.end(),
                           [name](const Stage& s) { return s.name == name; });
    if (it == stages_.end()) {
        stages_.push_back({name});
        it = std::prev(stages_.end());
    }
    it->totalMs += ms;
    it->maxMs = std::max(it->maxMs, ms);
    ++it->count;
}

}  // namespace vg
