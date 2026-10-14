#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace vg {

// Fixed-size worker pool for data-parallel loops. Portable (std::thread only):
// sized from std::thread::hardware_concurrency() at runtime, no affinity or
// OS-specific calls. The calling thread also does work, so a pool of N
// threads runs N-1 workers.
class ThreadPool {
public:
    // threads <= 0: use all hardware threads.
    explicit ThreadPool(int threads = 0);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Total threads working on a parallelFor, including the caller.
    int size() const { return static_cast<int>(workers_.size()) + 1; }

    // Calls fn(chunkBegin, chunkEnd) over [begin, end) in chunks of `grain`
    // and blocks until all chunks are done. Chunks may run in any order on
    // any thread, so fn must only write to per-index outputs. Calls made from
    // inside a running chunk execute serially on the calling thread.
    void parallelFor(int begin, int end, int grain, const std::function<void(int, int)>& fn);

    static int hardwareThreads();

private:
    struct Job {
        const std::function<void(int, int)>* fn = nullptr;
        int end = 0;
        int grain = 1;
        std::atomic<int> next{0};
        std::atomic<int> remaining{0};  // chunks not yet finished
    };

    void workerLoop();
    void runChunks(Job& job);

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    std::shared_ptr<Job> job_;
    unsigned long long generation_ = 0;
    bool stop_ = false;
};

// Runs fn over [begin, end): on `pool` when given, otherwise serially.
inline void parallelFor(ThreadPool* pool, int begin, int end, int grain,
                        const std::function<void(int, int)>& fn) {
    if (pool) {
        pool->parallelFor(begin, end, grain, fn);
    } else if (begin < end) {
        fn(begin, end);
    }
}

}  // namespace vg
