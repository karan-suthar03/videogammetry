#include "videogammetry/thread_pool.h"

#include <algorithm>

namespace vg {

namespace {
thread_local bool tInsideChunk = false;
}

int ThreadPool::hardwareThreads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n > 0 ? static_cast<int>(n) : 1;
}

ThreadPool::ThreadPool(int threads) {
    const int total = threads > 0 ? threads : hardwareThreads();
    workers_.reserve(static_cast<std::size_t>(total - 1));
    for (int i = 1; i < total; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : workers_) t.join();
}

void ThreadPool::runChunks(Job& job) {
    tInsideChunk = true;
    while (true) {
        const int start = job.next.fetch_add(job.grain, std::memory_order_relaxed);
        if (start >= job.end) break;
        (*job.fn)(start, std::min(start + job.grain, job.end));
        if (job.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(mutex_);
            done_.notify_all();
        }
    }
    tInsideChunk = false;
}

void ThreadPool::workerLoop() {
    unsigned long long seen = 0;
    while (true) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            job = job_;  // keeps the job alive even if the caller has moved on
        }
        if (job) runChunks(*job);
    }
}

void ThreadPool::parallelFor(int begin, int end, int grain,
                             const std::function<void(int, int)>& fn) {
    if (begin >= end) return;
    grain = std::max(grain, 1);
    const int chunks = (end - begin + grain - 1) / grain;

    if (workers_.empty() || chunks == 1 || tInsideChunk) {
        for (int s = begin; s < end; s += grain) fn(s, std::min(s + grain, end));
        return;
    }

    auto job = std::make_shared<Job>();
    job->fn = &fn;
    job->end = end;
    job->grain = grain;
    job->next.store(begin, std::memory_order_relaxed);
    job->remaining.store(chunks, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = job;
        ++generation_;
    }
    wake_.notify_all();

    runChunks(*job);

    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [&] { return job->remaining.load(std::memory_order_acquire) == 0; });
    job_.reset();
}

}  // namespace vg
