#pragma once
#include <cstdint>
#include <mutex>
#include <vector>

// Common interface for sample sources feeding the scope. A source runs its own
// acquisition thread and exposes a non-blocking drain() the render thread calls once
// per frame. Each drained "chunk" is one record (a trace in trigger mode; appended
// to the ring in roll mode).
class ISource {
public:
    virtual ~ISource() = default;
    virtual void start() = 0;
    virtual void stop()  = 0;
    // Move all pending chunks into `out` (cleared first). Returns the count.
    virtual size_t drain(std::vector<std::vector<float>>& out) = 0;
    // Native sample rate in Hz (sets the time axis in roll mode).
    virtual float    sample_rate() const = 0;
    virtual uint64_t produced()    const = 0;
    virtual uint64_t dropped()     const = 0;
};

// Thread-safe bounded chunk queue shared by source implementations. The producer
// pushes chunks; the consumer swaps them all out. If the consumer falls behind,
// the oldest chunks are dropped to bound latency and memory.
class SampleQueue {
public:
    explicit SampleQueue(size_t max_chunks = 512) : max_chunks_(max_chunks) {}

    void push(std::vector<float>&& chunk) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (queue_.size() >= max_chunks_) {
            size_t drop = queue_.size() - max_chunks_ + 1;
            queue_.erase(queue_.begin(), queue_.begin() + drop);
            dropped_ += drop;
        }
        queue_.push_back(std::move(chunk));
        ++produced_;
    }

    size_t drain(std::vector<std::vector<float>>& out) {
        out.clear();
        std::lock_guard<std::mutex> lock(mtx_);
        out.swap(queue_);
        return out.size();
    }

    uint64_t produced() const { std::lock_guard<std::mutex> l(mtx_); return produced_; }
    uint64_t dropped()  const { std::lock_guard<std::mutex> l(mtx_); return dropped_; }

private:
    mutable std::mutex              mtx_;
    std::vector<std::vector<float>> queue_;
    size_t                         max_chunks_;
    uint64_t                       produced_ = 0;
    uint64_t                       dropped_  = 0;
};
