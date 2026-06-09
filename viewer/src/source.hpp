#pragma once
#include <cstdint>
#include <mutex>
#include <vector>

// One delivered record of one channel, carrying enough metadata for the render
// side to reassemble a gap-aware sample timeline. `seq` is an absolute frame index
// assigned by the producer: a gap in the seq of drained frames is exactly the
// number of frames lost (in the network for the Red Pitaya, or in SampleQueue when
// the consumer falls behind). `timestamp` is in source ticks at the first sample
// (Red Pitaya: 125 MHz adc_clk; synthetic/audio sources synthesize 1 tick/sample).
// `samples_per_frame` is fixed within a session.
struct Frame {
    int64_t            seq              = 0;
    int64_t            timestamp        = 0;
    uint32_t           samples_per_frame = 0;
    std::vector<float> samples;
};

// Common interface for sample sources feeding the scope. A source runs its own
// acquisition thread and exposes a non-blocking drain() the render thread calls once
// per frame. Each drained Frame is one record (a trace in trigger mode; appended to
// the ring in roll mode).
class ISource {
public:
    virtual ~ISource() = default;
    virtual void start() = 0;
    virtual void stop()  = 0;
    // Move all pending frames into `out` (cleared first). Returns the count.
    virtual size_t drain(std::vector<Frame>& out) = 0;
    // Native sample rate in Hz (sets the time axis in roll mode).
    virtual float    sample_rate() const = 0;
    virtual uint64_t produced()    const = 0;
    virtual uint64_t dropped()     const = 0;
};

// Thread-safe bounded frame queue shared by source implementations. The producer
// pushes frames; the consumer swaps them all out. If the consumer falls behind, the
// oldest frames are dropped to bound latency and memory. Because each frame's `seq`
// is assigned before it is pushed, a queue-drop leaves a seq gap that the render
// side treats as missing samples — the same way a network drop is handled.
class SampleQueue {
public:
    explicit SampleQueue(size_t max_frames = 512) : max_frames_(max_frames) {}

    void push(Frame&& frame) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (queue_.size() >= max_frames_) {
            size_t drop = queue_.size() - max_frames_ + 1;
            queue_.erase(queue_.begin(), queue_.begin() + drop);
            dropped_ += drop;
        }
        queue_.push_back(std::move(frame));
        ++produced_;
    }

    size_t drain(std::vector<Frame>& out) {
        out.clear();
        std::lock_guard<std::mutex> lock(mtx_);
        out.swap(queue_);
        return out.size();
    }

    uint64_t produced() const { std::lock_guard<std::mutex> l(mtx_); return produced_; }
    uint64_t dropped()  const { std::lock_guard<std::mutex> l(mtx_); return dropped_; }

private:
    mutable std::mutex   mtx_;
    std::vector<Frame>   queue_;
    size_t               max_frames_;
    uint64_t             produced_ = 0;
    uint64_t             dropped_  = 0;
};
