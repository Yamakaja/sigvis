#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

// Synthetic free-running acquisition source. A background thread generates
// fixed-length records ("chunks") of a continuous signal at a target rate and
// pushes them into a bounded queue. The render thread calls drain() once per frame
// to take all pending chunks. This stands in for real hardware; each chunk is one
// scope trace.
//
// The generated signal is an amplitude-modulated sine with per-record phase jitter
// and additive noise. Its parameters are live-tunable from the UI thread via
// set_signal() (read atomically by the producer each record).
class DataSource {
public:
    DataSource(uint32_t chunk_size, uint32_t chunks_per_sec);
    ~DataSource();

    DataSource(const DataSource&)            = delete;
    DataSource& operator=(const DataSource&) = delete;

    void start();
    void stop();

    // Move all pending chunks into `out` (cleared first). Returns the count.
    size_t drain(std::vector<std::vector<float>>& out);

    // Tunable signal parameters (thread-safe; applied to subsequent records).
    struct Signal {
        float cycles    = 3.0f;   // sine cycles per record (frequency)
        float amplitude = 1.0f;   // carrier amplitude
        float noise_std = 0.04f;  // additive Gaussian noise stddev
        float jitter    = 0.3f;   // per-record phase jitter, +/- radians
        float am_depth  = 0.15f;  // amplitude-modulation depth [0,1]
        float am_rate   = 0.05f;  // AM phase increment per record
    };
    void set_signal(const Signal& s);

    uint32_t chunk_size() const { return chunk_size_; }
    uint64_t produced()   const { return produced_.load(std::memory_order_relaxed); }
    uint64_t dropped()    const { return dropped_.load(std::memory_order_relaxed); }

private:
    void run();

    uint32_t chunk_size_;
    uint32_t chunks_per_sec_;

    std::thread           thread_;
    std::atomic<bool>     running_{false};
    std::atomic<uint64_t> produced_{0};
    std::atomic<uint64_t> dropped_{0};

    // Live signal parameters (individually atomic; coherence across fields not required).
    std::atomic<float> cycles_{3.0f};
    std::atomic<float> amplitude_{1.0f};
    std::atomic<float> noise_std_{0.04f};
    std::atomic<float> jitter_{0.3f};
    std::atomic<float> am_depth_{0.15f};
    std::atomic<float> am_rate_{0.05f};

    std::mutex                      mtx_;
    std::vector<std::vector<float>> queue_;
    static constexpr size_t MAX_QUEUED = 512; // drop oldest beyond this (render fell behind)
};
