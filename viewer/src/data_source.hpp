#pragma once
#include <atomic>
#include <cstdint>
#include <thread>

#include "source.hpp"

// Synthetic free-running acquisition source. A background thread generates
// fixed-length records ("chunks") of a continuous signal at a target rate. Each
// chunk is one scope trace. Stands in for real hardware.
//
// The generated signal is an amplitude-modulated sine with per-record phase jitter
// and additive noise. Its parameters are live-tunable from the UI thread via
// set_signal() (read atomically by the producer each record).
class DataSource : public ISource {
public:
    DataSource(uint32_t chunk_size, uint32_t chunks_per_sec);
    ~DataSource() override;

    DataSource(const DataSource&)            = delete;
    DataSource& operator=(const DataSource&) = delete;

    void   start() override;
    void   stop()  override;
    size_t drain(std::vector<std::vector<float>>& out) override { return queue_.drain(out); }
    float    sample_rate() const override { return static_cast<float>(chunk_size_) * chunks_per_sec_; }
    uint64_t produced()    const override { return queue_.produced(); }
    uint64_t dropped()     const override { return queue_.dropped(); }

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

private:
    void run();

    uint32_t chunk_size_;
    uint32_t chunks_per_sec_;

    std::thread       thread_;
    std::atomic<bool> running_{false};
    SampleQueue       queue_;

    // Live signal parameters (individually atomic; field coherence not required).
    std::atomic<float> cycles_{3.0f};
    std::atomic<float> amplitude_{1.0f};
    std::atomic<float> noise_std_{0.04f};
    std::atomic<float> jitter_{0.3f};
    std::atomic<float> am_depth_{0.15f};
    std::atomic<float> am_rate_{0.05f};
};
