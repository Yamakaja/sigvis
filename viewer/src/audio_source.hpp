#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

#include "source.hpp"

// Live audio capture via the PulseAudio simple API (works transparently against a
// PipeWire pulse server). Captures mono float32 at a fixed rate and emits
// fixed-size records. Pass a source/monitor name to capture a specific device, or
// leave empty for the server default. To visualize what's playing, use a sink's
// ".monitor" source name.
class AudioSource : public ISource {
public:
    AudioSource(uint32_t rate_hz = 48000, uint32_t chunk_size = 1024,
                std::string device = {});
    ~AudioSource() override;

    AudioSource(const AudioSource&)            = delete;
    AudioSource& operator=(const AudioSource&) = delete;

    void   start() override;
    void   stop()  override;
    size_t drain(std::vector<std::vector<float>>& out) override { return queue_.drain(out); }
    float    sample_rate() const override { return static_cast<float>(rate_hz_); }
    uint64_t produced()    const override { return queue_.produced(); }
    uint64_t dropped()     const override { return queue_.dropped(); }

    uint32_t           chunk_size() const { return chunk_size_; }
    bool               ok()         const { return ok_.load(std::memory_order_relaxed); }
    const std::string& error()      const { return error_; } // valid after a failed start

private:
    void run();

    uint32_t    rate_hz_;
    uint32_t    chunk_size_;
    std::string device_;

    std::thread       thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> ok_{false};
    std::string       error_;
    SampleQueue       queue_;
};
