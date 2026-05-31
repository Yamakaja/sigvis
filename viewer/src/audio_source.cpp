#include "audio_source.hpp"

#include <pulse/simple.h>
#include <pulse/error.h>

#include <iostream>

AudioSource::AudioSource(uint32_t rate_hz, uint32_t chunk_size, std::string device)
    : rate_hz_(rate_hz), chunk_size_(chunk_size), device_(std::move(device)) {}

AudioSource::~AudioSource() {
    stop();
}

void AudioSource::start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread([this] { run(); });
}

void AudioSource::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
}

void AudioSource::run() {
    // Mono float32 capture at the requested rate.
    pa_sample_spec ss{};
    ss.format   = PA_SAMPLE_FLOAT32NE;
    ss.rate     = rate_hz_;
    ss.channels = 1;

    pa_buffer_attr attr{};
    attr.maxlength = (uint32_t) -1; // Use the default maximum buffer size
    attr.tlength   = (uint32_t) -1; // Playback only (Target length)
    attr.prebuf    = (uint32_t) -1; // Playback only (Pre-buffering)
    attr.minreq    = (uint32_t) -1; // Playback only (Minimum request)
    
    // CRITICAL for recording: Force the server to wake up and deliver 
    // data in chunks of exactly your requested size.
    attr.fragsize  = sizeof(float) * chunk_size_;

    int err = 0;
    const char* dev = device_.empty() ? nullptr : device_.c_str();
    pa_simple* s = pa_simple_new(nullptr, "sigvis", PA_STREAM_RECORD, dev,
                                 "scope capture", &ss, nullptr, &attr, &err);
    if (!s) {
        error_ = pa_strerror(err);
        ok_.store(false, std::memory_order_relaxed);
        running_.store(false, std::memory_order_relaxed);
        return;
    }
    ok_.store(true, std::memory_order_relaxed);

    const size_t bytes = static_cast<size_t>(chunk_size_) * sizeof(float);
    while (running_.load(std::memory_order_relaxed)) {
        std::vector<float> chunk(chunk_size_);
        if (pa_simple_read(s, chunk.data(), bytes, &err) < 0) {
            error_ = pa_strerror(err);
            ok_.store(false, std::memory_order_relaxed);
            break;
        }
        queue_.push(std::move(chunk));
    }

    pa_simple_free(s);
}
