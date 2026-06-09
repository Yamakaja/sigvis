#include "data_source.hpp"

#include <chrono>
#include <cmath>
#include <numbers>
#include <random>

DataSource::DataSource(uint32_t chunk_size, uint32_t chunks_per_sec)
    : chunk_size_(chunk_size), chunks_per_sec_(chunks_per_sec) {}

DataSource::~DataSource() {
    stop();
}

void DataSource::start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread([this] { run(); });
}

void DataSource::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
}

void DataSource::set_signal(const Signal& s) {
    cycles_.store(s.cycles,       std::memory_order_relaxed);
    amplitude_.store(s.amplitude, std::memory_order_relaxed);
    noise_std_.store(s.noise_std, std::memory_order_relaxed);
    jitter_.store(s.jitter,       std::memory_order_relaxed);
    am_depth_.store(s.am_depth,   std::memory_order_relaxed);
    am_rate_.store(s.am_rate,     std::memory_order_relaxed);
}

void DataSource::run() {
    using clock = std::chrono::steady_clock;

    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float>       gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    const auto period = std::chrono::duration<double>(1.0 / static_cast<double>(chunks_per_sec_));
    auto next = clock::now();
    int64_t seq = 0;

    // Continuous oscillator state carried across records: the generated stream is a
    // single uninterrupted signal (no per-record phase reset), so it reassembles into a
    // gap-free timeline the way real hardware does. (A per-record reset would put a
    // phase discontinuity at every record boundary, which a trigger window straddling
    // the boundary would render as a pinch around the trigger point.)
    double carrier_phase = 0.0; // accumulated carrier phase (rad)
    double am_phase      = 0.0; // accumulated AM phase (rad)
    double jitter_off    = 0.0; // slow, continuous phase wander (rad)
    (void)uni;

    while (running_.load(std::memory_order_relaxed)) {
        // Snapshot live params for this record.
        const float cycles    = cycles_.load(std::memory_order_relaxed);
        const float amplitude = amplitude_.load(std::memory_order_relaxed);
        const float noise_std = noise_std_.load(std::memory_order_relaxed);
        const float jitter    = jitter_.load(std::memory_order_relaxed);
        const float am_depth  = am_depth_.load(std::memory_order_relaxed);
        const float am_rate   = am_rate_.load(std::memory_order_relaxed);

        // Per-sample phase increments (continuous across records).
        const double w       = 2.0 * std::numbers::pi * cycles / static_cast<double>(chunk_size_);
        const double am_step = am_rate / static_cast<double>(chunk_size_);

        Frame frame;
        frame.seq               = seq;
        frame.timestamp         = seq * static_cast<int64_t>(chunk_size_); // 1 tick/sample
        frame.samples_per_frame = chunk_size_;
        frame.samples.resize(chunk_size_);
        constexpr double TWO_PI = 2.0 * std::numbers::pi;
        for (uint32_t i = 0; i < chunk_size_; ++i) {
            // Wrap phases to [0, 2π) so the sin() argument never grows large (keeps
            // full double precision regardless of how long the source has run).
            carrier_phase += w;       if (carrier_phase >= TWO_PI) carrier_phase -= TWO_PI;
            am_phase      += am_step;  if (am_phase     >= TWO_PI) am_phase     -= TWO_PI;
            // Phase jitter as a bounded continuous random walk (Ornstein-Uhlenbeck):
            // the trigger pins it at the crossing while it spreads the band away from it.
            jitter_off += static_cast<double>(jitter) * 0.02 * gauss(rng) - 0.005 * jitter_off;
            const float amp = amplitude *
                (1.0f + am_depth * std::sin(static_cast<float>(am_phase)));
            frame.samples[i] = amp * std::sin(static_cast<float>(carrier_phase + jitter_off))
                             + noise_std * gauss(rng);
        }

        queue_.push(std::move(frame));
        ++seq;

        next += std::chrono::duration_cast<clock::duration>(period);
        std::this_thread::sleep_until(next);
    }
}
