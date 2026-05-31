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

size_t DataSource::drain(std::vector<std::vector<float>>& out) {
    out.clear();
    std::lock_guard<std::mutex> lock(mtx_);
    out.swap(queue_);
    return out.size();
}

void DataSource::run() {
    using clock = std::chrono::steady_clock;

    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float>       gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    const auto period = std::chrono::duration<double>(1.0 / static_cast<double>(chunks_per_sec_));
    auto next = clock::now();
    double am_phase = 0.0;

    while (running_.load(std::memory_order_relaxed)) {
        // Snapshot live params for this record.
        const float cycles    = cycles_.load(std::memory_order_relaxed);
        const float amplitude = amplitude_.load(std::memory_order_relaxed);
        const float noise_std = noise_std_.load(std::memory_order_relaxed);
        const float jitter    = jitter_.load(std::memory_order_relaxed);
        const float am_depth  = am_depth_.load(std::memory_order_relaxed);
        const float am_rate   = am_rate_.load(std::memory_order_relaxed);

        const double w = 2.0 * std::numbers::pi * cycles / static_cast<double>(chunk_size_);
        const float  phase = jitter * uni(rng);
        am_phase += am_rate;
        const float amp = amplitude * (1.0f + am_depth * std::sin(static_cast<float>(am_phase)));

        std::vector<float> chunk(chunk_size_);
        for (uint32_t i = 0; i < chunk_size_; ++i)
            chunk[i] = amp * std::sin(static_cast<float>(w * i) + phase) + noise_std * gauss(rng);

        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (queue_.size() >= MAX_QUEUED) {
                size_t drop = queue_.size() - MAX_QUEUED + 1;
                queue_.erase(queue_.begin(), queue_.begin() + drop);
                dropped_.fetch_add(drop, std::memory_order_relaxed);
            }
            queue_.push_back(std::move(chunk));
        }
        produced_.fetch_add(1, std::memory_order_relaxed);

        next += std::chrono::duration_cast<clock::duration>(period);
        std::this_thread::sleep_until(next);
    }
}
