#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

#include "source.hpp"

// Live Red Pitaya streaming-ADC source.
//
// Control plane: a small TCP client to plstreamd (HELLO handshake, CONFIGURE the
// receiver UDP port, START / STOP). Data plane: raw UDP frames (a 32-byte acq frame
// header + interleaved int16 channel A/B samples) arrive on our UDP port and are
// reassembled into Frames carrying the hardware seq/timestamp, so the render side's
// gap-aware ring places dropped frames in the right place. One channel is selected
// and normalized to ~[-1, 1]; the sample rate is derived from the frame's decimation
// (125 MHz / 2^decim) once the first frame arrives.
//
// Self-contained: the (documented, stable) wire structs live in the .cpp, so this has
// no dependency on the external plstream headers or include path.
class PlstreamSource : public ISource {
public:
    enum class Channel { A = 0, B = 1 };

    PlstreamSource(std::string host, uint16_t ctrl_port = 7654,
                   uint16_t udp_port = 5000, Channel channel = Channel::A);
    ~PlstreamSource() override;

    PlstreamSource(const PlstreamSource&)            = delete;
    PlstreamSource& operator=(const PlstreamSource&) = delete;

    void   start() override;
    void   stop()  override;
    size_t drain(std::vector<Frame>& out) override { return queue_.drain(out); }
    float    sample_rate() const override { return rate_hz_.load(std::memory_order_relaxed); }
    uint64_t produced()    const override { return queue_.produced(); }
    uint64_t dropped()     const override { return queue_.dropped(); }

    bool               ok()    const { return ok_.load(std::memory_order_relaxed); }
    const std::string& error() const { return error_; } // valid after a failed start

    // Drop accounting split by stage (Red Pitaya only):
    //  - fabric:  frames the PL dropped before transmit (header drop_count deltas).
    //  - network: frames transmitted but lost before recvfrom (seq gap minus fabric) —
    //             i.e. lost in the network or the kernel UDP socket buffer.
    // Renderer drops (our queue overflow) are reported by dropped() above.
    uint64_t fabric_drops()  const { return fabric_drops_.load(std::memory_order_relaxed); }
    uint64_t network_drops() const { return net_drops_.load(std::memory_order_relaxed); }

    // Switch the displayed channel (A/B); takes effect on the next frame. The caller
    // should reset the scope so A and B samples don't mix on the timeline.
    void    set_channel(Channel c) { channel_.store(static_cast<int>(c), std::memory_order_relaxed); }
    Channel channel() const { return static_cast<Channel>(channel_.load(std::memory_order_relaxed)); }

private:
    void run();

    std::string host_;
    uint16_t    ctrl_port_;
    uint16_t    udp_port_;

    std::atomic<int>   channel_;
    std::atomic<float> rate_hz_{0.0f};

    // Stage-split drop counters (worker-updated, UI-read).
    std::atomic<uint64_t> fabric_drops_{0};
    std::atomic<uint64_t> net_drops_{0};
    bool     have_seq_ = false; // worker-thread-only seq/drop_count tracking state
    int64_t  prev_seq_ = 0;
    uint32_t prev_dc_  = 0;

    std::thread       thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> ok_{false};
    std::string       error_;
    SampleQueue       queue_;
};
