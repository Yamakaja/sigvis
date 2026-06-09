#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vke/vke.hpp>

namespace plot {

// One delivered record of one channel as it enters the ring: sequence-numbered and
// timestamped so the timeline can be reassembled with gaps in the right places.
// Mirrors the viewer's source Frame (kept here so plot:: has no viewer dependency).
struct RingFrame {
    int64_t                  seq               = 0; // absolute frame index; gaps = drops
    int64_t                  timestamp         = 0; // source ticks at first sample
    uint32_t                 samples_per_frame = 0; // S, fixed within a session
    std::span<const float>   samples;               // one channel, normalized
};

// Shared gap-aware sample timeline feeding both the roll and trigger render paths.
//
// Packets are a transport unit, not a display unit: ingest() places each frame on a
// single absolute-sample-indexed timeline using its seq, so dropped frames leave
// holes in exactly the right place rather than silently compressing time. The ring
// keeps an authoritative CPU mirror (the trigger scan reads it directly) and a
// device-local copy the GPU draws/reduces from; record_upload() stages the received
// regions to the device each frame.
//
// Validity is tracked at frame granularity (drops are always whole frames): only
// received frames are uploaded, never a sentinel per missing sample. To keep the GPU
// int64-free, the CPU does all absolute-index arithmetic and uploads a uint32
// present[] map indexed by physical frame-slot; consumer shaders read present[] and
// never touch a 64-bit index.
class SampleRing {
public:
    SampleRing(vke::Context& ctx, float max_window_seconds = 0.1f,
               uint32_t frames_in_flight = 2);
    ~SampleRing();

    SampleRing(const SampleRing&)            = delete;
    SampleRing& operator=(const SampleRing&) = delete;
    SampleRing(SampleRing&&) noexcept;
    SampleRing& operator=(SampleRing&&) noexcept;

    // (Re)configure for a session and drop all history. Ring capacity C is rounded to
    // a whole number of S-sample frame slots covering max_window_seconds. ingest()
    // also auto-reconfigures if it sees a different samples_per_frame.
    void configure(float sample_rate_hz, uint32_t samples_per_frame);
    // Set the time axis without knowing S yet; ingest() finishes configuring on the
    // first frame. If S is already known, reconfigures (and resets) at the new rate.
    void set_sample_rate(float sample_rate_hz);
    void reset();

    // CPU reassembly: place a frame on the absolute timeline; account drops.
    void ingest(const RingFrame& f);

    // Stage received regions + the present[] map into cmd, with a barrier making them
    // visible to compute and mesh reads. Call once per frame before consumers read.
    void record_upload(vke::CommandBuffer& cmd);

    // ---- GPU resources for consumers (descriptor binding) ----
    const vke::Buffer& samples_buffer() const noexcept; // float[C], device-local
    const vke::Buffer& present_buffer() const noexcept; // uint32[F], device-local

    // ---- Geometry / state ----
    uint32_t capacity()          const noexcept; // C samples
    uint32_t samples_per_frame() const noexcept; // S
    uint32_t frame_slots()       const noexcept; // F = C / S
    float    sample_rate()       const noexcept;
    int64_t  head_index()        const noexcept; // one past the newest absolute sample
    uint64_t buffered_samples()  const noexcept; // min(received span, C)
    uint64_t dropped_frames()    const noexcept;
    // Bumps whenever the device buffers are (re)created (configure). Consumers compare
    // it to know when to rewrite descriptor sets that point at samples/present.
    uint64_t generation()        const noexcept;

    // ---- CPU timeline access (trigger scan); valid only within [head-C, head) ----
    float sample_at(int64_t a)  const noexcept;
    bool  present_at(int64_t a) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace plot
