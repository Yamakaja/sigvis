#include <plot/sample_ring.hpp>
#include <vke/vke.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

namespace plot {

namespace {
// Host upload cap per frame (samples). Bounds staging size and per-frame copy work;
// at the real link rate (≤125 MB/s) a frame's worth of new samples is well under it.
constexpr uint32_t MAX_PENDING = 1u << 21; // 2 M samples (8 MB staging)

inline uint32_t pos_mod(int64_t a, uint32_t m) {
    int64_t r = a % static_cast<int64_t>(m);
    if (r < 0) r += m;
    return static_cast<uint32_t>(r);
}
} // namespace

struct SampleRing::Impl {
    vke::Context& ctx;
    float         max_window_seconds;
    uint32_t      frames_in_flight;

    // Session geometry.
    float    sample_rate = 0.0f;
    uint32_t S = 0;   // samples per frame
    uint32_t C = 0;   // ring capacity (samples), multiple of S
    uint32_t F = 0;   // frame slots = C / S

    // Authoritative CPU timeline (read by the trigger scan).
    std::vector<float>   cpu_ring;      // size C
    std::vector<int64_t> slot_seq;      // size F, last seq written to each slot (-1 = none)

    // Reassembly bookkeeping.
    int64_t  head_index   = 0;   // one past newest absolute sample
    int64_t  first_index  = 0;   // lowest absolute sample ever written
    bool     have_data    = false;
    int64_t  expected_seq = 0;
    uint64_t dropped      = 0;

    // Device residency + per-frame host staging.
    vke::Buffer              samples_dev;   // float[C]
    vke::Buffer              present_dev;   // uint32[F]
    std::vector<vke::Buffer> sample_host;   // float[host_cap] × frames_in_flight
    std::vector<vke::Buffer> present_host;  // uint32[F]    × frames_in_flight
    uint32_t                 slot_index = 0;

    // Received regions awaiting upload (absolute start, count).
    std::vector<std::pair<int64_t, uint32_t>> pending;
    uint64_t generation = 0; // bumped on every (re)allocation of device buffers

    Impl(vke::Context& c, float win_s, uint32_t fif)
        : ctx(c), max_window_seconds(std::max(1e-3f, win_s)),
          frames_in_flight(std::max(1u, fif)) {}

    void allocate(uint32_t host_cap) {
        samples_dev = ctx.create_buffer({
            .size = static_cast<VkDeviceSize>(C) * sizeof(float),
            .usage = vke::BufferUsage::Storage | vke::BufferUsage::TransferDst,
            .domain = vke::MemoryDomain::Device, .debug_name = "sample_ring" });
        present_dev = ctx.create_buffer({
            .size = static_cast<VkDeviceSize>(F) * sizeof(uint32_t),
            .usage = vke::BufferUsage::Storage | vke::BufferUsage::TransferDst,
            .domain = vke::MemoryDomain::Device, .debug_name = "sample_ring_present" });

        sample_host.clear();  sample_host.resize(frames_in_flight);
        present_host.clear(); present_host.resize(frames_in_flight);
        for (auto& b : sample_host)
            b = ctx.create_buffer({
                .size = static_cast<VkDeviceSize>(host_cap) * sizeof(float),
                .usage = vke::BufferUsage::TransferSrc,
                .domain = vke::MemoryDomain::Host, .debug_name = "sample_ring_upload" });
        for (auto& b : present_host)
            b = ctx.create_buffer({
                .size = static_cast<VkDeviceSize>(F) * sizeof(uint32_t),
                .usage = vke::BufferUsage::TransferSrc,
                .domain = vke::MemoryDomain::Host, .debug_name = "sample_ring_present_host" });
        slot_index = 0;
        ++generation;
    }

    void configure(float rate, uint32_t spf) {
        sample_rate = rate;
        S = std::max(1u, spf);
        uint32_t want_frames = std::max<uint32_t>(1,
            static_cast<uint32_t>(std::ceil(max_window_seconds * rate / S)));
        F = want_frames;
        C = F * S;
        // Cap the device ring (256 MB): a fixed max_window auto-shrinks in time at high
        // rates (e.g. full-rate 125 MS/s) while keeping seconds of history at low rates.
        constexpr uint32_t MAX_RING_SAMPLES = 64u << 20;
        if (C > MAX_RING_SAMPLES) { F = std::max(1u, MAX_RING_SAMPLES / S); C = F * S; }

        cpu_ring.assign(C, 0.0f);
        slot_seq.assign(F, -1);
        reset_counters();

        uint32_t host_cap = std::max(S, std::min(C, MAX_PENDING));
        allocate(host_cap);
    }

    void reset_counters() {
        head_index = first_index = 0;
        have_data = false;
        expected_seq = 0;
        dropped = 0;
        pending.clear();
    }

    void reset() {
        std::fill(slot_seq.begin(), slot_seq.end(), int64_t{-1});
        reset_counters();
    }

    // Write `src` into the wrapped CPU ring starting at absolute index `base`.
    void write_cpu(int64_t base, std::span<const float> src) {
        uint32_t pos = pos_mod(base, C);
        uint32_t n   = static_cast<uint32_t>(src.size());
        uint32_t first = std::min(n, C - pos);
        std::memcpy(cpu_ring.data() + pos, src.data(), first * sizeof(float));
        if (n > first)
            std::memcpy(cpu_ring.data(), src.data() + first, (n - first) * sizeof(float));
    }

    void ingest(const RingFrame& f) {
        if (f.samples_per_frame != 0 && f.samples_per_frame != S)
            configure(sample_rate, f.samples_per_frame);
        if (S == 0 || f.samples.empty()) return;

        const int64_t  base = f.seq * static_cast<int64_t>(S);
        const uint32_t n    = std::min<uint32_t>(static_cast<uint32_t>(f.samples.size()), S);

        // Skip frames that fall entirely behind the ring window (stale / far reorder).
        if (have_data && base + static_cast<int64_t>(S) <= head_index - static_cast<int64_t>(C))
            return;

        if (f.seq > expected_seq) dropped += static_cast<uint64_t>(f.seq - expected_seq);
        expected_seq = f.seq + 1;

        write_cpu(base, f.samples.first(n));
        slot_seq[pos_mod(f.seq, F)] = f.seq;
        pending.emplace_back(base, n);

        head_index = std::max(head_index, base + n);
        if (!have_data) { first_index = base; have_data = true; }
        else            first_index = std::min(first_index, base);
    }

    // The frame the live window expects in physical slot `s`: the unique frame in
    // [Hf-F, Hf) congruent to s (mod F). present iff that frame was actually received.
    void compute_present(std::span<uint32_t> out) {
        const int64_t Hf = head_index / static_cast<int64_t>(S); // head frame (exclusive)
        for (uint32_t s = 0; s < F; ++s) {
            int64_t window_base = Hf - static_cast<int64_t>(F);
            int64_t f_exp = window_base + pos_mod(static_cast<int64_t>(s) - window_base, F);
            out[s] = (f_exp >= 0 && slot_seq[s] == f_exp) ? 1u : 0u;
        }
    }

    void record_upload(vke::CommandBuffer& cmd) {
        if (pending.empty()) return; // nothing changed → device copies still valid

        auto& sh = sample_host[slot_index];
        auto& ph = present_host[slot_index];
        slot_index = (slot_index + 1) % frames_in_flight;

        // ---- Pack received regions into the host staging buffer, record copies ----
        float*         sh_ptr   = sh.template mapped_as<float>().data();
        const uint32_t host_cap = static_cast<uint32_t>(sh.size() / sizeof(float));
        uint32_t       hoff     = 0;

        for (auto [base, count] : pending) {
            if (hoff + count > host_cap) break; // overflow: drop remaining (rare)
            uint32_t pos   = pos_mod(base, C);
            uint32_t first = std::min(count, C - pos);

            // host: contiguous at hoff (copy from the possibly-wrapped CPU ring).
            std::memcpy(sh_ptr + hoff, cpu_ring.data() + pos, first * sizeof(float));
            if (count > first)
                std::memcpy(sh_ptr + hoff + first, cpu_ring.data(),
                            (count - first) * sizeof(float));

            // device: same split (dest wraps at C).
            cmd.copy_buffer(sh, samples_dev,
                static_cast<VkDeviceSize>(hoff) * sizeof(float),
                static_cast<VkDeviceSize>(pos) * sizeof(float),
                static_cast<VkDeviceSize>(first) * sizeof(float));
            if (count > first)
                cmd.copy_buffer(sh, samples_dev,
                    static_cast<VkDeviceSize>(hoff + first) * sizeof(float), 0,
                    static_cast<VkDeviceSize>(count - first) * sizeof(float));
            hoff += count;
        }
        pending.clear();

        // ---- Presence map (whole buffer; small) ----
        compute_present(ph.template mapped_as<uint32_t>());
        cmd.copy_buffer(ph, present_dev, 0, 0,
                        static_cast<VkDeviceSize>(F) * sizeof(uint32_t));

        // ---- Make ring + present visible to compute (roll) and mesh (trigger) ----
        const VkPipelineStageFlags2 dst_stages =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
        cmd.buffer_barrier({ .buffer = samples_dev,
            .src_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .src_access_mask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dst_stage_mask = dst_stages, .dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT });
        cmd.buffer_barrier({ .buffer = present_dev,
            .src_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .src_access_mask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dst_stage_mask = dst_stages, .dst_access_mask = VK_ACCESS_2_SHADER_READ_BIT });
    }
};

SampleRing::SampleRing(vke::Context& ctx, float max_window_seconds, uint32_t frames_in_flight)
    : impl_(std::make_unique<Impl>(ctx, max_window_seconds, frames_in_flight)) {}

SampleRing::~SampleRing()                                = default;
SampleRing::SampleRing(SampleRing&&) noexcept            = default;
SampleRing& SampleRing::operator=(SampleRing&&) noexcept = default;

void SampleRing::configure(float sample_rate_hz, uint32_t samples_per_frame) {
    if (sample_rate_hz <= 0.0f || samples_per_frame == 0) return;
    impl_->configure(sample_rate_hz, samples_per_frame);
}
void SampleRing::set_sample_rate(float rate) {
    if (rate <= 0.0f || rate == impl_->sample_rate) return;
    if (impl_->S != 0) impl_->configure(rate, impl_->S); // resets at the new rate
    else               impl_->sample_rate = rate;        // S filled in on first ingest
}
void SampleRing::reset()                       { impl_->reset(); }
void SampleRing::ingest(const RingFrame& f)    { impl_->ingest(f); }
void SampleRing::record_upload(vke::CommandBuffer& cmd) { impl_->record_upload(cmd); }

const vke::Buffer& SampleRing::samples_buffer() const noexcept { return impl_->samples_dev; }
const vke::Buffer& SampleRing::present_buffer() const noexcept { return impl_->present_dev; }

uint32_t SampleRing::capacity()          const noexcept { return impl_->C; }
uint32_t SampleRing::samples_per_frame() const noexcept { return impl_->S; }
uint32_t SampleRing::frame_slots()       const noexcept { return impl_->F; }
float    SampleRing::sample_rate()       const noexcept { return impl_->sample_rate; }
int64_t  SampleRing::head_index()        const noexcept { return impl_->head_index; }
uint64_t SampleRing::dropped_frames()    const noexcept { return impl_->dropped; }
uint64_t SampleRing::generation()        const noexcept { return impl_->generation; }

uint64_t SampleRing::buffered_samples() const noexcept {
    auto& im = *impl_;
    if (!im.have_data) return 0;
    uint64_t span = static_cast<uint64_t>(im.head_index - im.first_index);
    return std::min<uint64_t>(span, im.C);
}

float SampleRing::sample_at(int64_t a) const noexcept {
    auto& im = *impl_;
    if (im.C == 0) return 0.0f;
    return im.cpu_ring[pos_mod(a, im.C)];
}

bool SampleRing::present_at(int64_t a) const noexcept {
    auto& im = *impl_;
    if (im.S == 0) return false;
    int64_t frame = a / static_cast<int64_t>(im.S);
    if (a < 0) return false;
    return im.slot_seq[pos_mod(frame, im.F)] == frame;
}

} // namespace plot
