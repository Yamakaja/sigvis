#include "plstream_source.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

// ---- plstreamd control protocol (wire-stable; mirrors plstreamd_proto.h) ----
constexpr uint32_t PLS_MAGIC   = 0x54534C50u; // "PLST"
constexpr uint16_t PLS_VERSION = 1;
enum : uint16_t {
    PLS_HELLO = 1, PLS_HELLO_ACK = 2, PLS_CONFIGURE = 3,
    PLS_START = 4, PLS_STOP = 5, PLS_RESULT = 8,
};
#pragma pack(push, 1)
struct pls_hdr       { uint16_t type; uint16_t flags; uint32_t length; };
struct pls_hello     { uint32_t magic; uint16_t version; uint16_t caps; };
struct pls_configure { uint16_t dst_port; uint16_t _pad; };
struct pls_result    { int32_t  code; };

// ---- acq frame header (wire-stable; mirrors hardware_handoff.md §4) ----
constexpr uint32_t ACQ_MAGIC = 0x52504143u; // "RPAC"
struct acq_frame_header {
    uint32_t magic;
    uint16_t flags;
    uint8_t  decim_log2;
    uint8_t  version;
    uint64_t seq;
    uint64_t timestamp;
    uint32_t drop_count;
    uint32_t payload_words; // 64-bit words; samples/channel = payload_words * 2
};
#pragma pack(pop)
static_assert(sizeof(acq_frame_header) == 32);

bool write_all(int fd, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    while (n) {
        ssize_t k = ::write(fd, p, n);
        if (k < 0) { if (errno == EINTR) continue; return false; }
        p += k; n -= static_cast<size_t>(k);
    }
    return true;
}
bool read_all(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    while (n) {
        ssize_t k = ::read(fd, p, n);
        if (k == 0) return false;
        if (k < 0) { if (errno == EINTR) continue; return false; }
        p += k; n -= static_cast<size_t>(k);
    }
    return true;
}
bool send_msg(int fd, uint16_t type, const void* pl, uint32_t len) {
    pls_hdr h{ .type = type, .flags = 0, .length = len };
    if (!write_all(fd, &h, sizeof(h))) return false;
    return len == 0 || write_all(fd, pl, len);
}
// Send a request and read the daemon's RESULT code (0 = ok). Returns false on IO error.
bool request(int fd, uint16_t type, const void* pl, uint32_t len, int32_t& code) {
    if (!send_msg(fd, type, pl, len)) return false;
    for (;;) {
        pls_hdr h;
        if (!read_all(fd, &h, sizeof(h))) return false;
        std::vector<char> buf(h.length);
        if (h.length && !read_all(fd, buf.data(), h.length)) return false;
        if (h.type == PLS_RESULT) {
            if (h.length < sizeof(pls_result)) return false;
            pls_result r; std::memcpy(&r, buf.data(), sizeof(r));
            code = r.code; return true;
        }
        // ignore anything else (e.g. STATS) until the RESULT arrives
    }
}

// Non-blocking connect bounded by timeout_ms and abortable via `running`, so a wrong
// address fails in seconds (not the ~2 min default SYN-retransmit) and stop() stays
// responsive. Returns 0 on success, -1 on error/timeout/abort.
int connect_timeout(int fd, const sockaddr* addr, socklen_t len, int timeout_ms,
                    const std::atomic<bool>& running) {
    int flags = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    int rc = ::connect(fd, addr, len);
    if (rc != 0 && errno != EINPROGRESS) return -1;
    int ok = -1;
    if (rc == 0) {
        ok = 0;
    } else {
        for (int waited = 0; waited < timeout_ms && running.load(std::memory_order_relaxed);
             waited += 200) {
            pollfd pfd{ .fd = fd, .events = POLLOUT, .revents = 0 };
            int pr = ::poll(&pfd, 1, 200);
            if (pr > 0) {
                int soerr = 0; socklen_t el = sizeof(soerr);
                ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &el);
                if (soerr == 0) ok = 0; else errno = soerr;
                break;
            }
            if (pr < 0 && errno != EINTR) break;
        }
    }
    ::fcntl(fd, F_SETFL, flags); // restore blocking for the request/response IO
    return ok;
}

// Connect TCP to host:port; returns fd or -1 (msg in `err`).
int ctrl_connect(const std::string& host, uint16_t port,
                 const std::atomic<bool>& running, std::string& err) {
    addrinfo hints{}; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    char portstr[16]; std::snprintf(portstr, sizeof(portstr), "%u", port);
    addrinfo* res = nullptr;
    int e = ::getaddrinfo(host.c_str(), portstr, &hints, &res);
    if (e) { err = std::string("resolve: ") + gai_strerror(e); return -1; }
    int fd = -1;
    for (addrinfo* rp = res; rp; rp = rp->ai_next) {
        fd = ::socket(rp->ai_family, rp->ai_socktype | SOCK_CLOEXEC, rp->ai_protocol);
        if (fd < 0) continue;
        if (connect_timeout(fd, rp->ai_addr, rp->ai_addrlen, 5000, running) == 0) break;
        ::close(fd); fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0) { err = std::string("connect: ") + std::strerror(errno); return -1; }
    int one = 1; ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    // HELLO handshake.
    pls_hello hello{ .magic = PLS_MAGIC, .version = PLS_VERSION, .caps = 0 };
    if (!send_msg(fd, PLS_HELLO, &hello, sizeof(hello))) { err = "send hello"; ::close(fd); return -1; }
    pls_hdr h;
    if (!read_all(fd, &h, sizeof(h)) || h.type != PLS_HELLO_ACK || h.length < sizeof(pls_hello)) {
        err = "bad hello ack"; ::close(fd); return -1;
    }
    pls_hello ack{}; std::vector<char> buf(h.length);
    if (!read_all(fd, buf.data(), h.length)) { err = "short hello ack"; ::close(fd); return -1; }
    std::memcpy(&ack, buf.data(), sizeof(ack));
    if (ack.magic != PLS_MAGIC || ack.version != PLS_VERSION) {
        err = "version mismatch"; ::close(fd); return -1;
    }
    return fd;
}

} // namespace

PlstreamSource::PlstreamSource(std::string host, uint16_t ctrl_port,
                               uint16_t udp_port, Channel channel)
    : host_(std::move(host)), ctrl_port_(ctrl_port), udp_port_(udp_port),
      channel_(static_cast<int>(channel)) {}

PlstreamSource::~PlstreamSource() { stop(); }

void PlstreamSource::start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread([this] { run(); });
}

void PlstreamSource::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
}

void PlstreamSource::run() {
    // ---- UDP receive socket (bound to our port) ----
    int udp = ::socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (udp < 0) { error_ = std::string("udp socket: ") + std::strerror(errno);
                   ok_.store(false); running_.store(false); return; }
    int rcvbuf = 64 << 20; // large SO_RCVBUF to absorb bursts
    ::setsockopt(udp, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    timeval tv{ .tv_sec = 0, .tv_usec = 100000 }; // 100 ms: lets the loop see stop()
    ::setsockopt(udp, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(udp_port_);
    if (::bind(udp, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        error_ = std::string("udp bind: ") + std::strerror(errno);
        ok_.store(false); running_.store(false); ::close(udp); return;
    }

    // ---- Control: connect, configure our UDP port, start streaming ----
    std::string err;
    int ctrl = ctrl_connect(host_, ctrl_port_, running_, err);
    if (ctrl < 0) { error_ = err; ok_.store(false); running_.store(false); ::close(udp); return; }

    int32_t code = 0;
    pls_configure cfg{ .dst_port = udp_port_, ._pad = 0 };
    if (!request(ctrl, PLS_CONFIGURE, &cfg, sizeof(cfg), code) || code != 0) {
        error_ = "configure failed (" + std::to_string(code) + ")";
        ok_.store(false); running_.store(false); ::close(ctrl); ::close(udp); return;
    }
    if (!request(ctrl, PLS_START, nullptr, 0, code) || code != 0) {
        error_ = "start failed (" + std::to_string(code) + ")";
        ok_.store(false); running_.store(false); ::close(ctrl); ::close(udp); return;
    }
    ok_.store(true, std::memory_order_relaxed);

    // ---- Receive loop ----
    std::vector<uint8_t> pkt(16 * 1024); // > max frame (32 + 8192)
    while (running_.load(std::memory_order_relaxed)) {
        ssize_t n = ::recvfrom(udp, pkt.data(), pkt.size(), 0, nullptr, nullptr);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
            break;
        }
        if (static_cast<size_t>(n) < sizeof(acq_frame_header)) continue;

        acq_frame_header h;
        std::memcpy(&h, pkt.data(), sizeof(h));
        if (h.magic != ACQ_MAGIC) continue;

        const uint32_t samples_per_ch = h.payload_words * 2;
        const size_t   need = sizeof(acq_frame_header) +
                              static_cast<size_t>(samples_per_ch) * 2 * sizeof(int16_t);
        if (samples_per_ch == 0 || static_cast<size_t>(n) < need) continue;

        // Split the drops we can see at this stage. drop_count is cumulative fabric
        // drops; the received-seq gap is fabric + network/kernel losses.
        const int64_t  fseq = static_cast<int64_t>(h.seq);
        const uint32_t fdc  = h.drop_count;
        if (!have_seq_) {
            prev_seq_ = fseq; prev_dc_ = fdc; have_seq_ = true;
        } else if (fseq > prev_seq_) {
            int64_t gap = fseq - prev_seq_ - 1;                          // total missing
            int64_t fab = std::max<int64_t>(0, static_cast<int64_t>(fdc) - prev_dc_);
            fab = std::min(fab, gap);                                    // fabric subset
            fabric_drops_.fetch_add(static_cast<uint64_t>(fab), std::memory_order_relaxed);
            net_drops_.fetch_add(static_cast<uint64_t>(gap - fab), std::memory_order_relaxed);
            prev_seq_ = fseq; prev_dc_ = fdc;
        } else if (fseq < prev_seq_) {                                    // re-arm / reset
            prev_seq_ = fseq; prev_dc_ = fdc;
        }

        rate_hz_.store(125.0e6f / static_cast<float>(1u << h.decim_log2),
                       std::memory_order_relaxed);
        const float scale = (h.decim_log2 == 0) ? (1.0f / 8192.0f)   // bypass: 14-bit
                                                : (1.0f / 32768.0f); // decimated: 16-bit

        const int       ch  = channel_.load(std::memory_order_relaxed) & 1;
        const int16_t*  s16 = reinterpret_cast<const int16_t*>(pkt.data() + sizeof(acq_frame_header));

        Frame frame;
        frame.seq               = static_cast<int64_t>(h.seq);
        frame.timestamp         = static_cast<int64_t>(h.timestamp);
        frame.samples_per_frame = samples_per_ch;
        frame.samples.resize(samples_per_ch);
        for (uint32_t i = 0; i < samples_per_ch; ++i)
            frame.samples[i] = static_cast<float>(s16[2 * i + ch]) * scale;

        queue_.push(std::move(frame));
    }

    request(ctrl, PLS_STOP, nullptr, 0, code);
    ::close(ctrl);
    ::close(udp);
}
