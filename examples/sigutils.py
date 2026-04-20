import numpy as np
from scipy import signal,interpolate
# import matplotlib.pyplot as plt

def rcf(t, f, beta):
    "rcf: Raised cosine in frequency Filter (RCF)"
    
    # Time Domain
    h = np.sinc(t) * np.cos(np.pi*beta*t)/(1-(2*beta*t)**2);
    h[t == -0.5/beta] = np.pi/4 * np.sinc(1/(2*beta))
    h[t == +0.5/beta] = np.pi/4 * np.sinc(1/(2*beta))

    # Frequency Domain
    H = (1 + np.cos((np.pi/beta) * (np.abs(f) - (1-beta) / 2))) / 2;
    H[np.abs(f) > (1+beta)/2] = 0
    H[np.abs(f) < (1-beta)/2] = 1
    H[np.abs(f) == 1/2] = 1/2;
    
    return h,H

def ga(t, f, T):
    h = np.exp(-4*np.log(2)*t**2 / T**2)
    H = np.sqrt(np.pi/(4*np.log(2)/T**2))*np.exp(-np.pi**2 * f**2 /(4 * np.log(2) / T**2))
    return h,H

def cconv(a, b):
    l = max(a.size, b.size)
    return np.fft.ifft(np.fft.fft(a, n=l) * np.fft.fft(b, n=l))

def rrcf(t, f, beta):
    """
    rrcf Root Raised cosine in frequency filter (RRCF-Filter) 
    """

    h           = (np.sin(np.pi * t * (1-beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))) / (np.pi * t * (1 - (4 * beta * t)**2));
    h[t == 0]    = 1 - beta * (1 - 4/np.pi);
    h[np.abs(t) == 1/(4*beta)]  = beta/np.sqrt(2) * ((1 + 2/np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2/np.pi) * np.cos(np.pi / (4*beta)));

    # Frequency Domain
    H                     = np.sqrt(np.abs(((1 + np.cos((np.pi/beta) * (np.abs(f)-(1-beta)/2)))/2)));
    H[np.abs(f)>(1+beta)/2]  = 0;
    H[np.abs(f)<(1-beta)/2]  = 1;
    H[np.abs(f)==1/2]        = 1/np.sqrt(2);
    
    return h,H

def rcf_signal(n_symbols, sps, beta, C, filter_length=8, rng=None):
    """Generate a pulse-shaped signal from constellation C.

    Symbols are drawn uniformly from C, upsampled to sps samples/symbol,
    then convolved with a raised cosine filter via circular convolution so
    the filter wraps at the signal boundary — no startup transients.

    Parameters
    ----------
    n_symbols     : number of symbols to generate
    sps           : samples per symbol
    beta          : RCF roll-off factor (0 < beta <= 1)
    C             : 1-D constellation array (real or complex)
    filter_length : one-sided filter length in symbols;
                    the kernel spans [-filter_length, +filter_length] symbol periods
    rng           : numpy Generator (created fresh if None)

    Returns
    -------
    float32 (real C) or complex64 (complex C) ndarray of length n_symbols * sps
    """
    if rng is None:
        rng = np.random.default_rng()
    C = np.asarray(C)
    symbols = C[rng.integers(0, len(C), n_symbols)]

    upsampled = np.zeros(n_symbols * sps, dtype=np.complex128 if np.iscomplexobj(C) else np.float64)
    upsampled[::sps] = symbols

    t = np.arange(-filter_length * sps, filter_length * sps + 1) / sps
    h, _ = rcf(t, np.ones(1), beta)

    out = cconv(upsampled, h)
    if np.iscomplexobj(C):
        return out.astype(np.complex64)
    return out.real.astype(np.float32)


def phase_noise(delta_f, T_s, size):
    return np.cumsum(np.sqrt(2*np.pi*delta_f*T_s) * np.random.randn(size))

def modulation_alphabet(mod):
    if mod == "BPSK":
        return np.array([1, -1], dtype=complex)
    if mod == "QPSK":
        return np.exp(1j*(np.pi/2 * np.arange(4) + np.pi/4)).astype(dtype=np.complex64)
    raise RuntimeError("Invalid modulation type requested!")

def int_log2(x):
    assert isinstance(x, int)
    assert x > 0
    i = 0
    while x != 1:
        i += 1
        x >>= 1
        
    return i

def int_to_bit_list(v, width):
    return [(v & (0x1 << (width-1-i))) >> (width-1-i) for i in range(width)]

def int_to_bytes(v, width):
    # return [0xFF & (v >> ((width-i-1)*8)) for i in width]
    return int(v).to_bytes(width, "big")

def CRC32(poly, data):
    crc32_lookup_table = np.zeros(256, dtype=int)

    for dividend in range(256):
        curByte = dividend << 24
        for _ in range(8):
            if curByte & 0x80000000 != 0:
                curByte = 0xFFFFFFFF & (curByte << 1)
                curByte ^= poly
            else:
                curByte = 0xFFFFFFFF & (curByte << 1)

        crc32_lookup_table[dividend] = curByte

    crc = 0xFFFFFFFF
    for d in data:
        idx = (crc ^ (int(d) << 24)) >> 24
        crc = (0xFFFFFFFF & (crc << 8)) ^ crc32_lookup_table[idx]
    return crc ^ 0xFFFFFFFF

def viterbi_viterbi_phase_recovery(x):
    v = x**2
    angles = np.unwrap(np.angle(np.convolve(v, np.ones(16)/16, mode="same")))
    return x * np.exp(-1j * angles/2)

class PIDController:
    def __init__(self, p, i, d):
        self.p = p
        self.i = i
        self.d = d
        self.errorsum = 0
        self.last_e = 0
    
    def update(self, dt, e):
        self.errorsum += e*dt
        diff = e - self.last_e
        last_e = e
        return self.p * e + self.i * self.errorsum + diff*self.d

def extract_messages(sig, block_size=16, threshold_mean_size=1000, offset=30, plots=False):
    # Find signals
    p = np.abs(sig)**2
    p = p.reshape(-1, block_size).sum(1)
    p_dB = 10 * np.log10(p)

    window = signal.windows.blackman(threshold_mean_size)
    threshold = np.convolve(p_dB, window / np.sum(window), mode="same")

    delim = np.diff((p_dB > threshold + offset).astype(int))
    
    if plots:
        plt.figure(figsize=(20, 5))
        plt.plot(p_dB);
        # plt.plot([0, 1200], [threshold, threshold])
        plt.plot(threshold);
        plt.plot(delim);
    
    starts = np.where(delim == 1)[0]
    stops = np.where(delim == -1)[0]

    extracts = []
    if stops[0] < starts[0]:
        stops = stops[1:]

    for start,stop in zip(starts, stops):
        l = stop - start
        # print(start, stop)
        extract = sig[block_size*(start-2):block_size*(stop+3)]

        # Compensate frequency offset
        w_off = np.angle(np.sum(extract[1:] * np.conj(extract[:-1])))
        extract = extract * np.exp(-1j*w_off*np.arange(extract.size))

        extracts.append(extract)
        
    return extracts

def gardner_timing_recovery(x, sps, p_gain=0.01, i_gain=0.01, d_gain=0.001, interpolation_order=5):
    controller = PIDController(p_gain, i_gain, d_gain)
    inter = interpolate.interp1d(np.arange(x.size), x, kind=interpolation_order)
    def x_at(t):
        base = int(round(t))
        a = base - interpolation_order//2
        b = a + interpolation_order + 1
        inter = interpolate.interp1d(np.arange(a, b), x[a:b], kind=interpolation_order)

        return inter(t)
    
    symbols = []
    t = sps+interpolation_order
    
    ts = []
    errors = []
    symbols = []
    offsets = []
    off = 0
    
    while t < x.size-interpolation_order-sps:
        x_a = inter(t)
        x_b = inter(t-sps+off)
        x_m = inter(t-(sps+off)/2)
        
        symbols.append(x_a)
        ts.append(t)

        # if x_a * x_b < 0:
        error = np.real(np.conj(x_a - x_b)*x_m)

        off = controller.update(1, error)
        offsets.append(off)
        errors.append(error)
            
        t += sps - off
    
    return ts,errors,np.array(symbols),offsets
