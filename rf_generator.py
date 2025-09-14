# rf_generator.py
import numpy as np
from scipy import signal
import math

# --- Sampling and default parameters (adjustable) ---
fs = 20e6        # sampling rate (Hz)
duration = 0.02  # seconds
snr_db_default = 30

# ---------- modulation helpers (from your snippet) ----------
def qam_mod(bits, M=16):
    k = int(np.log2(M))
    bits = bits.reshape((-1, k))
    ints = bits.dot(1 << np.arange(k)[::-1])
    m = int(np.sqrt(M))
    re = 2*(ints % m) - (m-1)
    im = 2*(ints // m) - (m-1)
    return (re + 1j*im) / np.sqrt((2/3)*(M-1))

def ofdm_symbol(num_subcarriers=256, qam_order=16, cp_len=64):
    bits = np.random.randint(0, 2, size=(num_subcarriers * int(np.log2(qam_order)),))
    symbols = qam_mod(bits, M=qam_order)
    td = np.fft.ifft(symbols)
    return np.concatenate([td[-cp_len:], td])

def generate_ofdm_stream(duration_samples, subc=256, qam=16, cplen=64):
    out = np.zeros(duration_samples, dtype=complex)
    idx = 0
    while idx < duration_samples:
        sym = ofdm_symbol(subc, qam, cplen)
        L = min(len(sym), duration_samples - idx)
        out[idx:idx+L] = sym[:L]
        idx += L
    return out

def gaussian_filter(bt, span, sps):
    t = np.linspace(-span/2, span/2, int(span*sps))
    alpha = math.sqrt(math.log(2)) / (2*bt)
    g = np.exp(- (t**2) / (2*alpha*alpha))
    return g / np.sum(g)

def gfsk(bits, bt=0.5, sps=20, h=0.5):
    nrz = 2*bits - 1
    up = np.repeat(nrz, sps)
    g = gaussian_filter(bt=bt, span=4, sps=sps)
    shaped = np.convolve(up, g, mode='same')
    phase = 2*np.pi*h * np.cumsum(shaped) / sps
    return np.exp(1j*phase)

def oqpsk(bits, sps=8):
    bits = bits[: (len(bits)//2)*2 ]
    i_bits = bits[0::2]; q_bits = bits[1::2]
    i = 2*i_bits - 1; q = 2*q_bits - 1
    pulse = np.ones(sps)
    i_up = np.repeat(i, sps)
    q_up = np.repeat(q, sps)
    q_up = np.roll(q_up, sps//2)
    i_shaped = np.convolve(i_up, pulse, mode='same')
    q_shaped = np.convolve(q_up, pulse, mode='same')
    return (i_shaped + 1j*q_shaped) / np.sqrt(2)

# ---------- frequency shift helper ----------
def freq_shift(x, f_shift, t):
    return x * np.exp(1j*2*np.pi*f_shift*t)

# ---------- main generator: returns spectrogram and labels ----------
def generate_example(fs_local=None, duration_local=None,
                     include_5g=True, include_wifi=True, include_bt=True, include_zb=True,
                     snr_db=snr_db_default, rng=None):
    """
    Returns:
      Sxx_db_norm : (F, T) spectrogram (float32) normalized per-example
      labels      : np.array([5g, wifi, bt, zb]) binary (0/1)
      f_s, t_s    : frequency and time axes (Hz, s)
    """
    if rng is None:
        rng = np.random.RandomState()

    fs_use = fs_local or fs
    duration_use = duration_local or duration
    n = int(fs_use * duration_use)
    t = np.arange(n) / fs_use

    # relative powers and offsets (you can randomize these later)
    power_5g = 1.0
    power_wifi = 0.6
    power_bt = 0.4
    power_zb = 0.2

    f_5g = -3e6
    f_wifi = 0.0
    f_bt = 4e6
    f_zb = 6e6

    signals = []
    labels = [0, 0, 0, 0]  # [5g, wifi, bt, zb]

    # 5G (OFDM wide)
    if include_5g:
        sg = generate_ofdm_stream(n, subc=512, qam=64, cplen=128) * np.sqrt(power_5g)
        sg = freq_shift(sg, f_5g, t)
        signals.append(sg)
        labels[0] = 1

    # WiFi (OFDM medium)
    if include_wifi:
        sw = generate_ofdm_stream(n, subc=256, qam=16, cplen=64) * np.sqrt(power_wifi)
        sw = freq_shift(sw, f_wifi, t)
        signals.append(sw)
        labels[1] = 1

    # Bluetooth (GFSK)
    if include_bt:
        bt_bits = rng.randint(0, 2, size=500)
        sb_bb = gfsk(bt_bits, bt=0.3, sps=max(1, int(fs_use/1e6)))
        sb = np.zeros(n, dtype=complex); L = min(len(sb_bb), n); sb[:L] = sb_bb[:L]
        sb *= np.sqrt(power_bt)
        sb = freq_shift(sb, f_bt, t)
        signals.append(sb)
        labels[2] = 1

    # ZigBee (OQPSK)
    if include_zb:
        zb_bits = rng.randint(0,2, size=1000)
        sz_bb = oqpsk(zb_bits, sps=max(1, int(fs_use/1e6)))
        sz = np.zeros(n, dtype=complex); L = min(len(sz_bb), n); sz[:L] = sz_bb[:L]
        sz *= np.sqrt(power_zb)
        sz = freq_shift(sz, f_zb, t)
        signals.append(sz)
        labels[3] = 1

    if len(signals) == 0:
        mixed = np.zeros(n, dtype=complex)
    else:
        mixed = np.sum(signals, axis=0)

    # Add noise according to SNR
    signal_power = np.mean(np.abs(mixed)**2) if np.any(mixed) else 1e-12
    snr_linear = 10**(snr_db/10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power/2) * (rng.randn(n) + 1j*rng.randn(n))
    rx = mixed + noise

    # Compute spectrogram (complex -> magnitude dB)
    nperseg = 1024
    noverlap = 512
    f_s, t_s, Sxx = signal.spectrogram(rx, fs=fs_use, nperseg=nperseg, noverlap=noverlap, mode='complex')
    Sxx_db = 20 * np.log10(np.abs(Sxx) + 1e-20)

    # normalize per-example (z-score)
    Sxx_db_norm = (Sxx_db - np.mean(Sxx_db)) / (np.std(Sxx_db) + 1e-12)

    return Sxx_db_norm.astype(np.float32), np.array(labels, dtype=np.int64), f_s, t_s
