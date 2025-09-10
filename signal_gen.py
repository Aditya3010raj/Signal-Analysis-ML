import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fftshift
import math

fs = 20e6  # Sampling rate (Hz)
duration = 0.02  # seconds of signal
t = np.arange(int(fs*duration)) / fs
n = t.size

# Relative powers
power_5g = 1.0
power_wifi = 0.6
power_bt = 0.4
power_zb = 0.2

# Frequency offsets (Hz)
f_5g = -3e6
f_wifi = 0.0
f_bt = 4e6
f_zb = 6e6

snr_db = 30  # SNR in dB

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
    t = np.linspace(-span/2, span/2, span*sps)
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
    i_bits = bits[0::2]
    q_bits = bits[1::2]
    i = 2*i_bits - 1
    q = 2*q_bits - 1
    pulse = np.ones(sps)
    i_up = np.repeat(i, sps)
    q_up = np.repeat(q, sps)
    q_up = np.roll(q_up, sps//2)
    i_shaped = np.convolve(i_up, pulse, mode='same')
    q_shaped = np.convolve(q_up, pulse, mode='same')
    return (i_shaped + 1j*q_shaped) / np.sqrt(2)

def freq_shift(x, f_shift):
    return x * np.exp(1j*2*np.pi*f_shift*t)

# Signal Generation 
sig_5g = generate_ofdm_stream(n, subc=512, qam=64, cplen=128) * np.sqrt(power_5g)
sig_wifi = generate_ofdm_stream(n, subc=256, qam=16, cplen=64) * np.sqrt(power_wifi)

bt_bits = np.random.randint(0,2, size=500)
sig_bt_bb = gfsk(bt_bits, bt=0.3, sps=int(fs/1e6))
sig_bt = np.zeros(n, dtype=complex)
sig_bt[:len(sig_bt_bb)] = sig_bt_bb
sig_bt *= np.sqrt(power_bt)

zb_bits = np.random.randint(0,2, size=1000)
sig_zb_bb = oqpsk(zb_bits, sps=int(fs/1e6))
sig_zb = np.zeros(n, dtype=complex)
sig_zb[:len(sig_zb_bb)] = sig_zb_bb
sig_zb *= np.sqrt(power_zb)

s5 = freq_shift(sig_5g, f_5g)
sw = freq_shift(sig_wifi, f_wifi)
sb = freq_shift(sig_bt, f_bt)
sz = freq_shift(sig_zb, f_zb)

mixed = s5 + sw + sb + sz

# Add noise
signal_power = np.mean(np.abs(mixed)**2)
snr_linear = 10**(snr_db/10)
noise_power = signal_power / snr_linear
noise = np.sqrt(noise_power/2) * (np.random.randn(n) + 1j*np.random.randn(n))
rx = mixed + noise

# Time domain
plt.figure(figsize=(12,3))
win_samples = min(2000, n)
plt.plot(np.arange(win_samples)/fs*1e3, np.real(rx[:win_samples]))
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (real)")
plt.title("Mixed Signal — Time Domain")
plt.grid(True)
plt.tight_layout()

# PSD
f, Pxx = signal.welch(rx, fs=fs, nperseg=4096, return_onesided=False)
Pxx_db = 10*np.log10(fftshift(Pxx) + 1e-20)
f_shifted = fftshift(f) / 1e6
plt.figure(figsize=(12,4))
plt.plot(f_shifted, Pxx_db)
plt.xlabel("Frequency (MHz)")
plt.ylabel("PSD (dB)")
plt.title("Power Spectral Density — Mixed Signal")
plt.grid(True)
plt.tight_layout()

# Spectrogram
f_s, t_s, Sxx = signal.spectrogram(rx, fs=fs, nperseg=2048, noverlap=1024)
plt.figure(figsize=(12,5))
plt.pcolormesh(t_s, f_s/1e6, 10*np.log10(Sxx + 1e-20), shading='gouraud')
plt.ylabel("Frequency (MHz)")
plt.xlabel("Time (s)")
plt.title("Spectrogram (dB) — Mixed Signal")
plt.colorbar(label='Power (dB)')
plt.tight_layout()

plt.show()

print(f"Sampling rate: {fs/1e6} MHz, Duration: {duration} s, Samples: {n}")
print(f"Signal power: {signal_power:.6f}, Noise power: {noise_power:.6e}, SNR set: {snr_db} dB")
