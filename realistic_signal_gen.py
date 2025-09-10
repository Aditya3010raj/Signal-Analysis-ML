import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows

# global stitched frequency range (Hz)
f_min = 2.35e9
f_max = 3.6e9
BW_total = f_max - f_min
N = 2**16  # number of frequency bins (adjust: larger => finer freq resolution)
df = BW_total / N
f_center = (f_min + f_max) / 2
freq_bins = (np.arange(N) - N//2) * df + f_center   # centered frequency axis (Hz)

S = np.zeros(N, dtype=complex)    # complex spectrum (two-sided, centered)

def place_band(center_hz, bw_hz, peak_linear=1.0, qam_order=None, window_type='hann'):
    K = max(4, int(np.round(bw_hz / df)))    # number of bins for this band
    if K % 2 == 0:
        K += 1
    idx_center = int(np.argmin(np.abs(freq_bins - center_hz)))
    half = K // 2
    left = idx_center - half
    right = idx_center + half + 1
    # small random QPSK/QAM-like spectrum
    phases = np.exp(1j * 2*np.pi*np.random.rand(K))
    if qam_order is None:
        mag = np.ones(K)
    else:
        # create some amplitude variation across subcarriers to mimic real OFDM PSD shape
        mag = np.linspace(1.0, 0.6, K)
    w = getattr(windows, window_type)(K)
    spec = peak_linear * (mag * w) * phases
    # bounds handling
    if left < 0:
        spec = spec[-left:]
        left = 0
    if right > N:
        spec = spec[:N-right]
        right = N
    S[left:right] += spec[:right-left]

# realistic band definitions (centers and approximate bandwidths)
bands = [
    ('ZigBee', 2.405e9, 2e6, 0.6, None, 'hann'),
    ('WiFi',   2.437e9, 20e6, 1.0, 16, 'hann'),
    ('Bluetooth', 2.44e9, 2e6, 0.5, None, 'hann'),
    ('5G_NR',  3.50e9, 100e6, 1.2, 64, 'hann')
]

for name, fc, bw, amp, qam, wint in bands:
    place_band(fc, bw, peak_linear=amp, qam_order=qam, window_type=wint)

# add a low-level noise floor in frequency domain
noise_floor = (np.random.randn(N) + 1j*np.random.randn(N)) * 1e-3
S += noise_floor

# time-domain analytic signal (centered) via IFFT
s_time = np.fft.ifft(np.fft.ifftshift(S))
# set real passband-like signal by taking real part of s_time*exp(j*2π f_center t).
# to visualize passband energy vs absolute GHz, we work with the analytic s_time and map frequencies to freq_bins
Fs = BW_total               # effective sampling rate of stitched waveform
T = N / Fs
t = np.arange(N) / Fs

# PSD from stitched spectrum (use S directly for highest fidelity)
S_shifted = np.fft.fftshift(S)
PSD_db = 20 * np.log10(np.abs(S_shifted) + 1e-20)

# Plot PSD with labeled regions (GHz)
plt.figure(figsize=(12,5))
plt.plot(freq_bins/1e9, PSD_db, lw=0.6)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Amplitude (dB)')
plt.title('Stitched Passband PSD (2.35–3.60 GHz)')
plt.grid(True)

# draw shaded regions and text labels
annot_params = dict(alpha=0.25)
plt.axvspan(2.403, 2.407, color='tab:purple', **annot_params); plt.text(2.403, PSD_db.max()-10, 'ZigBee\n(2.405 GHz)', color='k')
plt.axvspan(2.427, 2.447, color='tab:blue', **annot_params);   plt.text(2.427, PSD_db.max()-20, 'WiFi 20 MHz\n(2.437 GHz)', color='k')
plt.axvspan(2.439, 2.441, color='tab:green', **annot_params);  plt.text(2.439, PSD_db.max()-30, 'Bluetooth\n(2.44 GHz)', color='k')
plt.axvspan(3.45, 3.55, color='tab:orange', **annot_params);   plt.text(3.45, PSD_db.max()-15, '5G NR 100 MHz\n(3.5 GHz)', color='k')

plt.xlim(f_min/1e9, f_max/1e9)
plt.ylim(PSD_db.min()-5, PSD_db.max()+2)
plt.tight_layout()

# Spectrogram of stitched time signal (use complex spectrogram to preserve passband mapping)
f_s, t_s, Sxx = spectrogram(s_time, fs=Fs, nperseg=4096, noverlap=2048, return_onesided=False, mode='complex')
Sxx_db = 20 * np.log10(np.abs(np.fft.fftshift(Sxx, axes=0)) + 1e-20)
f_s_shifted = np.fft.fftshift(f_s) + f_center   # absolute freq axis (Hz)

plt.figure(figsize=(12,5))
plt.pcolormesh(t_s*1e3, f_s_shifted/1e9, Sxx_db, shading='gouraud')
plt.colorbar(label='Amplitude (dB)')
plt.ylabel('Frequency (GHz)')
plt.xlabel('Time (ms)')
plt.title('Spectrogram (stitched passband)')
plt.ylim(f_min/1e9, f_max/1e9)

# horizontal shaded bands on spectrogram with labels
plt.axhspan(2.403, 2.407, color='tab:purple', alpha=0.25)
plt.text(0.5 * t_s.max()*1e3, 2.405, 'ZigBee', ha='center', va='center', color='k')
plt.axhspan(2.427, 2.447, color='tab:blue', alpha=0.25)
plt.text(0.5 * t_s.max()*1e3, 2.437, 'WiFi 20 MHz', ha='center', va='center', color='k')
plt.axhspan(2.439, 2.441, color='tab:green', alpha=0.25)
plt.text(0.5 * t_s.max()*1e3, 2.440, 'Bluetooth', ha='center', va='center', color='k')
plt.axhspan(3.45, 3.55, color='tab:orange', alpha=0.25)
plt.text(0.5 * t_s.max()*1e3, 3.50, '5G NR 100 MHz', ha='center', va='center', color='k')

plt.tight_layout()
plt.show()
