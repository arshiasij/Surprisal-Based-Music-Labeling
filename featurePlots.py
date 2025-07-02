import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio
filename = 'songs/shadow_on_the_sun.mp3'  # Try a 30–60 sec file to start
y, sr = librosa.load(filename, sr=22050)

# Parameters
hop_length = 4096
frame_duration = hop_length / sr  # ~23ms per frame

# 1. Spectral Flux (change in spectrum over time)
S = np.abs(librosa.stft(y, hop_length=hop_length))
flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))

# 2. Onset Strength (like a drum hit detector)
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

# 3. RMS Loudness
rms = librosa.feature.rms(y=y, frame_length=hop_length*2, hop_length=hop_length)[0]

# Time axis for plotting
times = librosa.frames_to_time(np.arange(len(flux)), sr=sr, hop_length=hop_length)

# Normalize for visualization
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

flux_norm = normalize(flux)
onset_norm = normalize(onset_env[:len(flux)])
rms_norm = normalize(rms[:len(flux)])

def smooth(x, window_size=10, iterations=10):
  for i in range(iterations):
    x = np.convolve(x, np.ones(window_size)/window_size, mode='same')
  return np.convolve(x, np.ones(window_size*10)/window_size*10, mode='same')

flux_smooth = smooth(flux_norm, window_size=3, iterations=100)
onset_smooth = smooth(onset_norm, window_size=3, iterations=100)
rms_smooth = smooth(rms_norm, window_size=3, iterations=100)

# === Surprisal Estimate ===
surprisal_est = 0.4 * flux_smooth + 0.4 * onset_smooth + 0.2 * rms_smooth

# === Compute individual derivatives ===
d_flux = np.abs(np.diff(flux_smooth))
d_onset = np.abs(np.diff(onset_smooth))
d_rms = np.abs(np.diff(rms_smooth))
times_diff = times[:-1]  # diff reduces length by 1

min_gap=10.0
# === Thresholds for each feature ===
def detect_peaks(diff_array, time_array, std_factor=1.0):
    threshold = np.mean(diff_array) + std_factor * np.std(diff_array)
    raw_peaks = time_array[np.where(diff_array > threshold)[0]]

    # Filter to avoid clustering
    filtered = []
    last = -min_gap
    for t in raw_peaks:
        if t - last >= min_gap:
            filtered.append(t)
            last = t
    return filtered

flux_peaks = detect_peaks(d_flux, times_diff, std_factor=0.9)
onset_peaks = detect_peaks(d_onset, times_diff, std_factor=0.9)
rms_peaks = detect_peaks(d_rms, times_diff, std_factor=0.9)

# === Merge and deduplicate timepoints across features
all_peaks = sorted(flux_peaks + onset_peaks + rms_peaks)

# === Suppress nearby duplicates (min_gap seconds apart)
def suppress_close_times(times, min_gap=10.0):
    final = []
    last_time = -min_gap
    for t in times:
        if t - last_time >= min_gap:
            final.append(t)
            last_time = t
    return final

merged_peaks = suppress_close_times(all_peaks)



# === Plotting ===
fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex=True)

# Flux plot
axs[0, 0].plot(times, flux_smooth, color='blue')
axs[0, 0].set_title('Spectral Flux')
axs[0, 0].set_ylabel('Normalized')
axs[0, 0].grid(True)
for t in flux_peaks:
    axs[0, 0].axvline(x=t, color='blue', linestyle='--', alpha=0.6)

# Onset plot
axs[0, 1].plot(times, onset_smooth, color='green')
axs[0, 1].set_title('Onset Strength')
axs[0, 1].grid(True)
for t in onset_peaks:
    axs[0, 1].axvline(x=t, color='green', linestyle='--', alpha=0.6)

# RMS plot
axs[1, 0].plot(times, rms_smooth, color='red')
axs[1, 0].set_title('RMS Loudness')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Normalized')
axs[1, 0].grid(True)
for t in rms_peaks:
    axs[1, 0].axvline(x=t, color='red', linestyle='--', alpha=0.6)

# Surprisal plot
axs[1, 1].plot(times, surprisal_est, color='purple')
axs[1, 1].set_title('Estimated Surprisal')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].grid(True)
for t in merged_peaks:
    axs[1, 1].axvline(x=t, color='purple', linestyle='--', alpha=0.6)

plt.suptitle('Audio Features and Detected Sudden Changes (per Feature)', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

