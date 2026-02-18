import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import scipy.ndimage

# Load Weights
def load_weights(filename, rows, cols):
    data = np.fromfile(filename, dtype=np.uint16)
    data_u32 = data.astype(np.uint32) << 16
    return data_u32.view(np.float32).reshape(rows, cols)

print("Loading Brain...")
W1 = load_weights("../flux_vs_cpp/flux_proto/hello_w1.bin", 64, 512)
W2 = load_weights("../flux_vs_cpp/flux_proto/hello_w2.bin", 512, 80)

# 1. Inputs (Time Curve)
num_frames = 73 
input_batch = np.zeros((num_frames, 64), dtype=np.float32)

for t in range(num_frames):
    time_val = t / num_frames
    for k in range(10):
        freq = 2.0 ** k
        input_batch[t, 2*k]     = np.sin(time_val * freq * 3.1415)
        input_batch[t, 2*k + 1] = np.cos(time_val * freq * 3.1415)
    input_batch[t, 63] = 1.0

# 2. Inference
hidden = np.dot(input_batch, W1)
hidden = np.where(hidden > 0, hidden, hidden * 0.1) 
outputs = np.dot(hidden, W2)

# --- CLEANUP PHASE (The Static Killer) ---

# A. The Noise Gate
# Any pixel brighter than 0.0 but darker than 0.2 is just noise. Kill it.
print("Applying Noise Gate...")
outputs = np.where(outputs < 0.2, 0.0, outputs)

# B. The Smoother
# Neural networks output "jagged" pixels. 
# A tiny blur connects the dots, making it sound organic, not metallic.
print("Smoothing Spectrogram...")
outputs = scipy.ndimage.gaussian_filter(outputs, sigma=0.8)

# -----------------------------------------

# 3. Audio Reconstruction
spectrogram = outputs.T

# Denormalize (Map 0.0-1.0 back to Decibels -80 to 0)
# We assume the training data maxed at 0dB and min was around -80dB
spectrogram = spectrogram * 80.0 - 80.0 
power_spec = librosa.db_to_power(spectrogram)

# 4. High-Quality Griffin-Lim
# Increased n_iter from 32 to 64 for clearer phase reconstruction
print("Synthesizing Audio (High Quality)...")
audio = librosa.griffinlim(power_spec, n_fft=2*79, hop_length=256, n_iter=64)

sf.write('flux_clean_hello.wav', audio, 22050)
print("✅ Saved 'flux_clean_hello.wav'")

# Visualize the Cleaned Spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
plt.title("Cleaned AI Voice (No Static)")
plt.colorbar()
plt.show()