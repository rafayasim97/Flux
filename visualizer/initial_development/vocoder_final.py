import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import scipy.ndimage

# --- CONFIG ---
SR = 22050
N_FFT = 1024  # Standard size for human voice (not 158!)
N_MELS = 80
HOP_LEN = 256

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
mel_output = np.dot(hidden, W2)

# --- THE FIX: MEL -> LINEAR CONVERSION ---

# A. Smooth the raw Mel output
# This removes "pixelation" noise from the neural net
mel_output = scipy.ndimage.gaussian_filter(mel_output, sigma=0.6)

# B. Transpose to (Freq, Time)
mel_spec = mel_output.T

# C. Denormalize to Decibels
# Map 0.0 -> -80dB, 1.0 -> 0dB (Adjust these if your volume is too low/high)
mel_db = mel_spec * 100.0 - 80.0 
mel_power = librosa.db_to_power(mel_db)

# D. The "Translator": Mel -> Linear
# This creates the filter bank that un-warps the frequencies
mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
# Mathematical inversion (Approximate)
inv_mel_basis = np.linalg.pinv(mel_basis)
linear_spec = np.dot(inv_mel_basis, mel_power)

# Ensure no negative energy (impossible in physics)
linear_spec = np.maximum(1e-10, linear_spec)

# 3. High-Quality Synthesis
print("Synthesizing Audio (Mel -> Linear -> Audio)...")
# Now we use the correct N_FFT (1024), which gives rich, deep sound
audio = librosa.griffinlim(linear_spec, n_fft=N_FFT, hop_length=HOP_LEN, n_iter=64)

sf.write('flux_final_hello.wav', audio, SR)
print("✅ Saved 'flux_final_hello.wav'")

# Visualize
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("What AI Predicted (Mel)")
plt.imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
plt.subplot(1, 2, 2)
plt.title("What We Hear (Linear)")
plt.imshow(librosa.power_to_db(linear_spec), aspect='auto', origin='lower', cmap='viridis')
plt.show()