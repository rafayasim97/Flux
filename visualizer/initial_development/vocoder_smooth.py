import numpy as np
import soundfile as sf
import librosa
import scipy.ndimage

def load_weights(filename, rows, cols):
    data = np.fromfile(filename, dtype=np.uint16)
    data_u32 = data.astype(np.uint32) << 16
    weights = data_u32.view(np.float32)
    return weights.reshape(rows, cols)

print("Loading Flux Brain...")
W1 = load_weights("../flux_vs_cpp/flux_proto/hello_w1.bin", 64, 256)
W2 = load_weights("../flux_vs_cpp/flux_proto/hello_w2.bin", 256, 80)

# Inputs
input_batch = np.zeros((5, 64), dtype=np.float32)
for i in range(5):
    input_batch[i, i] = 1.0
    input_batch[i, 63] = 1.0

# Forward Pass
hidden = np.dot(input_batch, W1)
hidden = np.where(hidden > 0, hidden, hidden * 0.1) # Leaky ReLU
outputs = np.dot(hidden, W2) # (5, 80)

# --- THE SMOOTHING TRICK ---
# We have 5 "Keyframes" of the word Hello.
# We need to stretch this to ~60 frames (1 second) by interpolating.

target_length = 60
spectrogram = np.zeros((80, target_length))

# Resize using Zoom (Bilinear Interpolation)
# This blends Frame 1 into Frame 2 smoothly, creating "Movement"
spectrogram = scipy.ndimage.zoom(outputs.T, (1, target_length / 5), order=1)

print(f"Generated Spectrogram: {spectrogram.shape}")

# Reconstruct Audio
# We need to 'denormalize' slightly to get volume back
spectrogram = spectrogram * 100.0 - 80.0 # Convert back to approx dB range
spectrogram = librosa.db_to_power(spectrogram)

# Griffin-Lim
audio = librosa.griffinlim(spectrogram, n_fft=2*79, hop_length=256)

sf.write('flux_real_hello.wav', audio, 22050)
print("✅ Saved 'flux_real_hello.wav'")