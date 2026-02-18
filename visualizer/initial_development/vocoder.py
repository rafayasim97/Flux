import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

def load_weights(filename, rows, cols):
    data = np.fromfile(filename, dtype=np.uint16)
    data_u32 = data.astype(np.uint32) << 16
    weights = data_u32.view(np.float32)
    return weights.reshape(rows, cols)

print("Loading Trained Flux Brain...")
W1 = load_weights("../flux_vs_cpp/flux_proto/trained_w1.bin", 64, 128)
W2 = load_weights("../flux_vs_cpp/flux_proto/trained_w2.bin", 128, 80)

# 1. Simulate Forward Pass
# Input was 1.0 at index 30-35
input_vec = np.zeros((1, 64), dtype=np.float32)
input_vec[0, 30:35] = 1.0

# Input -> W1 -> ReLU -> W2
hidden = np.maximum(0, np.dot(input_vec, W1))
spectrogram_slice = np.dot(hidden, W2) # Shape: (1, 80)

# 2. FIX: Stretch time
# Transpose to get (Frequency x Time) = (80, 1)
spectrogram_slice = spectrogram_slice.T 

# Repeat the slice 300 times to create a sustained sound
spectrogram = np.tile(spectrogram_slice, (1, 300))

print(f"Spectrogram Shape: {spectrogram.shape} (80 Frequencies x 300 Time Steps)")

# 3. Griffin-Lim Vocoder
# We explicitly set n_fft to handle the custom 80-bin height
n_fft = 2 * (spectrogram.shape[0] - 1)
audio = librosa.griffinlim(spectrogram, n_fft=n_fft)

# 4. Save and Plot
sf.write('flux_trained_speech.wav', audio, 22050)
print("Saved 'flux_trained_speech.wav'")

plt.figure(figsize=(10, 5))
plt.imshow(spectrogram, aspect='auto', origin='lower')
plt.title("The AI's Learned Pattern (Stretched)")
plt.ylabel("Frequency (Mel Bins)")
plt.xlabel("Time (Repeated Frames)")
plt.colorbar()
plt.show()