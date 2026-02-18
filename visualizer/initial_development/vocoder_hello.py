import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

def load_weights(filename, rows, cols):
    data = np.fromfile(filename, dtype=np.uint16)
    data_u32 = data.astype(np.uint32) << 16
    weights = data_u32.view(np.float32)
    return weights.reshape(rows, cols)

print("Loading Flux Brain...")
# NOTE: Dimensions must match your Rust engine (64 -> 256 -> 80)
W1 = load_weights("../flux_vs_cpp/flux_proto/hello_w1.bin", 64, 256)
W2 = load_weights("../flux_vs_cpp/flux_proto/hello_w2.bin", 256, 80)

# 1. Prepare Input (H, E, L, L, O)
# We must include the BIAS column (63) just like in training
input_batch = np.zeros((5, 64), dtype=np.float32)
for i in range(5):
    input_batch[i, i] = 1.0
    input_batch[i, 63] = 1.0 # Bias

# 2. Forward Pass (Replicating the Rust Engine logic)
# Input * W1
hidden_pre = np.dot(input_batch, W1)
# Leaky ReLU (0.1)
hidden = np.where(hidden_pre > 0, hidden_pre, hidden_pre * 0.1)
# Hidden * W2
outputs = np.dot(hidden, W2) 

# 3. Sequence & Stitch
full_spectrogram = []
print("Synthesizing Speech...")

for i in range(5):
    frame = outputs[i]
    # Stretch each phoneme to 15 frames (~100ms)
    # Clip negative values (Sound energy can't be negative)
    frame = np.maximum(0, frame) 
    stretched = np.tile(frame.reshape(80, 1), (1, 15))
    
    full_spectrogram.append(stretched)
    
    # Tiny silence between letters for clarity
    full_spectrogram.append(np.zeros((80, 3)))

# Flatten
final_spec = np.hstack(full_spectrogram)

# 4. Griffin-Lim (Spectrogram -> Audio)
# n_fft must match the height (80 bins -> 158 fft size)
n_fft = 2 * (80 - 1)
audio = librosa.griffinlim(final_spec, n_fft=n_fft)

sf.write('flux_hello.wav', audio, 22050)
print("✅ Done! Open 'flux_hello.wav' to hear your AI speak.")

plt.figure(figsize=(10, 4))
plt.imshow(final_spec, aspect='auto', origin='lower', cmap='inferno')
plt.title("Final AI Output Spectrogram")
plt.colorbar()
plt.show()