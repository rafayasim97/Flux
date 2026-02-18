import numpy as np
import matplotlib.pyplot as plt

ROWS = 5 
COLS_IN = 64
COLS_OUT = 80

print("Generating ROBOTIC VOWEL Data...")

# 1. Inputs (Identity + Bias)
input_vec = np.zeros((ROWS, COLS_IN), dtype=np.float32)
for i in range(ROWS):
    input_vec[i, i] = 1.0       # The Letter Signal
    input_vec[i, 63] = 1.0      # The Bias Signal (Always On)

# 2. Rich Targets (Formants)
target_vec = np.zeros((ROWS, COLS_OUT), dtype=np.float32)

def add_formant(row, base_freq, strength=1.0):
    target_vec[row, base_freq] = strength
    # Add harmonics for texture
    if base_freq + 10 < 80: target_vec[row, base_freq + 10] = strength * 0.6
    if base_freq + 20 < 80: target_vec[row, base_freq + 20] = strength * 0.4

# H (White Noise)
target_vec[0, 50:] = np.random.rand(30) * 0.5 

# E (Vowel Formants)
add_formant(1, 15)

# L (Low Tone)
add_formant(2, 5)

# L (Repeat)
add_formant(3, 5)

# O (Deep Vowel)
add_formant(4, 2)

# Normalize to prevent explosion
target_vec = np.clip(target_vec, 0, 1)

def save_bin(data, filename):
    data_u32 = data.view(np.uint32)
    data_bf16 = (data_u32 >> 16).astype(np.uint16)
    data_bf16.tofile(filename)
    print(f"Saved {filename}")

save_bin(input_vec, "input_hello.bin")
save_bin(target_vec, "target_hello.bin")

plt.imshow(target_vec.T, aspect='auto', origin='lower', cmap='inferno')
plt.title("Target Spectrogram: H-E-L-L-O")
plt.show()