import numpy as np
import matplotlib.pyplot as plt

# We have 5 letters (Rows)
ROWS = 5 
COLS_IN = 64
COLS_OUT = 80

# --- 1. INPUTS (One-Hot Encoding) ---
# Each row represents a letter. 
# Row 0 is H (Index 0=1), Row 1 is E (Index 1=1), etc.
input_vec = np.zeros((ROWS, COLS_IN), dtype=np.float32)
for i in range(ROWS):
    input_vec[i, i] = 1.0 # Simple diagonal identity

# --- 2. TARGETS (Spectrogram Slices) ---
target_vec = np.zeros((ROWS, COLS_OUT), dtype=np.float32)

# Letter H (Row 0): High frequency noise
target_vec[0, 60:] = 0.8 # Top bands active

# Letter E (Row 1): Mid frequency harmonics
target_vec[1, 20:50:5] = 0.9 # Every 5th band lit up

# Letter L (Row 2 & 3): Low frequency solid
target_vec[2, 5:15] = 0.9
target_vec[3, 5:15] = 0.9

# Letter O (Row 4): Deep Formants
target_vec[4, 0:10] = 0.9
target_vec[4, 20:25] = 0.6

# --- 3. SAVE ---
def save_bin(data, filename):
    data_u32 = data.view(np.uint32)
    data_bf16 = (data_u32 >> 16).astype(np.uint16)
    data_bf16.tofile(filename)
    print(f"Saved {filename}")

save_bin(input_vec, "input_hello.bin")
save_bin(target_vec, "target_hello.bin")

plt.imshow(target_vec.T, aspect='auto', origin='lower')
plt.title("Target Spectrogram (H - E - L - L - O)")
plt.show()