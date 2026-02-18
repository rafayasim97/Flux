import numpy as np
import matplotlib.pyplot as plt

# 1. Settings
ROWS = 1  # Batch size
COLS_IN = 64
COLS_OUT = 80 # 80 Mel bands

# 2. Create Input (The letter "A" concept)
# A simple pattern: 1.0 in the middle, 0.0 elsewhere
input_vec = np.zeros((ROWS, COLS_IN), dtype=np.float32)
input_vec[0, 30:35] = 1.0 

# 3. Create Target (A Frequency Sweep / "Woop" sound)
# We want a diagonal line in the spectrogram
target_vec = np.zeros((ROWS, COLS_OUT), dtype=np.float32)
for i in range(COLS_OUT):
    # A simple sine wave shape across the frequency bands
    val = np.sin(i * 0.1) * 0.5 + 0.5 
    target_vec[0, i] = val

# 4. Save as bfloat16 (simulated as uint16)
def save_bin(data, filename):
    # Convert float32 to bfloat16 (truncate bottom 16 bits)
    data_u32 = data.view(np.uint32)
    data_bf16 = (data_u32 >> 16).astype(np.uint16)
    data_bf16.tofile(filename)
    print(f"Saved {filename}")

save_bin(input_vec, "input_data.bin")
save_bin(target_vec, "target_data.bin")

# 5. Visualize what we are teaching
plt.plot(target_vec[0])
plt.title("The Target Sound (Spectrogram Slice)")
plt.show()