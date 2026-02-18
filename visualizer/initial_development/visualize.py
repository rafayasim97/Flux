import numpy as np
import matplotlib.pyplot as plt

def load_weights(filename, rows, cols):
    data = np.fromfile(filename, dtype=np.uint16)
    # Convert bfloat16 representation to float32
    data_u32 = data.astype(np.uint32) << 16
    weights = data_u32.view(np.float32)
    return weights.reshape(rows, cols)

W1 = load_weights("../flux_vs_cpp/flux_proto/voice_layer1.bin", 64, 128)
W2 = load_weights("../flux_vs_cpp/flux_proto/voice_layer2.bin", 128, 80)

# --- NEW: PRINT STATS ---
print(f"Layer 1 Shape: {W1.shape}")
print(f"Layer 1 Sample: {W1[0, :5]}") # Print first 5 weights
print(f"Layer 1 Min: {W1.min()}, Max: {W1.max()}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.title("Layer 1"); plt.imshow(W1, aspect='auto')
plt.subplot(1, 2, 2); plt.title("Layer 2"); plt.imshow(W2, aspect='auto')
plt.show()