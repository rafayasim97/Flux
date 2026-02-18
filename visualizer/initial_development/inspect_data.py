import numpy as np
import matplotlib.pyplot as plt
import os

def load_bf16_bin(filename, rows, cols):
    if not os.path.exists(filename):
        print(f"❌ MISSING: {filename}")
        return None
        
    # Read raw bytes
    raw_bytes = np.fromfile(filename, dtype=np.uint16)
    
    # Check size
    expected = rows * cols
    if raw_bytes.size != expected:
        print(f"❌ SIZE MISMATCH: {filename} expected {expected} elements, got {raw_bytes.size}")
        return None

    # Convert BF16 back to Float32
    # Shift left 16 bits to move the bf16 into the high-bits of an f32
    data_u32 = raw_bytes.astype(np.uint32) << 16
    data_f32 = data_u32.view(np.float32)
    
    return data_f32.reshape(rows, cols)

print("--- INSPECTING FLUX DATA ---")

# 1. Inspect Input
print("\n[INPUT_HELLO.BIN]")
inp = load_bf16_bin("input_hello.bin", 5, 64)
if inp is not None:
    print(f"  Max Value: {np.max(inp):.4f}")
    print(f"  Min Value: {np.min(inp):.4f}")
    print(f"  Mean Value: {np.mean(inp):.4f}")
    print(f"  Non-Zero Count: {np.count_nonzero(inp)}")
    if np.max(inp) == 0:
        print("  ⚠️ CRITICAL: INPUT IS ALL ZEROS")
    else:
        print("  ✅ Input looks valid.")
        # Plot Input
        plt.figure(figsize=(10, 2))
        plt.imshow(inp, aspect='auto', cmap='magma')
        plt.title("Visual Input (Must NOT be black)")
        plt.colorbar()
        plt.show()

# 2. Inspect Target
print("\n[TARGET_HELLO.BIN]")
tgt = load_bf16_bin("target_hello.bin", 5, 80)
if tgt is not None:
    print(f"  Max Value: {np.max(tgt):.4f}")
    print(f"  Mean Value: {np.mean(tgt):.4f}")
    if np.max(tgt) == 0:
        print("  ⚠️ CRITICAL: TARGET IS ALL ZEROS")
    else:
        print("  ✅ Target looks valid.")
        plt.figure(figsize=(10, 2))
        plt.imshow(tgt, aspect='auto', cmap='viridis')
        plt.title("Visual Target (Patterns)")
        plt.colorbar()
        plt.show()