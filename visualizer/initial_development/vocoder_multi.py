import numpy as np
import soundfile as sf
import librosa
import scipy.ndimage
import sys

# CONFIG
WORDS = ["Hello", "Flux"]
FRAMES_PER_WORD = 70 # Approx duration (adjust if cut off)
COLS_IN = 70

def load_weights(filename, rows, cols):
    data = np.fromfile(filename, dtype=np.uint16)
    data_u32 = data.astype(np.uint32) << 16
    return data_u32.view(np.float32).reshape(rows, cols)

print("Loading Brain...")
W1 = load_weights("../flux_vs_cpp/flux_proto/hello_w1.bin", COLS_IN, 512)
W2 = load_weights("../flux_vs_cpp/flux_proto/hello_w2.bin", 512, 80)

def generate_word(word_id, filename):
    print(f"Synthesizing Word ID {word_id} ({WORDS[word_id]})...")
    
    # 1. Construct Input for THIS specific word
    # We create a timeline from 0.0 to 1.0
    input_batch = np.zeros((FRAMES_PER_WORD, COLS_IN), dtype=np.float32)

    for t in range(FRAMES_PER_WORD):
        time_val = t / FRAMES_PER_WORD
        
        # Positional Encoding
        for k in range(10):
            freq = 2.0 ** k
            input_batch[t, 2*k]     = np.sin(time_val * freq * 3.1415)
            input_batch[t, 2*k + 1] = np.cos(time_val * freq * 3.1415)
        
        # Bias
        input_batch[t, 63] = 1.0
        
        # THE SELECTOR SWITCH
        # Only turn on the column for the requested word
        input_batch[t, 64 + word_id] = 1.0

    # 2. Inference
    hidden = np.dot(input_batch, W1)
    hidden = np.where(hidden > 0, hidden, hidden * 0.1) 
    mel_output = np.dot(hidden, W2)

    # 3. DSP Cleanup (Mel -> Linear)
    mel_output = scipy.ndimage.gaussian_filter(mel_output, sigma=0.6)
    mel_db = mel_output.T * 100.0 - 80.0
    mel_power = librosa.db_to_power(mel_db)
    
    mel_basis = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    linear_spec = np.dot(inv_mel_basis, mel_power)
    linear_spec = np.maximum(1e-10, linear_spec)

    # 4. Save
    audio = librosa.griffinlim(linear_spec, n_fft=1024, hop_length=256, n_iter=64)
    sf.write(filename, audio, 22050)
    print(f"✅ Saved {filename}")

# Generate BOTH words
generate_word(0, "flux_word_0.wav")
generate_word(1, "flux_word_1.wav")