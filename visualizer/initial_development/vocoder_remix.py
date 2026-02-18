import numpy as np
import soundfile as sf
import librosa
import scipy.ndimage

# CONFIG
FRAMES_PER_WORD = 70
COLS_IN = 70
W1 = np.fromfile("../flux_vs_cpp/flux_proto/hello_w1.bin", dtype=np.uint16).astype(np.uint32) << 16
W1 = W1.view(np.float32).reshape(COLS_IN, 512)
W2 = np.fromfile("../flux_vs_cpp/flux_proto/hello_w2.bin", dtype=np.uint16).astype(np.uint32) << 16
W2 = W2.view(np.float32).reshape(512, 80)

def synthesize_hybrid(name, switch_point=0.5):
    print(f"Generating Mutant Word: {name}...")
    
    input_batch = np.zeros((FRAMES_PER_WORD, COLS_IN), dtype=np.float32)

    for t in range(FRAMES_PER_WORD):
        time_val = t / FRAMES_PER_WORD
        
        # 1. Standard Time & Positional Encoding
        for k in range(10):
            freq = 2.0 ** k
            input_batch[t, 2*k]     = np.sin(time_val * freq * 3.1415)
            input_batch[t, 2*k + 1] = np.cos(time_val * freq * 3.1415)
        input_batch[t, 63] = 1.0 # Bias
        
        # 2. THE FRANKENSTEIN SWITCH
        # If we are in the first half of the audio...
        if time_val < switch_point:
            # Turn ON "Flux" (Word 1) -> "Fff" sound
            input_batch[t, 65] = 1.0 
        else:
            # Turn ON "Hello" (Word 0) -> "ello" sound
            input_batch[t, 64] = 1.0

    # 3. Inference
    hidden = np.dot(input_batch, W1)
    hidden = np.where(hidden > 0, hidden, hidden * 0.1) 
    mel_output = np.dot(hidden, W2)

    # 4. DSP Cleanup
    mel_output = scipy.ndimage.gaussian_filter(mel_output, sigma=0.5)
    mel_db = mel_output.T * 100.0 - 80.0
    mel_power = librosa.db_to_power(mel_db)
    
    mel_basis = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    linear_spec = np.dot(inv_mel_basis, mel_power)
    linear_spec = np.maximum(1e-10, linear_spec)

    audio = librosa.griffinlim(linear_spec, n_fft=1024, hop_length=256, n_iter=64)
    sf.write(f'{name}.wav', audio, 22050)
    print(f"✅ Saved '{name}.wav'")

# Experiment 1: "Fullo" (Start of Flux, End of Hello)
synthesize_hybrid("flux_fullo", switch_point=0.6) 

# Experiment 2: "Hex" (Start of Hello, End of Flux)
# We flip the logic manually inside the loop or simpler:
# Just create a new function for the reverse if needed, 
# but "Fullo" proves the point best.