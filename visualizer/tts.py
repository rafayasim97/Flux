import numpy as np
import soundfile as sf
import librosa
import scipy.ndimage
import sys

# CHANGE THIS TO MATCH V2 (25)
FRAMES_PER_CHAR = 25 
COLS_TIME = 64
COLS_CHARS = 26
COLS_TOTAL = 90
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def load_weights(filename, rows, cols):
    data = np.fromfile(filename, dtype=np.uint16)
    data_u32 = data.astype(np.uint32) << 16
    return data_u32.view(np.float32).reshape(rows, cols)

print("Loading Alphabet Brain...")
W1 = load_weights("../flux_vs_cpp/flux_proto/hello_w1.bin", COLS_TOTAL, 512)
W2 = load_weights("../flux_vs_cpp/flux_proto/hello_w2.bin", 512, 80)

def speak(text):
    text = text.upper()
    print(f"Speaking: '{text}'")
    
    full_output = []
    
    for char in text:
        if char not in CHARS:
            # Short silence for space
            silence = np.zeros((5, 80), dtype=np.float32)
            full_output.append(silence)
            continue
            
        char_idx = CHARS.index(char)
        
        # Generate Input
        inputs = np.zeros((FRAMES_PER_CHAR, COLS_TOTAL), dtype=np.float32)
        for t in range(FRAMES_PER_CHAR):
            time_val = t / FRAMES_PER_CHAR
            
            for k in range(10):
                freq = 2.0 ** k
                inputs[t, 2*k]     = np.sin(time_val * freq * 3.1415)
                inputs[t, 2*k + 1] = np.cos(time_val * freq * 3.1415)
            
            inputs[t, 63] = 1.0 
            inputs[t, COLS_TIME + char_idx] = 1.0
        
        # Inference
        hidden = np.dot(inputs, W1)
        hidden = np.where(hidden > 0, hidden, hidden * 0.1)
        mel = np.dot(hidden, W2)
        
        full_output.append(mel)
    
    if not full_output: return
    
    # --- CROSSFADE STITCHING ---
    # Instead of vstack, we should overlap if possible, but for now 
    # let's just smooth the junction aggressively.
    spectrogram = np.vstack(full_output)
    
    # HEAVY SMOOTHING along the Time Axis (sigma 1.0)
    # This blurs the "seams" between letters
    spectrogram = scipy.ndimage.gaussian_filter(spectrogram, sigma=[1.0, 0.5])

    # --- VOCODER ---
    mel_db = spectrogram.T * 100.0 - 80.0
    mel_power = librosa.db_to_power(mel_db)
    
    mel_basis = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    linear_spec = np.dot(inv_mel_basis, mel_power)
    linear_spec = np.maximum(1e-10, linear_spec)
    
    # More Iterations = Clearer Voice
    audio = librosa.griffinlim(linear_spec, n_fft=1024, hop_length=256, n_iter=100)
    
    filename = f"tts_{text}.wav"
    sf.write(filename, audio, 22050)
    print(f"✅ Saved '{filename}'")

if len(sys.argv) > 1:
    speak(" ".join(sys.argv[1:]))
else:
    speak("HELLO")