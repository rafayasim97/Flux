import numpy as np
import librosa
from gtts import gTTS
import os
import string

# CONFIG
LETTERS = list(string.ascii_uppercase) # ['A', 'B', 'C', ...]
COLS_TIME = 64
COLS_CHARS = 26 # One flag for each letter
COLS_TOTAL = COLS_TIME + COLS_CHARS # 90 Inputs
N_MELS = 80

all_inputs = []
all_targets = []

print("Generating Alphabet Dataset...")

for i, char in enumerate(LETTERS):
    filename = f"char_{char}.mp3"
    
    # 1. Generate Audio for letter (e.g., "Ay", "Bee", "See")
    # We add a pause to ensure gTTS pronounces it clearly
    tts = gTTS(char, lang='en')
    tts.save(filename)
    
    y, sr = librosa.load(filename, sr=22050)
    y, _ = librosa.effects.trim(y)

    # 2. Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=256)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    target = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    target = target.T 
    
    num_frames = target.shape[0]
    
    # 3. Create Inputs
    inputs = np.zeros((num_frames, COLS_TOTAL), dtype=np.float32)
    for t in range(num_frames):
        time_val = t / num_frames
        
        # Time Encoding (Cols 0-63)
        for k in range(10):
            freq = 2.0 ** k
            inputs[t, 2*k]     = np.sin(time_val * freq * 3.1415)
            inputs[t, 2*k + 1] = np.cos(time_val * freq * 3.1415)
        
        inputs[t, 63] = 1.0 # Bias
        
        # CHARACTER FLAG (Cols 64-89)
        # If letter is 'A' (index 0), set Col 64 = 1.0
        # If letter is 'B' (index 1), set Col 65 = 1.0
        inputs[t, COLS_TIME + i] = 1.0 

    all_inputs.append(inputs)
    all_targets.append(target)
    print(f"  Processed '{char}' ({num_frames} frames)")

# Stack
final_input = np.vstack(all_inputs)
final_target = np.vstack(all_targets)

print(f"\nTotal Batch Size: {final_input.shape[0]}") 

# Save
def save_bin(data, filename):
    data_u32 = data.astype(np.float32).view(np.uint32)
    data_bf16 = (data_u32 >> 16).astype(np.uint16)
    data_bf16.tofile(filename)

save_bin(final_input, "input_alphabet.bin")
save_bin(final_target, "target_alphabet.bin")
print("Saved alphabet dataset.")