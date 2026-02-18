import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from gtts import gTTS
import os
import string

# CONFIG
PHONEME_MAP = {
    'A': 'Apple', 'B': 'Ball', 'C': 'Cat', 'D': 'Dog', 'E': 'Elephant',
    'F': 'Fish', 'G': 'Goat', 'H': 'Hot', 'I': 'Igloo', 'J': 'Jug',
    'K': 'Kite', 'L': 'Lion', 'M': 'Monkey', 'N': 'Nest', 'O': 'Octopus',
    'P': 'Pig', 'Q': 'Queen', 'R': 'Rat', 'S': 'Sun', 'T': 'Top',
    'U': 'Umbrella', 'V': 'Van', 'W': 'Watch', 'X': 'Box', 'Y': 'Yo-yo', 'Z': 'Zebra'
}

LETTERS = list(string.ascii_uppercase)
COLS_TIME = 64
COLS_CHARS = 26
COLS_TOTAL = 90
N_MELS = 80

# Capture slightly more audio (0.35s) to get the full phoneme
FRAMES_TO_KEEP = 25 

all_inputs = []
all_targets = []

print("Generating Smart Phoneme Dataset...")

plt.figure(figsize=(15, 8))

for i, char in enumerate(LETTERS):
    word = PHONEME_MAP.get(char, char)
    filename = f"phoneme_{char}.mp3"
    
    # 1. Generate Word
    tts = gTTS(word, lang='en')
    tts.save(filename)
    y, sr = librosa.load(filename, sr=22050)
    
    # 2. AGGRESSIVE TRIM (The Fix)
    # top_db=20 means "Cut anything quieter than the main voice"
    yt, _ = librosa.effects.trim(y, top_db=20)
    
    # 3. Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=N_MELS, hop_length=256)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    target = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    target = target.T 
    
    # 4. SMART SLICE
    # Ensure we have enough data, pad if short
    if target.shape[0] < FRAMES_TO_KEEP:
        # Pad with zeros if the sound is too short (e.g. "It")
        pad_amt = FRAMES_TO_KEEP - target.shape[0]
        target = np.pad(target, ((0, pad_amt), (0, 0)), mode='constant')
    else:
        # Take the first N frames
        target = target[:FRAMES_TO_KEEP]
    
    # DEBUG: Plot the first 5 letters to ensure they aren't empty
    if i < 5:
        plt.subplot(1, 5, i+1)
        plt.title(f"'{char}' Input")
        plt.imshow(target.T, aspect='auto', origin='lower')

    num_frames = target.shape[0]
    
    # 5. Create Inputs
    inputs = np.zeros((num_frames, COLS_TOTAL), dtype=np.float32)
    for t in range(num_frames):
        time_val = t / num_frames
        
        for k in range(10):
            freq = 2.0 ** k
            inputs[t, 2*k]     = np.sin(time_val * freq * 3.1415)
            inputs[t, 2*k + 1] = np.cos(time_val * freq * 3.1415)
        
        inputs[t, 63] = 1.0 
        inputs[t, COLS_TIME + i] = 1.0

    all_inputs.append(inputs)
    all_targets.append(target)
    print(f"  Processed '{char}' (Kept {num_frames} frames)")

# Stack
final_input = np.vstack(all_inputs)
final_target = np.vstack(all_targets)

# Save
def save_bin(data, filename):
    data_u32 = data.astype(np.float32).view(np.uint32)
    data_bf16 = (data_u32 >> 16).astype(np.uint16)
    data_bf16.tofile(filename)

save_bin(final_input, "../flux_vs_cpp/flux_proto/input_alphabet.bin")
save_bin(final_target, "../flux_vs_cpp/flux_proto/target_alphabet.bin")

plt.tight_layout()
plt.show() # <--- LOOK AT THIS POPUP
print(f"Saved Smart Dataset. Batch: {final_input.shape[0]}")