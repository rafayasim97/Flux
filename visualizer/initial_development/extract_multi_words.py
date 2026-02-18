import numpy as np
import librosa
from gtts import gTTS
import os

# CONFIG
WORDS = ["Hello", "Flux"]
COLS_IN = 70  # Increased from 64 to make room for flags
N_MELS = 80

def process_word(word_text, word_id):
    print(f"Processing '{word_text}'...")
    filename = f"real_{word_text}.mp3"
    
    # 1. Generate Audio
    tts = gTTS(word_text, lang='en')
    tts.save(filename)
    y, sr = librosa.load(filename, sr=22050)
    y, _ = librosa.effects.trim(y)

    # 2. Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=256)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize [0, 1]
    target = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    target = target.T # (Time, 80)
    
    num_frames = target.shape[0]
    
    # 3. Create Inputs
    inputs = np.zeros((num_frames, COLS_IN), dtype=np.float32)
    for t in range(num_frames):
        # Time Reset (0.0 to 1.0) for EACH word
        time_val = t / num_frames
        
        # Positional Encodings (Cols 0-20)
        for k in range(10):
            freq = 2.0 ** k
            inputs[t, 2*k]     = np.sin(time_val * freq * 3.1415)
            inputs[t, 2*k + 1] = np.cos(time_val * freq * 3.1415)
            
        # Bias (Col 63)
        inputs[t, 63] = 1.0
        
        # --- THE WORD SELECTOR ---
        # If Word 0 (Hello), set Col 64 = 1.0
        # If Word 1 (Flux),  set Col 65 = 1.0
        inputs[t, 64 + word_id] = 1.0 

    return inputs, target

# Main Loop
all_inputs = []
all_targets = []

for i, word in enumerate(WORDS):
    inp, tar = process_word(word, i)
    all_inputs.append(inp)
    all_targets.append(tar)

# Concatenate into one massive dataset
final_input = np.vstack(all_inputs)
final_target = np.vstack(all_targets)

print(f"\nTotal Batch Size: {final_input.shape[0]}") # <--- REMEMBER THIS NUMBER

# Save
def save_bin(data, filename):
    data_u32 = data.astype(np.float32).view(np.uint32)
    data_bf16 = (data_u32 >> 16).astype(np.uint16)
    data_bf16.tofile(filename)

save_bin(final_input, "input_multi.bin")
save_bin(final_target, "target_multi.bin")
print("Saved input_multi.bin & target_multi.bin")