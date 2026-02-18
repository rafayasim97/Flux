import numpy as np
import librosa
import soundfile as sf
from gtts import gTTS
import os

# 1. Generate Real Audio
print("Generating clean 'Hello'...")
tts = gTTS("Hello", lang='en')
tts.save("real_hello.mp3")

# Load it
y, sr = librosa.load("real_hello.mp3", sr=22050)
# Trim silence
y, _ = librosa.effects.trim(y)

# 2. Convert to Spectrogram (Mel Scale)
# This is how AI hears sound
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
# Convert to Log scale (Decibels) - Crucial for speech
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Normalize to [0.0, 1.0] range for the Neural Network
mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

# 3. Extract 5 Representative Frames (H, E, L, L, O)
# The word "Hello" has duration. We need to pick 5 distinct moments.
total_frames = mel_spec_norm.shape[1]
indices = np.linspace(0, total_frames-1, 5, dtype=int)

# Target: 5 Rows x 80 Columns
target_vec = mel_spec_norm[:, indices].T # Transpose to match (Rows, Cols)

# Input: Standard Identity Matrix + Bias
input_vec = np.zeros((5, 64), dtype=np.float32)
for i in range(5):
    input_vec[i, i] = 1.0
    input_vec[i, 63] = 1.0 # Bias

# 4. Save
def save_bin(data, filename):
    data_u32 = data.astype(np.float32).view(np.uint32)
    data_bf16 = (data_u32 >> 16).astype(np.uint16)
    data_bf16.tofile(filename)
    print(f"Saved {filename} {data.shape}")

save_bin(input_vec, "input_hello.bin")
save_bin(target_vec, "target_hello.bin")

print("\nProcessing Complete.")
print(f"Captured 5 frames from real audio.")