import numpy as np
import librosa
from gtts import gTTS

# 1. Generate Audio
print("Generating HD Audio Data...")
tts = gTTS("Hello", lang='en')
tts.save("real_hello.mp3")
y, sr = librosa.load("real_hello.mp3", sr=22050)
y, _ = librosa.effects.trim(y)

# 2. HD Spectrogram
# hop_length=256 means ~86 frames per second
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
# Normalize to [0, 1]
target_vec = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
target_vec = target_vec.T # Shape: (TimeSteps, 80)

# 3. Create Inputs: "Time Embeddings"
# We map Time (0.0 to 1.0) into sine waves so the AI learns sharp details.
num_frames = target_vec.shape[0]
COLS_IN = 64
input_vec = np.zeros((num_frames, COLS_IN), dtype=np.float32)

print(f"Captured {num_frames} frames of audio.")

for t in range(num_frames):
    # Normalized Time (0.0 to 1.0)
    time_val = t / num_frames
    
    # Position Encoding (The Secret Sauce)
    # Frequencies: 1, 2, 4, 8, 16...
    # This lets the AI learn both slow vowels and fast consonants.
    for k in range(10): 
        freq = 2.0 ** k
        input_vec[t, 2*k]     = np.sin(time_val * freq * 3.1415)
        input_vec[t, 2*k + 1] = np.cos(time_val * freq * 3.1415)
    
    # Bias
    input_vec[t, 63] = 1.0

# 4. Save
def save_bin(data, filename):
    data_u32 = data.astype(np.float32).view(np.uint32)
    data_bf16 = (data_u32 >> 16).astype(np.uint16)
    data_bf16.tofile(filename)

save_bin(input_vec, "input_hello.bin")
save_bin(target_vec, "target_hello.bin")
print(f"Saved binary files. Batch Size will be: {num_frames}")