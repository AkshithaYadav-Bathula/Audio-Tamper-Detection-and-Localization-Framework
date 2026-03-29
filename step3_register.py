import os
import numpy as np
import librosa
import pickle
import warnings
warnings.filterwarnings('ignore')

OUTPUT_PATH = r"A:\audio_watermark\output"
SR          = 16000
FRAME_LEN   = int(0.025 * SR)
HOP_LEN     = int(0.010 * SR)
N_MFCC      = 13

# ── Henon map key generator ──────────────────────────────────────────────────
def henon_key(length, x0=0.1, y0=0.3, a=1.4, b=0.3):
    x, y   = x0, y0
    bits   = []
    for _ in range(length):
        x, y = 1 - a * x * x + y, b * x
        bits.append(1 if x > 0 else 0)
    return np.array(bits, dtype=np.uint8)

# ── Feature extraction (same as step 1) ─────────────────────────────────────
def extract_features(y):
    mfcc    = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=FRAME_LEN, hop_length=HOP_LEN)
    delta   = librosa.feature.delta(mfcc)
    zcr     = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LEN, hop_length=HOP_LEN)
    pitch   = librosa.yin(y, fmin=50, fmax=400, sr=SR, frame_length=FRAME_LEN, hop_length=HOP_LEN).reshape(1, -1)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=FRAME_LEN, hop_length=HOP_LEN)
    rms     = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN)
    min_len = min(mfcc.shape[1], delta.shape[1], zcr.shape[1],
                  pitch.shape[1], rolloff.shape[1], rms.shape[1])
    return np.vstack([
        mfcc[:, :min_len],
        delta[:, :min_len],
        zcr[:, :min_len],
        pitch[:, :min_len],
        rolloff[:, :min_len],
        rms[:, :min_len]
    ]).T

# ── Binarize feature matrix ──────────────────────────────────────────────────
def binarize(feats):
    thresholds = np.mean(feats, axis=0)
    binary     = (feats >= thresholds).astype(np.uint8)
    return binary, thresholds

# ── Register one audio file ──────────────────────────────────────────────────
def register(audio_path):
    print(f"Registering: {audio_path}")

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    print(f"  Duration     : {len(y)/SR:.2f} seconds")

    feats = extract_features(y)
    print(f"  Total frames : {len(feats)}")
    print(f"  Feature size : {feats.shape[1]}")

    binary, thresholds = binarize(feats)

    flat_binary = binary.flatten()
    key         = henon_key(len(flat_binary))
    encrypted   = np.bitwise_xor(flat_binary, key)

    print(f"  Binary bits  : {len(flat_binary)}")
    print(f"  Key bits     : {len(key)}")

    data = {
        "audio_path"  : audio_path,
        "encrypted"   : encrypted,
        "thresholds"  : thresholds,
        "n_frames"    : len(feats),
        "n_features"  : feats.shape[1],
        "henon_x0"    : 0.1,
        "henon_y0"    : 0.3,
        "henon_a"     : 1.4,
        "henon_b"     : 0.3,
    }

    save_path = os.path.join(OUTPUT_PATH, "registered.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\nFingerprint saved to: {save_path}")
    print("Step 3 complete. Run step4_verify.py next.")
    return data

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Change this path to any TIMIT .WAV file you want to register
    AUDIO_TO_REGISTER = r"A:\audio_watermark\dataset\data\TEST\DR2\FDRD1\SA1.WAV"

    if not os.path.exists(AUDIO_TO_REGISTER):
        print("ERROR: Audio file not found.")
        print("Please update AUDIO_TO_REGISTER path to a valid .WAV file.")
        print("Example: A:\\audio_watermark\\dataset\\data\\TEST\\DR1\\FAKS0\\SA1.WAV")
        print("\nAvailable TEST speakers:")
        test_path = r"A:\audio_watermark\dataset\data\TEST\DR1"
        if os.path.exists(test_path):
            for spk in sorted(os.listdir(test_path))[:5]:
                print(f"  {spk}")
    else:
        register(AUDIO_TO_REGISTER)
