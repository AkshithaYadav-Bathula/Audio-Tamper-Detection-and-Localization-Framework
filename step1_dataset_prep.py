import os
import numpy as np
import librosa
import pickle
import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = r"A:\audio_watermark\dataset\data\TRAIN"
OUTPUT_PATH  = r"A:\audio_watermark\output"
SR           = 16000
FRAME_LEN    = int(0.025 * SR)
HOP_LEN      = int(0.010 * SR)
N_MFCC       = 13
MAX_FILES    = 600

os.makedirs(OUTPUT_PATH, exist_ok=True)

def extract_features(y):
    mfcc     = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=FRAME_LEN, hop_length=HOP_LEN)
    delta    = librosa.feature.delta(mfcc)
    zcr      = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LEN, hop_length=HOP_LEN)
    pitch    = librosa.yin(y, fmin=50, fmax=400, sr=SR, frame_length=FRAME_LEN, hop_length=HOP_LEN).reshape(1, -1)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=FRAME_LEN, hop_length=HOP_LEN)
    rms      = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN)
    min_len  = min(mfcc.shape[1], delta.shape[1], zcr.shape[1],
                   pitch.shape[1], rolloff.shape[1], rms.shape[1])
    feats    = np.vstack([
        mfcc[:, :min_len],
        delta[:, :min_len],
        zcr[:, :min_len],
        pitch[:, :min_len],
        rolloff[:, :min_len],
        rms[:, :min_len]
    ]).T
    return feats

def attack_noise(y):
    # stronger noise — SNR ~10db, clearly audible
    y = y.copy()
    n = len(y)
    s, e = int(n * 0.25), int(n * 0.65)
    sig_pow   = np.mean(y[s:e] ** 2) + 1e-10
    noise_pow = sig_pow / (10 ** (10 / 10))
    y[s:e]    = np.clip(y[s:e] + np.random.normal(0, np.sqrt(noise_pow), e - s), -1.0, 1.0)
    return y, s, e

def attack_deletion(y):
    # zero out middle — complete silence = very obvious in features
    y = y.copy()
    n = len(y)
    s, e = int(n * 0.30), int(n * 0.60)
    y[s:e] = 0.0
    return y, s, e

def attack_splice(y, y2):
    # replace middle third with different speaker
    y = y.copy()
    n = len(y)
    s, e = int(n * 0.33), int(n * 0.66)
    if len(y2) < e:
        return y, None, None
    y[s:e] = y2[s:e]
    return y, s, e

def attack_amplify(y):
    # amplify middle region — RMS changes dramatically
    y = y.copy()
    n = len(y)
    s, e = int(n * 0.20), int(n * 0.55)
    y[s:e] = np.clip(y[s:e] * 4.0, -1.0, 1.0)
    return y, s, e

def collect_wav_files(root, max_files):
    files = []
    for dr in sorted(os.listdir(root)):
        dr_path = os.path.join(root, dr)
        if not os.path.isdir(dr_path):
            continue
        for spk in sorted(os.listdir(dr_path)):
            spk_path = os.path.join(dr_path, spk)
            if not os.path.isdir(spk_path):
                continue
            for w in os.listdir(spk_path):
                if w.upper().endswith(".WAV"):
                    files.append(os.path.join(spk_path, w))
                    if len(files) >= max_files:
                        return files
    return files

print("Scanning TIMIT TRAIN folders...")
all_files = collect_wav_files(DATASET_PATH, MAX_FILES)
print(f"Using {len(all_files)} audio files")
print(f"Features per frame: 30 (MFCC x13 + DMFCC x13 + ZCR + Pitch + Rolloff + RMS)")

X, y_labels = [], []

for idx, fpath in enumerate(all_files):
    try:
        y, _ = librosa.load(fpath, sr=SR, mono=True)
        if len(y) < SR * 0.5:
            continue

        # pick attack type — 4 attacks cycling
        attack_type = idx % 4

        if attack_type == 0:
            y_att, s_sample, e_sample = attack_noise(y)

        elif attack_type == 1:
            y_att, s_sample, e_sample = attack_deletion(y)

        elif attack_type == 2:
            other = all_files[(idx + 1) % len(all_files)]
            y2, _ = librosa.load(other, sr=SR, mono=True)
            y_att, s_sample, e_sample = attack_splice(y, y2)
            if s_sample is None:
                continue

        else:
            y_att, s_sample, e_sample = attack_amplify(y)

        # extract features from attacked audio
        feats_att = extract_features(y_att)
        n_att     = len(feats_att)
        if n_att < 10:
            continue

        # convert sample boundary to frame boundary
        s_frame = int((s_sample / len(y)) * n_att)
        e_frame = int((e_sample / len(y)) * n_att)

        # add ALL frames with correct labels
        for i, f in enumerate(feats_att):
            label = 0 if s_frame <= i < e_frame else 1
            X.append(f)
            y_labels.append(label)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(all_files)} files...")

    except Exception as ex:
        print(f"  Skipped {fpath}: {ex}")
        continue

X        = np.array(X)
y_labels = np.array(y_labels)

auth_count = np.sum(y_labels == 1)
tamp_count = np.sum(y_labels == 0)

print(f"\nDataset ready:")
print(f"  Total frames : {len(X)}")
print(f"  Authentic (1): {auth_count}")
print(f"  Tampered  (0): {tamp_count}")
print(f"  Ratio        : {auth_count/tamp_count:.1f}:1")
print(f"  Feature size : {X.shape[1]}")

save_path = os.path.join(OUTPUT_PATH, "dataset.pkl")
with open(save_path, "wb") as f:
    pickle.dump({"X": X, "y": y_labels}, f)

print(f"\nSaved to: {save_path}")
print("Step 1 complete. Run step2_train.py next.")
