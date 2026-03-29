# import os
# import numpy as np
# import librosa
# import pickle
# import warnings
# warnings.filterwarnings('ignore')

# DATASET_PATH = r"A:\audio_watermark\dataset\data\TRAIN"
# OUTPUT_PATH  = r"A:\audio_watermark\output"
# SR           = 16000
# FRAME_LEN    = int(0.025 * SR)
# HOP_LEN      = int(0.010 * SR)
# N_MFCC       = 13
# MAX_FILES    = 400

# os.makedirs(OUTPUT_PATH, exist_ok=True)

# def extract_features(y):
#     mfcc  = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=FRAME_LEN, hop_length=HOP_LEN)
#     delta = librosa.feature.delta(mfcc)
#     zcr   = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LEN, hop_length=HOP_LEN)
#     pitch = librosa.yin(y, fmin=50, fmax=400, sr=SR, frame_length=FRAME_LEN, hop_length=HOP_LEN).reshape(1, -1)
#     min_len = min(mfcc.shape[1], delta.shape[1], zcr.shape[1], pitch.shape[1])
#     feats = np.vstack([mfcc[:, :min_len], delta[:, :min_len], zcr[:, :min_len], pitch[:, :min_len]]).T
#     return feats

# def attack_noise(y):
#     y = y.copy()
#     n = len(y)
#     s = int(n * 0.25)
#     e = int(n * 0.60)
#     y[s:e] = np.clip(y[s:e] + np.random.normal(0, 0.008, e - s), -1.0, 1.0)
#     return y, s, e

# def attack_deletion(y):
#     y = y.copy()
#     n = len(y)
#     s = int(n * 0.30)
#     e = int(n * 0.55)
#     y[s:e] = 0.0
#     return y, s, e

# def attack_splice(y, y2):
#     y = y.copy()
#     n = len(y)
#     s = int(n * 0.33)
#     e = int(n * 0.66)
#     if len(y2) < e:
#         return y, None, None
#     y[s:e] = y2[s:e]
#     return y, s, e

# def collect_wav_files(root, max_files):
#     files = []
#     for dr in sorted(os.listdir(root)):
#         dr_path = os.path.join(root, dr)
#         if not os.path.isdir(dr_path):
#             continue
#         for spk in sorted(os.listdir(dr_path)):
#             spk_path = os.path.join(dr_path, spk)
#             if not os.path.isdir(spk_path):
#                 continue
#             for w in os.listdir(spk_path):
#                 if w.upper().endswith(".WAV"):
#                     files.append(os.path.join(spk_path, w))
#                     if len(files) >= max_files:
#                         return files
#     return files

# print("Scanning TIMIT TRAIN folders...")
# all_files = collect_wav_files(DATASET_PATH, MAX_FILES)
# print(f"Using {len(all_files)} audio files")

# X, y_labels = [], []

# for idx, fpath in enumerate(all_files):
#     try:
#         y, _ = librosa.load(fpath, sr=SR, mono=True)
#         if len(y) < SR * 0.5:
#             continue

#         feats_clean = extract_features(y)
#         n_frames    = len(feats_clean)
#         if n_frames < 10:
#             continue

#         # Pass 1: ALL clean frames → label 1 (authentic)
#         for f in feats_clean:
#             X.append(f)
#             y_labels.append(1)

#         # Pass 2: ALL frames from attacked audio
#         # tampered region → label 0, rest → label 1
#         attack_type = idx % 3

#         if attack_type == 0:
#             y_att, s_sample, e_sample = attack_noise(y)

#         elif attack_type == 1:
#             y_att, s_sample, e_sample = attack_deletion(y)

#         else:
#             other = all_files[(idx + 1) % len(all_files)]
#             y2, _ = librosa.load(other, sr=SR, mono=True)
#             y_att, s_sample, e_sample = attack_splice(y, y2)
#             if s_sample is None:
#                 continue

#         feats_att = extract_features(y_att)
#         n_att     = len(feats_att)

#         # convert sample boundary to frame boundary
#         s_frame = int((s_sample / len(y)) * n_att)
#         e_frame = int((e_sample / len(y)) * n_att)

#         # add ALL frames from attacked audio with correct labels
#         for i, f in enumerate(feats_att):
#             label = 0 if s_frame <= i < e_frame else 1
#             X.append(f)
#             y_labels.append(label)

#         if (idx + 1) % 50 == 0:
#             print(f"  Processed {idx+1}/{len(all_files)} files...")

#     except Exception as ex:
#         print(f"  Skipped {fpath}: {ex}")
#         continue

# X        = np.array(X)
# y_labels = np.array(y_labels)

# auth_count = np.sum(y_labels == 1)
# tamp_count = np.sum(y_labels == 0)

# print(f"\nDataset ready:")
# print(f"  Total frames : {len(X)}")
# print(f"  Authentic (1): {auth_count}")
# print(f"  Tampered  (0): {tamp_count}")
# print(f"  Ratio        : {auth_count/tamp_count:.1f}:1")
# print(f"  Feature size : {X.shape[1]}")

# save_path = os.path.join(OUTPUT_PATH, "dataset.pkl")
# with open(save_path, "wb") as f:
#     pickle.dump({"X": X, "y": y_labels}, f)

# print(f"\nSaved to: {save_path}")
# print("Step 1 complete. Run step2_train.py next.")
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
    mfcc  = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=FRAME_LEN, hop_length=HOP_LEN)
    delta = librosa.feature.delta(mfcc)
    zcr   = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LEN, hop_length=HOP_LEN)
    pitch = librosa.yin(y, fmin=50, fmax=400, sr=SR, frame_length=FRAME_LEN, hop_length=HOP_LEN).reshape(1, -1)
    min_len = min(mfcc.shape[1], delta.shape[1], zcr.shape[1], pitch.shape[1])
    return np.vstack([mfcc[:, :min_len], delta[:, :min_len], zcr[:, :min_len], pitch[:, :min_len]]).T

def attack_noise(y):
    y = y.copy()
    n = len(y)
    s, e = int(n * 0.25), int(n * 0.60)
    y[s:e] = np.clip(y[s:e] + np.random.normal(0, 0.008, e - s), -1.0, 1.0)
    return y, s, e

def attack_deletion(y):
    y = y.copy()
    n = len(y)
    s, e = int(n * 0.30), int(n * 0.55)
    y[s:e] = 0.0
    return y, s, e

def attack_splice(y, y2):
    y = y.copy()
    n = len(y)
    s, e = int(n * 0.33), int(n * 0.66)
    if len(y2) < e:
        return y, None, None
    y[s:e] = y2[s:e]
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

X, y_labels = [], []

for idx, fpath in enumerate(all_files):
    try:
        y, _ = librosa.load(fpath, sr=SR, mono=True)
        if len(y) < SR * 0.5:
            continue

        # apply attack
        attack_type = idx % 3

        if attack_type == 0:
            y_att, s_sample, e_sample = attack_noise(y)

        elif attack_type == 1:
            y_att, s_sample, e_sample = attack_deletion(y)

        else:
            other = all_files[(idx + 1) % len(all_files)]
            y2, _ = librosa.load(other, sr=SR, mono=True)
            y_att, s_sample, e_sample = attack_splice(y, y2)
            if s_sample is None:
                continue

        # extract features from attacked audio only
        feats_att = extract_features(y_att)
        n_att     = len(feats_att)
        if n_att < 10:
            continue

        # convert sample boundary to frame boundary
        s_frame = int((s_sample / len(y)) * n_att)
        e_frame = int((e_sample / len(y)) * n_att)

        # add ALL frames: tampered region = 0, rest = 1
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
