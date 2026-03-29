# import os
# import numpy as np
# import librosa
# import soundfile as sf
# import pickle
# import warnings
# warnings.filterwarnings('ignore')

# DATASET_PATH = r"A:\audio_watermark\dataset\data\TRAIN"
# OUTPUT_PATH  = r"A:\audio_watermark\output"
# SR           = 16000
# FRAME_LEN    = int(0.025 * SR)
# HOP_LEN      = int(0.010 * SR)
# N_MFCC       = 13
# MAX_SPEAKERS = 50
# MAX_FILES    = 5

# os.makedirs(OUTPUT_PATH, exist_ok=True)

# def extract_features(y, sr=SR):
#     frames = librosa.util.frame(y, frame_length=FRAME_LEN, hop_length=HOP_LEN).T
#     feats = []
#     for f in frames:
#         if len(f) < FRAME_LEN:
#             continue
#         mfcc  = librosa.feature.mfcc(y=f, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_LEN).mean(axis=1)
#         dmfcc = np.diff(mfcc, prepend=mfcc[0])
#         zcr   = np.array([librosa.feature.zero_crossing_rate(f, frame_length=FRAME_LEN, hop_length=FRAME_LEN+1)[0, 0]])
#         pitches, _ = librosa.piptrack(y=f, sr=sr, n_fft=FRAME_LEN)
#         pitch = np.array([pitches[pitches > 0].mean() if pitches[pitches > 0].size > 0 else 0.0])
#         feats.append(np.concatenate([mfcc, dmfcc, zcr, pitch]))
#     return np.array(feats)

# def attack_noise(y, snr_db=25):
#     sig_pow   = np.mean(y ** 2)
#     noise_pow = sig_pow / (10 ** (snr_db / 10))
#     noise     = np.random.normal(0, np.sqrt(noise_pow), len(y))
#     return np.clip(y + noise, -1.0, 1.0)

# def attack_deletion(y, start_ratio=0.3, end_ratio=0.5):
#     s, e    = int(len(y) * start_ratio), int(len(y) * end_ratio)
#     silence = np.zeros(e - s)
#     return np.concatenate([y[:s], silence, y[e:]])

# def attack_splice(y, y2):
#     cut = len(y) // 3
#     end = cut + (len(y) - cut)
#     if len(y2) < end:
#         return y
#     return np.concatenate([y[:cut], y2[cut:end]])

# def collect_wav_files(root, max_speakers, max_files):
#     files = []
#     speakers = sorted(os.listdir(root))[:max_speakers]
#     for spk in speakers:
#         spk_path = os.path.join(root, spk)
#         if not os.path.isdir(spk_path):
#             continue
#         wavs = [f for f in os.listdir(spk_path) if f.upper().endswith(".WAV")][:max_files]
#         for w in wavs:
#             files.append(os.path.join(spk_path, w))
#     return files

# print("Scanning TIMIT TRAIN folders...")
# all_files = []
# for dr in sorted(os.listdir(DATASET_PATH)):
#     dr_path = os.path.join(DATASET_PATH, dr)
#     if os.path.isdir(dr_path):
#         all_files += collect_wav_files(dr_path, MAX_SPEAKERS, MAX_FILES)

# print(f"Found {len(all_files)} audio files")

# X, y_labels = [], []

# for idx, fpath in enumerate(all_files):
#     try:
#         y, _ = librosa.load(fpath, sr=SR, mono=True)
#         if len(y) < SR * 0.5:
#             continue

#         feats_clean = extract_features(y)
#         n = len(feats_clean)
#         if n < 10:
#             continue

#         # Pass 1: all clean frames → label 1 (authentic)
#         for f in feats_clean:
#             X.append(f)
#             y_labels.append(1)

#         # Pass 2: attacked audio features → tampered region gets label 0
#         attack_type = idx % 3

#         if attack_type == 0:
#             # Noise: extract features from noisy audio, label middle as tampered
#             y_att     = attack_noise(y)
#             feats_att = extract_features(y_att)
#             n_att     = len(feats_att)
#             s, e      = int(n_att * 0.25), int(n_att * 0.60)
#             for i, f in enumerate(feats_att):
#                 X.append(f)
#                 y_labels.append(0 if s <= i < e else 1)

#         elif attack_type == 1:
#             # Deletion: silence inserted in middle, those frames are tampered
#             y_att     = attack_deletion(y)
#             feats_att = extract_features(y_att)
#             n_att     = len(feats_att)
#             s, e      = int(n_att * 0.30), int(n_att * 0.50)
#             for i, f in enumerate(feats_att):
#                 X.append(f)
#                 y_labels.append(0 if s <= i < e else 1)

#         else:
#             # Splice: second third onwards replaced by different speaker → tampered
#             other     = all_files[(idx + 1) % len(all_files)]
#             y2, _     = librosa.load(other, sr=SR, mono=True)
#             y_att     = attack_splice(y, y2)
#             if len(y_att) < SR * 0.5:
#                 continue
#             feats_att = extract_features(y_att)
#             n_att     = len(feats_att)
#             s         = int(n_att * 0.33)
#             for i, f in enumerate(feats_att):
#                 X.append(f)
#                 y_labels.append(0 if i >= s else 1)

#         if (idx + 1) % 50 == 0:
#             print(f"  Processed {idx + 1}/{len(all_files)} files...")

#     except Exception as ex:
#         print(f"  Skipped {fpath}: {ex}")
#         continue

# X        = np.array(X)
# y_labels = np.array(y_labels)

# print(f"\nDataset ready:")
# print(f"  Total frames : {len(X)}")
# print(f"  Authentic (1): {np.sum(y_labels == 1)}")
# print(f"  Tampered  (0): {np.sum(y_labels == 0)}")
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
import random
import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = r"A:\audio_watermark\dataset\data\TRAIN"
OUTPUT_PATH  = r"A:\audio_watermark\output"

SR = 16000
FRAME_LEN = int(0.025 * SR)
HOP_LEN   = int(0.010 * SR)

N_MFCC = 13

MAX_SPEAKERS = 40
MAX_FILES    = 5

os.makedirs(OUTPUT_PATH, exist_ok=True)


def extract_features(y):

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=FRAME_LEN,
        hop_length=HOP_LEN
    )

    delta = librosa.feature.delta(mfcc)

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=FRAME_LEN,
        hop_length=HOP_LEN
    )

    pitch = librosa.yin(
        y,
        fmin=50,
        fmax=400,
        sr=SR,
        frame_length=FRAME_LEN,
        hop_length=HOP_LEN
    )

    pitch = pitch.reshape(1, -1)

    feats = np.vstack([mfcc, delta, zcr, pitch]).T

    return feats


def attack_noise_segment(y):

    y = y.copy()

    n = len(y)

    s = int(n * random.uniform(0.2, 0.4))
    e = int(n * random.uniform(0.5, 0.8))

    noise = np.random.normal(0, 0.005, e - s)

    y[s:e] += noise

    return y, s, e


def attack_delete_segment(y):

    y = y.copy()

    n = len(y)

    s = int(n * random.uniform(0.3, 0.5))
    e = int(n * random.uniform(0.55, 0.75))

    y[s:e] = 0

    return y, s, e


def attack_splice_segment(y, y2):

    y = y.copy()

    n = len(y)

    s = int(n * random.uniform(0.25, 0.45))
    e = int(n * random.uniform(0.55, 0.75))

    if len(y2) < e:
        return y, None, None

    y[s:e] = y2[s:e]

    return y, s, e


def collect_files():

    files = []

    for dr in sorted(os.listdir(DATASET_PATH)):

        dr_path = os.path.join(DATASET_PATH, dr)

        if not os.path.isdir(dr_path):
            continue

        speakers = sorted(os.listdir(dr_path))[:MAX_SPEAKERS]

        for spk in speakers:

            spk_path = os.path.join(dr_path, spk)

            if not os.path.isdir(spk_path):
                continue

            wavs = [f for f in os.listdir(spk_path) if f.endswith(".WAV")][:MAX_FILES]

            for w in wavs:

                files.append(os.path.join(spk_path, w))

    return files


print("Scanning dataset...")

all_files = collect_files()

print("Total files:", len(all_files))


X = []
y_labels = []


for idx, fpath in enumerate(all_files):

    try:

        y, _ = librosa.load(fpath, sr=SR)

        if len(y) < SR * 0.5:
            continue


        feats_clean = extract_features(y)

        for f in feats_clean:
            X.append(f)
            y_labels.append(1)


        attack_type = idx % 3


        if attack_type == 0:

            y_att, s, e = attack_noise_segment(y)


        elif attack_type == 1:

            y_att, s, e = attack_delete_segment(y)


        else:

            other = all_files[(idx + 1) % len(all_files)]

            y2, _ = librosa.load(other, sr=SR)

            y_att, s, e = attack_splice_segment(y, y2)

            if s is None:
                continue


        feats_att = extract_features(y_att)

        n_frames = len(feats_att)

        s_frame = int((s / len(y)) * n_frames)
        e_frame = int((e / len(y)) * n_frames)


        for i, f in enumerate(feats_att):

            if s_frame <= i <= e_frame:
                X.append(f)
                y_labels.append(0)


        if (idx + 1) % 50 == 0:

            print(f"Processed {idx+1}/{len(all_files)}")


    except Exception as ex:

        print("Skipped:", fpath, ex)

        continue


X = np.array(X)
y_labels = np.array(y_labels)


print("\nDataset ready:")
print("Total frames :", len(X))
print("Authentic (1):", np.sum(y_labels == 1))
print("Tampered (0) :", np.sum(y_labels == 0))
print("Feature size :", X.shape[1])


save_path = os.path.join(OUTPUT_PATH, "dataset.pkl")

with open(save_path, "wb") as f:

    pickle.dump({"X": X, "y": y_labels}, f)


print("\nSaved to:", save_path)
print("Step 1 complete.")