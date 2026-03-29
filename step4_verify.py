import os
import numpy as np
import librosa
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUTPUT_PATH = r"A:\audio_watermark\output"
SR          = 16000
FRAME_LEN   = int(0.025 * SR)
HOP_LEN     = int(0.010 * SR)
N_MFCC      = 13

BER_THRESHOLD = 0.35
NC_THRESHOLD  = 0.60

# ── Henon map key generator ──────────────────────────────────────────────────
def henon_key(length, x0=0.1, y0=0.3, a=1.4, b=0.3):
    x, y = x0, y0
    bits = []
    for _ in range(length):
        x, y = 1 - a * x * x + y, b * x
        bits.append(1 if x > 0 else 0)
    return np.array(bits, dtype=np.uint8)

# ── Feature extraction (identical to step 1 and step 3) ─────────────────────
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

# ── Binarize using stored thresholds ─────────────────────────────────────────
def binarize_with_thresholds(feats, thresholds):
    return (feats >= thresholds).astype(np.uint8)

# ── BER ───────────────────────────────────────────────────────────────────────
def compute_ber(bits1, bits2):
    min_len = min(len(bits1), len(bits2))
    return np.sum(bits1[:min_len] != bits2[:min_len]) / min_len

# ── NC ────────────────────────────────────────────────────────────────────────
def compute_nc(bits1, bits2):
    min_len = min(len(bits1), len(bits2))
    b1 = bits1[:min_len].astype(float)
    b2 = bits2[:min_len].astype(float)
    num  = np.sum(b1 * b2)
    den  = np.sqrt(np.sum(b1 ** 2) * np.sum(b2 ** 2))
    return num / den if den > 0 else 0.0

# ── Create tampered test audio ────────────────────────────────────────────────
def create_tampered(y, attack="deletion"):
    y_att = y.copy()
    n     = len(y_att)
    if attack == "deletion":
        s, e     = int(n * 0.35), int(n * 0.60)
        y_att[s:e] = 0.0
        true_s, true_e = s, e
    elif attack == "noise":
        s, e     = int(n * 0.25), int(n * 0.65)
        sig_pow  = np.mean(y[s:e] ** 2) + 1e-10
        noise_pow = sig_pow / (10 ** (10 / 10))
        y_att[s:e] = np.clip(y_att[s:e] + np.random.normal(0, np.sqrt(noise_pow), e - s), -1.0, 1.0)
        true_s, true_e = s, e
    elif attack == "splice":
        s, e     = int(n * 0.40), int(n * 0.70)
        noise    = np.random.uniform(-0.3, 0.3, e - s).astype(np.float32)
        y_att[s:e] = noise
        true_s, true_e = s, e
    return y_att, true_s, true_e

# ── Save tampered audio ───────────────────────────────────────────────────────
def save_wav(path, y):
    import soundfile as sf
    sf.write(path, y, SR)

# ── Plot heatmap ──────────────────────────────────────────────────────────────
def plot_heatmap(frame_labels, n_frames, duration, title, save_path, true_s_frame=None, true_e_frame=None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # frame-level bar
    ax1 = axes[0]
    colors = ['red' if l == 0 else 'green' for l in frame_labels]
    ax1.bar(range(len(frame_labels)), [1] * len(frame_labels), color=colors, width=1.0, edgecolor='none')
    ax1.set_xlim(0, len(frame_labels))
    ax1.set_yticks([])
    ax1.set_xlabel("Frame index")
    ax1.set_title("Frame-level prediction   (green = authentic,  red = tampered)")

    if true_s_frame is not None:
        ax1.axvline(x=true_s_frame, color='blue', linestyle='--', linewidth=1.5, label='True tamper start')
        ax1.axvline(x=true_e_frame, color='orange', linestyle='--', linewidth=1.5, label='True tamper end')
        ax1.legend(fontsize=8)

    # time-domain waveform coloured by prediction
    ax2 = axes[1]
    times = np.linspace(0, duration, n_frames)
    for i, label in enumerate(frame_labels):
        color = 'red' if label == 0 else 'green'
        ax2.axvspan(times[i], times[min(i + 1, n_frames - 1)], alpha=0.3, color=color)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_title("Time-domain tamper map")
    ax2.set_xlim(0, duration)
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap saved: {save_path}")

# ── Main verification function ────────────────────────────────────────────────
def verify(audio_path, label=""):
    print(f"\n{'='*55}")
    print(f"Verifying: {os.path.basename(audio_path)}  {label}")
    print(f"{'='*55}")

    # load registered fingerprint
    with open(os.path.join(OUTPUT_PATH, "registered.pkl"), "rb") as f:
        reg = pickle.load(f)

    # load trained model
    with open(os.path.join(OUTPUT_PATH, "svm_model.pkl"), "rb") as f:
        mdl = pickle.load(f)
    clf    = mdl["model"]
    scaler = mdl["scaler"]

    # load query audio
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    duration = len(y) / SR

    # extract features
    feats = extract_features(y)
    n_frames = len(feats)

    # ── SVM per-frame prediction ──
    feats_scaled  = scaler.transform(feats)
    frame_preds   = clf.predict(feats_scaled)

    tampered_frames = np.where(frame_preds == 0)[0]
    n_tampered      = len(tampered_frames)
    tamper_ratio    = n_tampered / n_frames

    # ── BER + NC fingerprint comparison ──
    binary_query = binarize_with_thresholds(feats, reg["thresholds"])
    flat_query   = binary_query.flatten()

    key          = henon_key(len(flat_query), reg["henon_x0"], reg["henon_y0"],
                             reg["henon_a"], reg["henon_b"])
    decrypted_reg = np.bitwise_xor(reg["encrypted"][:len(flat_query)], key[:len(flat_query)])

    ber = compute_ber(flat_query, decrypted_reg)
    nc  = compute_nc(flat_query.astype(float), decrypted_reg.astype(float))

    # ── Global decision ──
    svm_tampered = tamper_ratio > 0.05
    fp_tampered  = (ber > BER_THRESHOLD) or (nc < NC_THRESHOLD)
    is_tampered  = svm_tampered or fp_tampered

    # ── Print results ──
    print(f"\n  Total frames     : {n_frames}")
    print(f"  Tampered frames  : {n_tampered}  ({tamper_ratio*100:.1f}%)")
    print(f"  BER              : {ber:.4f}  (threshold < {BER_THRESHOLD})")
    print(f"  NC               : {nc:.4f}  (threshold > {NC_THRESHOLD})")
    print(f"\n  SVM decision     : {'TAMPERED' if svm_tampered else 'AUTHENTIC'}")
    print(f"  Fingerprint check: {'TAMPERED' if fp_tampered  else 'AUTHENTIC'}")
    print(f"\n{'─'*40}")
    print(f"  FINAL RESULT     : {'⚠ TAMPERED' if is_tampered else '✓ AUTHENTIC'}")
    print(f"{'─'*40}")

    if is_tampered and n_tampered > 0:
        s_frame = tampered_frames[0]
        e_frame = tampered_frames[-1]
        s_time  = s_frame * HOP_LEN / SR
        e_time  = e_frame * HOP_LEN / SR
        print(f"\n  Tampered region  : frames {s_frame} to {e_frame}")
        print(f"  Tampered time    : {s_time:.2f}s to {e_time:.2f}s")

    return frame_preds, n_frames, duration, is_tampered

# ── Run all 3 test cases ──────────────────────────────────────────────────────
if __name__ == "__main__":
    with open(os.path.join(OUTPUT_PATH, "registered.pkl"), "rb") as f:
        reg = pickle.load(f)
    original_path = reg["audio_path"]

    y_orig, _ = librosa.load(original_path, sr=SR, mono=True)

    # Test 1: original clean audio → should say AUTHENTIC
    preds, n, dur, _ = verify(original_path, "(original - should be AUTHENTIC)")
    plot_heatmap(preds, n, dur,
                 "Test 1: Original audio (expected: AUTHENTIC)",
                 os.path.join(OUTPUT_PATH, "heatmap_authentic.png"))

    # Test 2: deletion attack → should say TAMPERED + show where
    del_path = os.path.join(OUTPUT_PATH, "test_deletion.wav")
    y_del, true_s, true_e = create_tampered(y_orig, attack="deletion")
    save_wav(del_path, y_del)
    true_s_frame = int((true_s / len(y_orig)) * (len(y_orig) // HOP_LEN))
    true_e_frame = int((true_e / len(y_orig)) * (len(y_orig) // HOP_LEN))
    preds, n, dur, _ = verify(del_path, "(deletion attack - should be TAMPERED)")
    plot_heatmap(preds, n, dur,
                 "Test 2: Deletion attack (expected: TAMPERED)",
                 os.path.join(OUTPUT_PATH, "heatmap_deletion.png"),
                 true_s_frame, true_e_frame)

    # Test 3: noise attack → should say TAMPERED + show where
    noise_path = os.path.join(OUTPUT_PATH, "test_noise.wav")
    y_noise, true_s, true_e = create_tampered(y_orig, attack="noise")
    save_wav(noise_path, y_noise)
    true_s_frame = int((true_s / len(y_orig)) * (len(y_orig) // HOP_LEN))
    true_e_frame = int((true_e / len(y_orig)) * (len(y_orig) // HOP_LEN))
    preds, n, dur, _ = verify(noise_path, "(noise attack - should be TAMPERED)")
    plot_heatmap(preds, n, dur,
                 "Test 3: Noise attack (expected: TAMPERED)",
                 os.path.join(OUTPUT_PATH, "heatmap_noise.png"),
                 true_s_frame, true_e_frame)

    print(f"\n{'='*55}")
    print("All tests complete.")
    print(f"Heatmaps saved in: {OUTPUT_PATH}")
    print("  heatmap_authentic.png")
    print("  heatmap_deletion.png")
    print("  heatmap_noise.png")
