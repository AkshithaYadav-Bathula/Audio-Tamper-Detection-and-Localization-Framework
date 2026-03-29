import os
import numpy as np
import librosa
import pickle
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# CHANGE ONLY THIS SECTION — add your audio file paths here
# ======================================================================
TEST_FILES = [
    r"A:\audio_watermark\dataset\data\TEST\DR2\FDRD1\SA1.WAV",
    r"A:\audio_watermark\dataset\data\TEST\DR2\FDRD1\SA2.WAV",
    r"A:\audio_watermark\dataset\data\TEST\DR2\FDRD1\SI1544.WAV",
    r"A:\audio_watermark\dataset\data\TEST\DR2\FDRD1\SI1566.WAV",
]
# ======================================================================

OUTPUT_PATH   = r"A:\audio_watermark\output"
SR            = 16000
FRAME_LEN     = int(0.025 * SR)
HOP_LEN       = int(0.010 * SR)
N_MFCC        = 13
BER_THRESHOLD = 0.35
NC_THRESHOLD  = 0.60

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Henon map ──────────────────────────────────────────────────────────
def henon_key(length, x0=0.1, y0=0.3, a=1.4, b=0.3):
    x, y = x0, y0
    bits = []
    for _ in range(length):
        x, y = 1 - a * x * x + y, b * x
        bits.append(1 if x > 0 else 0)
    return np.array(bits, dtype=np.uint8)

# ── Feature extraction ─────────────────────────────────────────────────
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
        mfcc[:, :min_len], delta[:, :min_len], zcr[:, :min_len],
        pitch[:, :min_len], rolloff[:, :min_len], rms[:, :min_len]
    ]).T

# ── BER and NC ─────────────────────────────────────────────────────────
def compute_ber(b1, b2):
    n = min(len(b1), len(b2))
    return np.sum(b1[:n] != b2[:n]) / n

def compute_nc(b1, b2):
    n = min(len(b1), len(b2))
    a, b = b1[:n].astype(float), b2[:n].astype(float)
    den  = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return np.sum(a * b) / den if den > 0 else 0.0

# ── Register one audio file ────────────────────────────────────────────
def register(audio_path):
    y, _   = librosa.load(audio_path, sr=SR, mono=True)
    feats  = extract_features(y)
    thresh = np.mean(feats, axis=0)
    binary = (feats >= thresh).astype(np.uint8).flatten()
    key    = henon_key(len(binary))
    enc    = np.bitwise_xor(binary, key)
    return {
        "audio_path": audio_path,
        "encrypted" : enc,
        "thresholds": thresh,
        "n_frames"  : len(feats),
        "n_features": feats.shape[1],
        "henon_x0"  : 0.1, "henon_y0": 0.3,
        "henon_a"   : 1.4, "henon_b" : 0.3,
    }

# ── Apply attacks ──────────────────────────────────────────────────────
def apply_attack(y, attack):
    y = y.copy()
    n = len(y)
    if attack == "noise":
        s, e = int(n * 0.25), int(n * 0.65)
        sig_pow   = np.mean(y[s:e] ** 2) + 1e-10
        noise_pow = sig_pow / (10 ** (10 / 10))
        y[s:e]    = np.clip(y[s:e] + np.random.normal(0, np.sqrt(noise_pow), e-s), -1.0, 1.0)
        return y, s, e
    elif attack == "deletion":
        s, e   = int(n * 0.30), int(n * 0.60)
        y[s:e] = 0.0
        return y, s, e
    elif attack == "splice":
        s, e   = int(n * 0.33), int(n * 0.66)
        noise  = np.random.uniform(-0.3, 0.3, e - s).astype(np.float32)
        y[s:e] = noise
        return y, s, e
    elif attack == "amplify":
        s, e   = int(n * 0.20), int(n * 0.55)
        y[s:e] = np.clip(y[s:e] * 4.0, -1.0, 1.0)
        return y, s, e

# ── Verify one audio ───────────────────────────────────────────────────
def verify(y, reg, clf, scaler):
    feats        = extract_features(y)
    n_frames     = len(feats)
    frame_preds  = clf.predict(scaler.transform(feats))
    n_tampered   = np.sum(frame_preds == 0)
    tamper_ratio = n_tampered / n_frames

    binary_q  = (feats >= reg["thresholds"]).astype(np.uint8).flatten()
    key       = henon_key(len(binary_q), reg["henon_x0"], reg["henon_y0"],
                          reg["henon_a"], reg["henon_b"])
    dec_reg   = np.bitwise_xor(reg["encrypted"][:len(binary_q)], key[:len(binary_q)])

    ber = compute_ber(binary_q, dec_reg)
    nc  = compute_nc(binary_q, dec_reg)

    svm_t    = tamper_ratio > 0.05
    fp_t     = (ber > BER_THRESHOLD) or (nc < NC_THRESHOLD)
    tampered = svm_t or fp_t

    tampered_idx = np.where(frame_preds == 0)[0]
    s_time = tampered_idx[0]  * HOP_LEN / SR if len(tampered_idx) > 0 else 0
    e_time = tampered_idx[-1] * HOP_LEN / SR if len(tampered_idx) > 0 else 0

    return {
        "frame_preds" : frame_preds,
        "n_frames"    : n_frames,
        "n_tampered"  : n_tampered,
        "tamper_ratio": tamper_ratio,
        "ber"         : ber,
        "nc"          : nc,
        "tampered"    : tampered,
        "s_time"      : s_time,
        "e_time"      : e_time,
    }

# ── Draw comparison heatmap (original vs tampered side by side) ────────
def draw_comparison(fname, duration, orig_res, att_res, attack_name, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 6))
    fig.suptitle(f"{fname}  —  Attack: {attack_name.upper()}", fontsize=13, fontweight='bold')

    orig_label  = "AUTHENTIC" if not orig_res["tampered"] else "TAMPERED"
    att_label   = "AUTHENTIC" if not att_res["tampered"]  else "TAMPERED"
    orig_color  = "green" if not orig_res["tampered"] else "red"
    att_color   = "red"   if att_res["tampered"]      else "green"

    # Top left — original frame bar
    colors_o = ['red' if l == 0 else 'green' for l in orig_res["frame_preds"]]
    axes[0][0].bar(range(len(orig_res["frame_preds"])), [1]*len(orig_res["frame_preds"]),
                   color=colors_o, width=1.0, edgecolor='none')
    axes[0][0].set_xlim(0, orig_res["n_frames"])
    axes[0][0].set_yticks([])
    axes[0][0].set_title(f"Original  →  {orig_label}", color=orig_color, fontweight='bold')
    axes[0][0].set_xlabel("Frame index")

    # Top right — attacked frame bar
    colors_a = ['red' if l == 0 else 'green' for l in att_res["frame_preds"]]
    axes[0][1].bar(range(len(att_res["frame_preds"])), [1]*len(att_res["frame_preds"]),
                   color=colors_a, width=1.0, edgecolor='none')
    axes[0][1].set_xlim(0, att_res["n_frames"])
    axes[0][1].set_yticks([])
    axes[0][1].set_title(f"After {attack_name} attack  →  {att_label}", color=att_color, fontweight='bold')
    axes[0][1].set_xlabel("Frame index")

    # Bottom left — original time map
    times = np.linspace(0, duration, orig_res["n_frames"])
    for i, l in enumerate(orig_res["frame_preds"]):
        axes[1][0].axvspan(times[i], times[min(i+1, orig_res["n_frames"]-1)],
                           alpha=0.35, color='red' if l == 0 else 'green')
    axes[1][0].set_xlim(0, duration)
    axes[1][0].set_yticks([])
    axes[1][0].set_xlabel("Time (seconds)")
    axes[1][0].set_title(f"BER={orig_res['ber']:.4f}  NC={orig_res['nc']:.4f}  "
                         f"Tampered frames={orig_res['n_tampered']}/{orig_res['n_frames']}")

    # Bottom right — attacked time map
    times2 = np.linspace(0, duration, att_res["n_frames"])
    for i, l in enumerate(att_res["frame_preds"]):
        axes[1][1].axvspan(times2[i], times2[min(i+1, att_res["n_frames"]-1)],
                           alpha=0.35, color='red' if l == 0 else 'green')
    axes[1][1].set_xlim(0, duration)
    axes[1][1].set_yticks([])
    axes[1][1].set_xlabel("Time (seconds)")
    loc_text = f"Tampered: {att_res['s_time']:.2f}s to {att_res['e_time']:.2f}s" if att_res["tampered"] else "No tamper detected"
    axes[1][1].set_title(f"BER={att_res['ber']:.4f}  NC={att_res['nc']:.4f}  {loc_text}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ── MAIN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    model_path = os.path.join(OUTPUT_PATH, "svm_model.pkl")
    if not os.path.exists(model_path):
        print("ERROR: svm_model.pkl not found. Run step2_train.py first.")
        exit()

    with open(model_path, "rb") as f:
        mdl = pickle.load(f)
    clf    = mdl["model"]
    scaler = mdl["scaler"]

    attacks = ["noise", "deletion", "splice", "amplify"]
    summary = []

    for audio_path in TEST_FILES:
        if not os.path.exists(audio_path):
            print(f"Skipping (not found): {audio_path}")
            continue

        fname = os.path.basename(audio_path)
        print(f"\n{'='*60}")
        print(f"  File: {fname}")
        print(f"{'='*60}")

        y, _     = librosa.load(audio_path, sr=SR, mono=True)
        duration = len(y) / SR
        reg      = register(audio_path)

        print(f"  Duration : {duration:.2f}s   Frames : {reg['n_frames']}")

        orig_res = verify(y, reg, clf, scaler)
        orig_label = "AUTHENTIC" if not orig_res["tampered"] else "TAMPERED"
        print(f"\n  Original → {orig_label}  (BER={orig_res['ber']:.4f}  NC={orig_res['nc']:.4f})")

        for attack in attacks:
            y_att, s, e = apply_attack(y, attack)
            att_res     = verify(y_att, reg, clf, scaler)
            att_label   = "TAMPERED" if att_res["tampered"] else "AUTHENTIC"

            loc = f"{att_res['s_time']:.2f}s–{att_res['e_time']:.2f}s" if att_res["tampered"] else "—"
            print(f"  {attack:<10} → {att_label:<10}  BER={att_res['ber']:.4f}  NC={att_res['nc']:.4f}  Location={loc}")

            summary.append({
                "file": fname, "attack": attack,
                "orig": orig_label, "result": att_label,
                "ber": att_res["ber"], "nc": att_res["nc"],
                "location": loc,
            })

            save_name = f"{os.path.splitext(fname)[0]}_{attack}.png"
            save_path = os.path.join(OUTPUT_PATH, save_name)
            draw_comparison(fname, duration, orig_res, att_res, attack, save_path)
            print(f"  Heatmap  → {save_name}")

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  COMPLETE RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  {'File':<16} {'Attack':<12} {'Original':<12} {'After Attack':<14} {'BER':<8} {'NC':<8} {'Location'}")
    print(f"  {'-'*76}")
    for r in summary:
        print(f"  {r['file']:<16} {r['attack']:<12} {r['orig']:<12} {r['result']:<14} "
              f"{r['ber']:<8.4f} {r['nc']:<8.4f} {r['location']}")
    print(f"{'='*80}")
    print(f"\nAll heatmaps saved in: {OUTPUT_PATH}")