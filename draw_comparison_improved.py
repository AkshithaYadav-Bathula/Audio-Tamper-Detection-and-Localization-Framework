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
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import librosa

SR      = 16000
HOP_LEN = int(0.010 * SR)



# ======================================================================
# AUDIO FILES TO TEST
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

# ── Register and SAVE to pkl ───────────────────────────────────────────
def register_and_save(audio_path):
    fname     = os.path.splitext(os.path.basename(audio_path))[0]
    pkl_path  = os.path.join(OUTPUT_PATH, f"registered_{fname}.pkl")

    y, _      = librosa.load(audio_path, sr=SR, mono=True)
    feats     = extract_features(y)
    thresh    = np.mean(feats, axis=0)
    binary    = (feats >= thresh).astype(np.uint8).flatten()
    key       = henon_key(len(binary))
    enc       = np.bitwise_xor(binary, key)

    data = {
        "audio_path": audio_path,
        "encrypted" : enc,
        "thresholds": thresh,
        "n_frames"  : len(feats),
        "n_features": feats.shape[1],
        "henon_x0"  : 0.1, "henon_y0": 0.3,
        "henon_a"   : 1.4, "henon_b" : 0.3,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Registered and saved → {os.path.basename(pkl_path)}")
    print(f"  Frames: {len(feats)}   Features: {feats.shape[1]}   Bits: {len(binary)}")
    return data, pkl_path

# ── Load from pkl ──────────────────────────────────────────────────────
def load_registration(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

# ── Apply attacks ──────────────────────────────────────────────────────
def apply_attack(y, attack):
    y = y.copy()
    n = len(y)
    if attack == "noise":
        s, e      = int(n * 0.25), int(n * 0.65)
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
        y[s:e] = np.random.uniform(-0.3, 0.3, e - s).astype(np.float32)
        return y, s, e
    elif attack == "amplify":
        s, e   = int(n * 0.20), int(n * 0.55)
        y[s:e] = np.clip(y[s:e] * 4.0, -1.0, 1.0)
        return y, s, e

# ── Verify against loaded pkl ──────────────────────────────────────────
def verify(y, reg, clf, scaler):
    feats        = extract_features(y)
    n_frames     = len(feats)
    frame_preds  = clf.predict(scaler.transform(feats))
    n_tampered   = int(np.sum(frame_preds == 0))
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
    s_time = tampered_idx[0]  * HOP_LEN / SR if len(tampered_idx) > 0 else 0.0
    e_time = tampered_idx[-1] * HOP_LEN / SR if len(tampered_idx) > 0 else 0.0

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



def draw_comparison(fname, duration, orig_res, att_res, attack_name, save_path):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#FAFAFA')

    attack_display = {
        "noise"    : "Gaussian Noise Injection",
        "deletion" : "Segment Deletion",
        "splice"   : "Audio Splicing",
        "amplify"  : "Amplitude Modification",
    }.get(attack_name, attack_name.upper())

    fig.suptitle(
        f"Tamper Localization — {fname}   |   Attack: {attack_display}",
        fontsize=13, fontweight='bold', y=0.98, color='#1a1a1a'
    )

    n_o   = orig_res["n_frames"]
    n_a   = att_res["n_frames"]
    t_o   = np.arange(n_o) * HOP_LEN / SR
    t_a   = np.arange(n_a) * HOP_LEN / SR

    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.35,
                          left=0.07, right=0.97, top=0.91, bottom=0.06)

    # ── Row 0: frame-level prediction bars ──────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    for ax, res, title_str in [
        (ax0, orig_res, "Original audio — frame-level prediction"),
        (ax1, att_res,  f"After {attack_display.lower()} — frame-level prediction"),
    ]:
        preds  = res["frame_preds"]
        colors = ['#D32F2F' if p == 0 else '#388E3C' for p in preds]
        ax.bar(range(len(preds)), [1]*len(preds), color=colors,
               width=1.0, edgecolor='none')
        ax.set_xlim(0, len(preds))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Frame index", fontsize=9, color='#555')
        verdict       = "TAMPERED" if res["tampered"] else "AUTHENTIC"
        verdict_color = '#D32F2F' if res["tampered"] else '#388E3C'
        ax.set_title(title_str, fontsize=9, color='#333', pad=4)
        ax.text(0.99, 0.88, verdict, transform=ax.transAxes,
                fontsize=10, fontweight='bold', color=verdict_color,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=verdict_color, lw=1.2))
        ax.set_facecolor('#F5F5F5')
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#ccc')

    # ── Row 1: waveform + shaded tamper overlay ──────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    for ax, res, t_arr, title_str in [
        (ax2, orig_res, t_o, "Waveform — authentic region overlay"),
        (ax3, att_res,  t_a, "Waveform — detected tamper region"),
    ]:
        preds = res["frame_preds"]
        ax.set_facecolor('#F5F5F5')

        # shade spans
        in_tamp = False
        seg_start = 0.0
        for i, p in enumerate(preds):
            t_cur = t_arr[i] if i < len(t_arr) else t_arr[-1]
            t_nxt = t_arr[min(i+1, len(t_arr)-1)]
            if p == 0 and not in_tamp:
                seg_start = t_cur
                in_tamp = True
            elif p == 1 and in_tamp:
                ax.axvspan(seg_start, t_cur, alpha=0.30, color='#D32F2F', lw=0)
                in_tamp = False
        if in_tamp:
            ax.axvspan(seg_start, t_arr[-1], alpha=0.30, color='#D32F2F', lw=0)

        # draw a synthetic waveform proxy using frame energy (RMS-like)
        energy = np.array([1 if p == 1 else 0.3 for p in preds], dtype=float)
        noise  = np.random.default_rng(42).uniform(-0.15, 0.15, len(energy))
        wave   = (energy + noise) * 0.45
        ax.fill_between(t_arr[:len(wave)],  wave, -wave,
                        color='#1565C0', alpha=0.55, linewidth=0)
        ax.axhline(0, color='#aaa', linewidth=0.5)
        ax.set_xlim(0, duration)
        ax.set_ylim(-0.75, 0.75)
        ax.set_yticks([])
        ax.set_xlabel("Time (seconds)", fontsize=9, color='#555')
        ax.set_title(title_str, fontsize=9, color='#333', pad=4)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#ccc')

        # BER / NC annotation
        ax.text(0.01, 0.93,
                f"BER = {res['ber']:.4f}   NC = {res['nc']:.4f}   "
                f"Tampered frames: {res['n_tampered']}/{res['n_frames']}",
                transform=ax.transAxes, fontsize=8, color='#444',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#ccc', lw=0.8))

    # ── Row 2: continuous tamper probability timeline ─────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor('#F5F5F5')

    # smooth tamper probability: 1 = tampered, 0 = authentic
    from scipy.ndimage import uniform_filter1d
    prob_o = np.array([1.0 if p == 0 else 0.0 for p in orig_res["frame_preds"]])
    prob_a = np.array([1.0 if p == 0 else 0.0 for p in att_res["frame_preds"]])
    prob_o_s = uniform_filter1d(prob_o, size=8)
    prob_a_s = uniform_filter1d(prob_a, size=8)

    ax4.plot(t_o[:len(prob_o_s)], prob_o_s,
             color='#388E3C', linewidth=1.4, label='Original', alpha=0.85)
    ax4.plot(t_a[:len(prob_a_s)], prob_a_s,
             color='#D32F2F', linewidth=1.4, label=f'After attack', alpha=0.85)
    ax4.axhline(0.5, color='#888', linewidth=0.8, linestyle='--', label='Decision boundary (0.5)')
    ax4.fill_between(t_a[:len(prob_a_s)], prob_a_s, 0,
                     where=(prob_a_s > 0.5), color='#D32F2F', alpha=0.15)

    if att_res["tampered"] and att_res["n_tampered"] > 0:
        ax4.axvline(att_res["s_time"], color='#E65100', linewidth=1.2,
                    linestyle=':', label=f'Detected start ({att_res["s_time"]:.2f}s)')
        ax4.axvline(att_res["e_time"], color='#BF360C', linewidth=1.2,
                    linestyle=':', label=f'Detected end ({att_res["e_time"]:.2f}s)')

    ax4.set_xlim(0, duration)
    ax4.set_ylim(-0.05, 1.15)
    ax4.set_xlabel("Time (seconds)", fontsize=9, color='#555')
    ax4.set_ylabel("Tamper probability", fontsize=9, color='#555')
    ax4.set_title("Frame-level tamper probability timeline — original vs attacked audio",
                  fontsize=9, color='#333', pad=4)
    ax4.legend(fontsize=8, loc='upper right', framealpha=0.9,
               edgecolor='#ccc', fancybox=False)
    for spine in ax4.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#ccc')

    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"  Heatmap saved → {save_path}")



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
            print(f"\nSkipping (not found): {audio_path}")
            continue

        fname = os.path.basename(audio_path)
        fname_no_ext = os.path.splitext(fname)[0]

        print(f"\n{'='*60}")
        print(f"  FILE: {fname}")
        print(f"{'='*60}")

        # STEP 1: Register and save pkl
        print(f"\n  [REGISTRATION]")
        reg, pkl_path = register_and_save(audio_path)

        # STEP 2: Load from pkl (proves it was saved and loaded correctly)
        reg_loaded = load_registration(pkl_path)
        print(f"  Loaded from pkl  → {os.path.basename(pkl_path)} ✓")

        # STEP 3: Load audio
        y, _     = librosa.load(audio_path, sr=SR, mono=True)
        duration = len(y) / SR

        # STEP 4: Verify original
        print(f"\n  [VERIFICATION]")
        orig_res   = verify(y, reg_loaded, clf, scaler)
        orig_label = "AUTHENTIC" if not orig_res["tampered"] else "TAMPERED"
        print(f"  Original  → {orig_label:<10}  "
              f"BER={orig_res['ber']:.4f}  NC={orig_res['nc']:.4f}  "
              f"Tampered frames={orig_res['n_tampered']}/{orig_res['n_frames']}")

        # STEP 5: Apply each attack, verify, save heatmap
        for attack in attacks:
            y_att, s, e = apply_attack(y, attack)

            # save attacked wav
            att_wav = os.path.join(OUTPUT_PATH, f"{fname_no_ext}_{attack}.wav")
            sf.write(att_wav, y_att, SR)

            att_res   = verify(y_att, reg_loaded, clf, scaler)
            att_label = "TAMPERED" if att_res["tampered"] else "AUTHENTIC"
            loc       = (f"{att_res['s_time']:.2f}s–{att_res['e_time']:.2f}s"
                         if att_res["tampered"] else "—")

            print(f"  {attack:<10} → {att_label:<10}  "
                  f"BER={att_res['ber']:.4f}  NC={att_res['nc']:.4f}  "
                  f"Location={loc}")

            summary.append({
                "file": fname, "attack": attack,
                "orig": orig_label, "result": att_label,
                "ber": att_res["ber"], "nc": att_res["nc"],
                "location": loc,
            })

            heatmap_path = os.path.join(OUTPUT_PATH, f"{fname_no_ext}_{attack}.png")
            draw_comparison(fname, duration, orig_res, att_res, attack, heatmap_path)
            print(f"  Heatmap saved → {fname_no_ext}_{attack}.png")

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  COMPLETE RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  {'File':<16} {'Attack':<12} {'Original':<12} {'After Attack':<14} "
          f"{'BER':<8} {'NC':<8} Location")
    print(f"  {'-'*76}")
    for r in summary:
        print(f"  {r['file']:<16} {r['attack']:<12} {r['orig']:<12} {r['result']:<14} "
              f"{r['ber']:<8.4f} {r['nc']:<8.4f} {r['location']}")
    print(f"{'='*80}")

    print(f"\nPKL files saved in output folder:")
    for audio_path in TEST_FILES:
        fname_no_ext = os.path.splitext(os.path.basename(audio_path))[0]
        pkl = os.path.join(OUTPUT_PATH, f"registered_{fname_no_ext}.pkl")
        if os.path.exists(pkl):
            print(f"  {os.path.basename(pkl)} ✓")

    print(f"\nAll heatmaps saved in: {OUTPUT_PATH}")
