# import os
# import re
# import json
# import uuid
# import numpy as np
# import librosa
# import pickle
# import soundfile as sf
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from scipy.ndimage import uniform_filter1d
# from flask import Flask, render_template, request, jsonify, send_from_directory
# import warnings
# warnings.filterwarnings('ignore')

# app = Flask(__name__)

# OUTPUT_PATH   = r"A:\audio_watermark\output"
# MODEL_PATH    = os.path.join(OUTPUT_PATH, "svm_model.pkl")
# HISTORY_FILE  = os.path.join(OUTPUT_PATH, "history.json")
# UPLOAD_FOLDER = os.path.join(OUTPUT_PATH, "uploads")

# os.makedirs(OUTPUT_PATH,   exist_ok=True)
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# SR            = 16000
# FRAME_LEN     = int(0.025 * SR)
# HOP_LEN       = int(0.010 * SR)
# N_MFCC        = 13
# BER_THRESHOLD = 0.35
# NC_THRESHOLD  = 0.60

# # known attack suffixes to strip when auto-matching
# ATTACK_SUFFIXES = ['_noise', '_deletion', '_splice', '_amplify']

# clf, scaler = None, None
# if os.path.exists(MODEL_PATH):
#     with open(MODEL_PATH, "rb") as f:
#         mdl    = pickle.load(f)
#     clf    = mdl["model"]
#     scaler = mdl["scaler"]

# # ── auto-match: given a filename, find its registered pkl ──────────────
# def find_matching_pkl(filename):
#     name = os.path.splitext(filename)[0]
#     # strip known attack suffixes
#     for suffix in ATTACK_SUFFIXES:
#         if name.lower().endswith(suffix):
#             name = name[:len(name) - len(suffix)]
#             break
#     # also strip trailing _attacked or any trailing _word pattern
#     # try exact match first
#     exact = os.path.join(OUTPUT_PATH, f"registered_{name}.pkl")
#     if os.path.exists(exact):
#         return exact, f"registered_{name}.pkl"
#     # try case-insensitive match
#     for f in os.listdir(OUTPUT_PATH):
#         if f.lower() == f"registered_{name.lower()}.pkl":
#             return os.path.join(OUTPUT_PATH, f), f
#     return None, None

# # ── core functions ─────────────────────────────────────────────────────
# def henon_key(length, x0=0.1, y0=0.3, a=1.4, b=0.3):
#     x, y = x0, y0
#     bits = []
#     for _ in range(length):
#         x, y = 1 - a * x * x + y, b * x
#         bits.append(1 if x > 0 else 0)
#     return np.array(bits, dtype=np.uint8)

# def extract_features(y):
#     mfcc    = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=FRAME_LEN, hop_length=HOP_LEN)
#     delta   = librosa.feature.delta(mfcc)
#     zcr     = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LEN, hop_length=HOP_LEN)
#     pitch   = librosa.yin(y, fmin=50, fmax=400, sr=SR, frame_length=FRAME_LEN, hop_length=HOP_LEN).reshape(1, -1)
#     rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=FRAME_LEN, hop_length=HOP_LEN)
#     rms     = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN)
#     min_len = min(mfcc.shape[1], delta.shape[1], zcr.shape[1],
#                   pitch.shape[1], rolloff.shape[1], rms.shape[1])
#     return np.vstack([
#         mfcc[:, :min_len], delta[:, :min_len], zcr[:, :min_len],
#         pitch[:, :min_len], rolloff[:, :min_len], rms[:, :min_len]
#     ]).T

# def compute_ber(b1, b2):
#     n = min(len(b1), len(b2))
#     return float(np.sum(b1[:n] != b2[:n]) / n)

# def compute_nc(b1, b2):
#     n = min(len(b1), len(b2))
#     a, b = b1[:n].astype(float), b2[:n].astype(float)
#     den  = np.sqrt(np.sum(a**2) * np.sum(b**2))
#     return float(np.sum(a * b) / den) if den > 0 else 0.0

# def do_register(audio_path):
#     fname    = os.path.splitext(os.path.basename(audio_path))[0]
#     pkl_path = os.path.join(OUTPUT_PATH, f"registered_{fname}.pkl")
#     y, _     = librosa.load(audio_path, sr=SR, mono=True)
#     feats    = extract_features(y)
#     thresh   = np.mean(feats, axis=0)
#     binary   = (feats >= thresh).astype(np.uint8).flatten()
#     key      = henon_key(len(binary))
#     enc      = np.bitwise_xor(binary, key)
#     data = {
#         "audio_path": audio_path,
#         "encrypted" : enc,
#         "thresholds": thresh,
#         "n_frames"  : len(feats),
#         "n_features": feats.shape[1],
#         "henon_x0"  : 0.1, "henon_y0": 0.3,
#         "henon_a"   : 1.4, "henon_b" : 0.3,
#     }
#     with open(pkl_path, "wb") as f:
#         pickle.dump(data, f)
#     return {
#         "pkl_path"    : pkl_path,
#         "pkl_name"    : os.path.basename(pkl_path),
#         "fname"       : os.path.basename(audio_path),
#         "duration"    : round(len(y)/SR, 2),
#         "n_frames"    : len(feats),
#         "n_features"  : int(feats.shape[1]),
#         "n_bits"      : len(binary),
#         "key_preview" : list(map(int, key[:20])),
#         "raw_preview" : list(map(int, binary[:20])),
#         "enc_preview" : list(map(int, enc[:20])),
#         "henon_params": "a=1.4  b=0.3  x\u2080=0.1  y\u2080=0.3",
#     }

# def do_verify(audio_path, pkl_path):
#     with open(pkl_path, "rb") as f:
#         reg = pickle.load(f)
#     y, _         = librosa.load(audio_path, sr=SR, mono=True)
#     duration     = len(y) / SR
#     feats        = extract_features(y)
#     n_frames     = len(feats)
#     frame_preds  = clf.predict(scaler.transform(feats))
#     n_tampered   = int(np.sum(frame_preds == 0))
#     tamper_ratio = n_tampered / n_frames
#     binary_q     = (feats >= reg["thresholds"]).astype(np.uint8).flatten()
#     key          = henon_key(len(binary_q), reg["henon_x0"], reg["henon_y0"],
#                              reg["henon_a"], reg["henon_b"])
#     dec_reg      = np.bitwise_xor(reg["encrypted"][:len(binary_q)], key[:len(binary_q)])
#     ber          = compute_ber(binary_q, dec_reg)
#     nc           = compute_nc(binary_q, dec_reg)
#     svm_t        = tamper_ratio > 0.05
#     fp_t         = (ber > BER_THRESHOLD) or (nc < NC_THRESHOLD)
#     tampered     = svm_t or fp_t
#     tampered_idx = np.where(frame_preds == 0)[0]
#     s_time       = float(tampered_idx[0]  * HOP_LEN / SR) if len(tampered_idx) > 0 else 0.0
#     e_time       = float(tampered_idx[-1] * HOP_LEN / SR) if len(tampered_idx) > 0 else 0.0

#     heatmap_name = f"heatmap_{uuid.uuid4().hex[:8]}.png"
#     heatmap_path = os.path.join(OUTPUT_PATH, heatmap_name)
#     draw_heatmap(frame_preds, duration, ber, nc, n_tampered, n_frames,
#                  s_time, e_time, tampered, heatmap_path)
#     return {
#         "tampered"    : tampered,
#         "n_frames"    : n_frames,
#         "n_tampered"  : n_tampered,
#         "tamper_ratio": round(tamper_ratio * 100, 1),
#         "ber"         : round(ber, 4),
#         "nc"          : round(nc, 4),
#         "s_time"      : round(s_time, 2),
#         "e_time"      : round(e_time, 2),
#         "duration"    : round(duration, 2),
#         "heatmap"     : heatmap_name,
#         "svm_decision": "TAMPERED" if svm_t else "AUTHENTIC",
#         "fp_decision" : "TAMPERED" if fp_t  else "AUTHENTIC",
#     }

# def draw_heatmap(frame_preds, duration, ber, nc, n_tampered, n_frames,
#                  s_time, e_time, tampered, save_path):
#     fig = plt.figure(figsize=(14, 8))
#     fig.patch.set_facecolor('#FAFAFA')
#     t  = np.arange(len(frame_preds)) * HOP_LEN / SR
#     gs = fig.add_gridspec(3, 1, hspace=0.5, left=0.07, right=0.97, top=0.90, bottom=0.08)

#     verdict = "TAMPERED" if tampered else "AUTHENTIC"
#     vc      = "#D32F2F" if tampered else "#388E3C"
#     fig.suptitle(f"Tamper Localization Result  —  {verdict}",
#                  fontsize=13, fontweight='bold', color=vc, y=0.96)

#     ax0 = fig.add_subplot(gs[0])
#     colors = ['#D32F2F' if p == 0 else '#388E3C' for p in frame_preds]
#     ax0.bar(range(len(frame_preds)), [1]*len(frame_preds), color=colors, width=1.0, edgecolor='none')
#     ax0.set_xlim(0, len(frame_preds)); ax0.set_yticks([])
#     ax0.set_xlabel("Frame index", fontsize=9, color='#555')
#     ax0.set_title(f"Frame-level prediction  |  Tampered: {n_tampered}/{n_frames}  |  BER: {ber:.4f}  |  NC: {nc:.4f}",
#                   fontsize=9, color='#444', pad=5)
#     ax0.set_facecolor('#F5F5F5')

#     ax1 = fig.add_subplot(gs[1])
#     ax1.set_facecolor('#F5F5F5')
#     in_tamp, seg_s = False, 0.0
#     for i, p in enumerate(frame_preds):
#         tc = t[i] if i < len(t) else t[-1]
#         if p == 0 and not in_tamp:
#             seg_s, in_tamp = tc, True
#         elif p == 1 and in_tamp:
#             ax1.axvspan(seg_s, tc, alpha=0.3, color='#D32F2F', lw=0); in_tamp = False
#     if in_tamp:
#         ax1.axvspan(seg_s, t[-1], alpha=0.3, color='#D32F2F', lw=0)
#     energy = np.array([1 if p == 1 else 0.3 for p in frame_preds], dtype=float)
#     wave   = (energy + np.random.default_rng(42).uniform(-0.12, 0.12, len(energy))) * 0.4
#     ax1.fill_between(t[:len(wave)], wave, -wave, color='#1565C0', alpha=0.5, linewidth=0)
#     ax1.axhline(0, color='#aaa', linewidth=0.5)
#     ax1.set_xlim(0, duration); ax1.set_ylim(-0.7, 0.7); ax1.set_yticks([])
#     ax1.set_xlabel("Time (seconds)", fontsize=9, color='#555')
#     ax1.set_title("Waveform with tampered region highlighted in red", fontsize=9, color='#444', pad=5)

#     ax2 = fig.add_subplot(gs[2])
#     ax2.set_facecolor('#F5F5F5')
#     prob   = np.array([1.0 if p == 0 else 0.0 for p in frame_preds])
#     prob_s = uniform_filter1d(prob, size=8)
#     ax2.plot(t[:len(prob_s)], prob_s, color='#D32F2F', linewidth=1.5, label='Tamper probability')
#     ax2.axhline(0.5, color='#888', linewidth=0.8, linestyle='--', label='Decision boundary (0.5)')
#     ax2.fill_between(t[:len(prob_s)], prob_s, 0, where=(prob_s > 0.5), color='#D32F2F', alpha=0.15)
#     if tampered and n_tampered > 0:
#         ax2.axvline(s_time, color='#E65100', linewidth=1.2, linestyle=':', label=f'Start: {s_time:.2f}s')
#         ax2.axvline(e_time, color='#BF360C', linewidth=1.2, linestyle=':', label=f'End: {e_time:.2f}s')
#     ax2.set_xlim(0, duration); ax2.set_ylim(-0.05, 1.15)
#     ax2.set_xlabel("Time (seconds)", fontsize=9, color='#555')
#     ax2.set_ylabel("Tamper probability", fontsize=9, color='#555')
#     ax2.set_title("Frame-level tamper probability timeline", fontsize=9, color='#444', pad=5)
#     ax2.legend(fontsize=8, loc='upper right', framealpha=0.9, edgecolor='#ccc', fancybox=False)
#     for ax in [ax0, ax1, ax2]:
#         for sp in ax.spines.values():
#             sp.set_linewidth(0.5); sp.set_color('#ccc')

#     plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor='#FAFAFA')
#     plt.close()

# def apply_attack(y, attack):
#     y = y.copy(); n = len(y)
#     if attack == "noise":
#         s, e   = int(n*0.25), int(n*0.65)
#         sp     = np.mean(y[s:e]**2) + 1e-10
#         np_    = sp / (10**(10/10))
#         y[s:e] = np.clip(y[s:e] + np.random.normal(0, np.sqrt(np_), e-s), -1.0, 1.0)
#         return y, s, e
#     elif attack == "deletion":
#         s, e = int(n*0.30), int(n*0.60); y[s:e] = 0.0; return y, s, e
#     elif attack == "splice":
#         s, e   = int(n*0.33), int(n*0.66)
#         y[s:e] = np.random.uniform(-0.3, 0.3, e-s).astype(np.float32); return y, s, e
#     elif attack == "amplify":
#         s, e   = int(n*0.20), int(n*0.55)
#         y[s:e] = np.clip(y[s:e]*4.0, -1.0, 1.0); return y, s, e

# def load_history():
#     if os.path.exists(HISTORY_FILE):
#         with open(HISTORY_FILE) as f:
#             return json.load(f)
#     return []

# def save_history(entry):
#     h = load_history(); h.insert(0, entry)
#     with open(HISTORY_FILE, "w") as f:
#         json.dump(h[:50], f)

# # ── routes ─────────────────────────────────────────────────────────────
# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/register")
# def register_page():
#     return render_template("register.html")

# @app.route("/verify")
# def verify_page():
#     return render_template("verify.html")

# @app.route("/attacks")
# def attacks_page():
#     return render_template("attacks.html")

# @app.route("/history")
# def history_page():
#     return render_template("history.html", history=load_history())

# @app.route("/api/register", methods=["POST"])
# def api_register():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     f    = request.files["file"]
#     path = os.path.join(UPLOAD_FOLDER, f.filename)
#     f.save(path)
#     try:
#         return jsonify(do_register(path))
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/api/match_pkl", methods=["POST"])
# def api_match_pkl():
#     filename = request.json.get("filename", "")
#     pkl_path, pkl_name = find_matching_pkl(filename)
#     if pkl_name:
#         return jsonify({"found": True, "pkl_name": pkl_name})
#     else:
#         name = os.path.splitext(filename)[0]
#         for suffix in ATTACK_SUFFIXES:
#             if name.lower().endswith(suffix):
#                 name = name[:len(name)-len(suffix)]
#                 break
#         return jsonify({"found": False, "base_name": name})

# @app.route("/api/verify", methods=["POST"])
# def api_verify():
#     if clf is None:
#         return jsonify({"error": "Model not loaded. Run step2_train.py first."}), 500
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     f        = request.files["file"]
#     path     = os.path.join(UPLOAD_FOLDER, f.filename)
#     f.save(path)
#     pkl_path, pkl_name = find_matching_pkl(f.filename)
#     if not pkl_path:
#         name = os.path.splitext(f.filename)[0]
#         for suffix in ATTACK_SUFFIXES:
#             if name.lower().endswith(suffix):
#                 name = name[:len(name)-len(suffix)]
#                 break
#         return jsonify({
#             "error"    : f"No registered fingerprint found for '{name}'.",
#             "no_match" : True,
#             "base_name": name,
#         }), 404
#     try:
#         result = do_verify(path, pkl_path)
#         result["matched_pkl"] = pkl_name
#         save_history({
#             "file"    : f.filename,
#             "pkl"     : pkl_name,
#             "result"  : "TAMPERED" if result["tampered"] else "AUTHENTIC",
#             "ber"     : result["ber"],
#             "nc"      : result["nc"],
#             "location": f"{result['s_time']}s \u2013 {result['e_time']}s" if result["tampered"] else "\u2014",
#             "heatmap" : result["heatmap"],
#         })
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/api/attack", methods=["POST"])
# def api_attack():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     f      = request.files["file"]
#     attack = request.form.get("attack", "noise")
#     path   = os.path.join(UPLOAD_FOLDER, f.filename)
#     f.save(path)
#     try:
#         y, _        = librosa.load(path, sr=SR, mono=True)
#         y_att, s, e = apply_attack(y, attack)
#         out_name    = f"{os.path.splitext(f.filename)[0]}_{attack}.wav"
#         sf.write(os.path.join(OUTPUT_PATH, out_name), y_att, SR)
#         return jsonify({
#             "filename": out_name,
#             "s_time"  : round(s/SR, 2),
#             "e_time"  : round(e/SR, 2),
#             "duration": round(len(y)/SR, 2),
#             "attack"  : attack,
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/output/<path:filename>")
# def serve_output(filename):
#     return send_from_directory(OUTPUT_PATH, filename)

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)
import os
import re
import json
import uuid
import numpy as np
import librosa
import pickle
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from flask import Flask, render_template, request, jsonify, send_from_directory
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

OUTPUT_PATH   = r"A:\audio_watermark\output"
MODEL_PATH    = os.path.join(OUTPUT_PATH, "svm_model.pkl")
HISTORY_FILE  = os.path.join(OUTPUT_PATH, "history.json")
UPLOAD_FOLDER = os.path.join(OUTPUT_PATH, "uploads")

os.makedirs(OUTPUT_PATH,   exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SR            = 16000
FRAME_LEN     = int(0.025 * SR)
HOP_LEN       = int(0.010 * SR)
N_MFCC        = 13

# ── Tightened thresholds so BER+NC correctly catches all tampered files ──
BER_THRESHOLD = 0.05   # any BER above 0.05 = tampered
NC_THRESHOLD  = 0.97   # any NC below 0.97 = tampered

ATTACK_SUFFIXES = ['_noise', '_deletion', '_splice', '_amplify']

clf, scaler = None, None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        mdl    = pickle.load(f)
    clf    = mdl["model"]
    scaler = mdl["scaler"]

def find_matching_pkl(filename):
    name = os.path.splitext(filename)[0]
    for suffix in ATTACK_SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[:len(name) - len(suffix)]
            break
    exact = os.path.join(OUTPUT_PATH, f"registered_{name}.pkl")
    if os.path.exists(exact):
        return exact, f"registered_{name}.pkl"
    for f in os.listdir(OUTPUT_PATH):
        if f.lower() == f"registered_{name.lower()}.pkl":
            return os.path.join(OUTPUT_PATH, f), f
    return None, None

def henon_key(length, x0=0.1, y0=0.3, a=1.4, b=0.3):
    x, y = x0, y0
    bits = []
    for _ in range(length):
        x, y = 1 - a * x * x + y, b * x
        bits.append(1 if x > 0 else 0)
    return np.array(bits, dtype=np.uint8)

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

def compute_ber(b1, b2):
    n = min(len(b1), len(b2))
    return float(np.sum(b1[:n] != b2[:n]) / n)

def compute_nc(b1, b2):
    n = min(len(b1), len(b2))
    a, b = b1[:n].astype(float), b2[:n].astype(float)
    den  = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(np.sum(a * b) / den) if den > 0 else 0.0

def do_register(audio_path):
    fname    = os.path.splitext(os.path.basename(audio_path))[0]
    pkl_path = os.path.join(OUTPUT_PATH, f"registered_{fname}.pkl")
    y, _     = librosa.load(audio_path, sr=SR, mono=True)
    feats    = extract_features(y)
    thresh   = np.mean(feats, axis=0)
    binary   = (feats >= thresh).astype(np.uint8).flatten()
    key      = henon_key(len(binary))
    enc      = np.bitwise_xor(binary, key)
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
    return {
        "pkl_name"    : os.path.basename(pkl_path),
        "fname"       : os.path.basename(audio_path),
        "duration"    : round(len(y)/SR, 2),
        "n_frames"    : len(feats),
        "n_features"  : int(feats.shape[1]),
        "n_bits"      : len(binary),
        "key_preview" : list(map(int, key[:20])),
        "raw_preview" : list(map(int, binary[:20])),
        "enc_preview" : list(map(int, enc[:20])),
        "henon_params": "a=1.4  b=0.3  x\u2080=0.1  y\u2080=0.3",
    }

def do_verify(audio_path, pkl_path):
    with open(pkl_path, "rb") as f:
        reg = pickle.load(f)
    y, _         = librosa.load(audio_path, sr=SR, mono=True)
    duration     = len(y) / SR
    feats        = extract_features(y)
    n_frames     = len(feats)

    # ── Layer 1: Random Forest — tamper LOCALIZATION ──────────────────
    frame_preds  = clf.predict(scaler.transform(feats))
    n_tampered   = int(np.sum(frame_preds == 0))
    tamper_ratio = n_tampered / n_frames
    svm_tampered = tamper_ratio > 0.05

    # find tamper region
    tampered_idx = np.where(frame_preds == 0)[0]
    s_time = float(tampered_idx[0]  * HOP_LEN / SR) if len(tampered_idx) > 0 else 0.0
    e_time = float(tampered_idx[-1] * HOP_LEN / SR) if len(tampered_idx) > 0 else 0.0

    # ── Layer 2: BER + NC — global AUTHENTICATION ─────────────────────
    binary_q  = (feats >= reg["thresholds"]).astype(np.uint8).flatten()
    key       = henon_key(len(binary_q), reg["henon_x0"], reg["henon_y0"],
                          reg["henon_a"], reg["henon_b"])
    dec_reg   = np.bitwise_xor(reg["encrypted"][:len(binary_q)], key[:len(binary_q)])
    ber       = compute_ber(binary_q, dec_reg)
    nc        = compute_nc(binary_q, dec_reg)
    fp_tampered = (ber > BER_THRESHOLD) or (nc < NC_THRESHOLD)

    # ── Final decision: tampered if EITHER layer says so ──────────────
    tampered = svm_tampered or fp_tampered

    heatmap_name = f"heatmap_{uuid.uuid4().hex[:8]}.png"
    draw_heatmap(frame_preds, duration, ber, nc, n_tampered, n_frames,
                 s_time, e_time, tampered,
                 os.path.join(OUTPUT_PATH, heatmap_name))

    return {
        "tampered"        : tampered,
        "n_frames"        : n_frames,
        "n_tampered"      : n_tampered,
        "tamper_ratio"    : round(tamper_ratio * 100, 1),
        "ber"             : round(ber, 4),
        "nc"              : round(nc, 4),
        "s_time"          : round(s_time, 2),
        "e_time"          : round(e_time, 2),
        "duration"        : round(duration, 2),
        "heatmap"         : heatmap_name,
        "rf_tampered"     : svm_tampered,
        "fp_tampered"     : fp_tampered,
        "rf_label"        : "TAMPERED" if svm_tampered else "AUTHENTIC",
        "fp_label"        : "TAMPERED" if fp_tampered  else "AUTHENTIC",
    }

def draw_heatmap(frame_preds, duration, ber, nc, n_tampered, n_frames,
                 s_time, e_time, tampered, save_path):
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#FAFAFA')
    t  = np.arange(len(frame_preds)) * HOP_LEN / SR
    gs = fig.add_gridspec(3, 1, hspace=0.5,
                          left=0.07, right=0.97, top=0.90, bottom=0.08)

    verdict = "TAMPERED" if tampered else "AUTHENTIC"
    vc      = "#D32F2F" if tampered else "#388E3C"
    fig.suptitle(f"Tamper Localization Result  —  {verdict}",
                 fontsize=13, fontweight='bold', color=vc, y=0.96)

    ax0 = fig.add_subplot(gs[0])
    colors = ['#D32F2F' if p == 0 else '#388E3C' for p in frame_preds]
    ax0.bar(range(len(frame_preds)), [1]*len(frame_preds),
            color=colors, width=1.0, edgecolor='none')
    ax0.set_xlim(0, len(frame_preds)); ax0.set_yticks([])
    ax0.set_xlabel("Frame index", fontsize=9, color='#555')
    ax0.set_title(
        f"Frame-level prediction  |  Tampered: {n_tampered}/{n_frames}"
        f"  |  BER: {ber:.4f}  |  NC: {nc:.4f}",
        fontsize=9, color='#444', pad=5)
    ax0.set_facecolor('#F5F5F5')

    ax1 = fig.add_subplot(gs[1])
    ax1.set_facecolor('#F5F5F5')
    in_tamp, seg_s = False, 0.0
    for i, p in enumerate(frame_preds):
        tc = t[i] if i < len(t) else t[-1]
        if p == 0 and not in_tamp:
            seg_s, in_tamp = tc, True
        elif p == 1 and in_tamp:
            ax1.axvspan(seg_s, tc, alpha=0.3, color='#D32F2F', lw=0)
            in_tamp = False
    if in_tamp:
        ax1.axvspan(seg_s, t[-1], alpha=0.3, color='#D32F2F', lw=0)
    energy = np.array([1 if p == 1 else 0.3 for p in frame_preds], dtype=float)
    wave   = (energy + np.random.default_rng(42).uniform(-0.12, 0.12,
              len(energy))) * 0.4
    ax1.fill_between(t[:len(wave)], wave, -wave,
                     color='#1565C0', alpha=0.5, linewidth=0)
    ax1.axhline(0, color='#aaa', linewidth=0.5)
    ax1.set_xlim(0, duration); ax1.set_ylim(-0.7, 0.7); ax1.set_yticks([])
    ax1.set_xlabel("Time (seconds)", fontsize=9, color='#555')
    ax1.set_title("Waveform — tampered region highlighted in red",
                  fontsize=9, color='#444', pad=5)

    ax2 = fig.add_subplot(gs[2])
    ax2.set_facecolor('#F5F5F5')
    prob   = np.array([1.0 if p == 0 else 0.0 for p in frame_preds])
    prob_s = uniform_filter1d(prob, size=8)
    ax2.plot(t[:len(prob_s)], prob_s, color='#D32F2F',
             linewidth=1.5, label='Tamper probability')
    ax2.axhline(0.5, color='#888', linewidth=0.8, linestyle='--',
                label='Decision boundary (0.5)')
    ax2.fill_between(t[:len(prob_s)], prob_s, 0,
                     where=(prob_s > 0.5), color='#D32F2F', alpha=0.15)
    if tampered and n_tampered > 0:
        ax2.axvline(s_time, color='#E65100', linewidth=1.2, linestyle=':',
                    label=f'Start: {s_time:.2f}s')
        ax2.axvline(e_time, color='#BF360C', linewidth=1.2, linestyle=':',
                    label=f'End: {e_time:.2f}s')
    ax2.set_xlim(0, duration); ax2.set_ylim(-0.05, 1.15)
    ax2.set_xlabel("Time (seconds)", fontsize=9, color='#555')
    ax2.set_ylabel("Tamper probability", fontsize=9, color='#555')
    ax2.set_title("Frame-level tamper probability timeline",
                  fontsize=9, color='#444', pad=5)
    ax2.legend(fontsize=8, loc='upper right',
               framealpha=0.9, edgecolor='#ccc', fancybox=False)
    for ax in [ax0, ax1, ax2]:
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color('#ccc')

    plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()

def apply_attack(y, attack):
    y = y.copy(); n = len(y)
    if attack == "noise":
        s, e   = int(n*0.25), int(n*0.65)
        sp     = np.mean(y[s:e]**2) + 1e-10
        np_    = sp / (10**(10/10))
        y[s:e] = np.clip(y[s:e] + np.random.normal(0, np.sqrt(np_), e-s), -1.0, 1.0)
        return y, s, e
    elif attack == "deletion":
        s, e = int(n*0.30), int(n*0.60); y[s:e] = 0.0; return y, s, e
    elif attack == "splice":
        s, e   = int(n*0.33), int(n*0.66)
        y[s:e] = np.random.uniform(-0.3, 0.3, e-s).astype(np.float32)
        return y, s, e
    elif attack == "amplify":
        s, e   = int(n*0.20), int(n*0.55)
        y[s:e] = np.clip(y[s:e]*4.0, -1.0, 1.0); return y, s, e

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(entry):
    h = load_history(); h.insert(0, entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(h[:50], f)

# ── routes ─────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/verify")
def verify_page():
    return render_template("verify.html")

@app.route("/attacks")
def attacks_page():
    return render_template("attacks.html")

@app.route("/history")
def history_page():
    return render_template("history.html", history=load_history())

@app.route("/api/register", methods=["POST"])
def api_register():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f    = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    try:
        return jsonify(do_register(path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/match_pkl", methods=["POST"])
def api_match_pkl():
    filename = request.json.get("filename", "")
    pkl_path, pkl_name = find_matching_pkl(filename)
    if pkl_name:
        return jsonify({"found": True, "pkl_name": pkl_name})
    else:
        name = os.path.splitext(filename)[0]
        for suffix in ATTACK_SUFFIXES:
            if name.lower().endswith(suffix):
                name = name[:len(name)-len(suffix)]; break
        return jsonify({"found": False, "base_name": name})

@app.route("/api/verify", methods=["POST"])
def api_verify():
    if clf is None:
        return jsonify({"error": "Model not loaded. Run step2_train.py first."}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f    = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    pkl_path, pkl_name = find_matching_pkl(f.filename)
    if not pkl_path:
        name = os.path.splitext(f.filename)[0]
        for suffix in ATTACK_SUFFIXES:
            if name.lower().endswith(suffix):
                name = name[:len(name)-len(suffix)]; break
        return jsonify({
            "error"    : f"No registered fingerprint found for '{name}'.",
            "no_match" : True,
            "base_name": name,
        }), 404
    try:
        result = do_verify(path, pkl_path)
        save_history({
            "file"    : f.filename,
            "pkl"     : pkl_name,
            "result"  : "TAMPERED" if result["tampered"] else "AUTHENTIC",
            "ber"     : result["ber"],
            "nc"      : result["nc"],
            "location": f"{result['s_time']}s \u2013 {result['e_time']}s"
                        if result["tampered"] else "\u2014",
            "heatmap" : result["heatmap"],
        })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/attack", methods=["POST"])
def api_attack():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f      = request.files["file"]
    attack = request.form.get("attack", "noise")
    path   = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    try:
        y, _        = librosa.load(path, sr=SR, mono=True)
        y_att, s, e = apply_attack(y, attack)
        out_name    = f"{os.path.splitext(f.filename)[0]}_{attack}.wav"
        sf.write(os.path.join(OUTPUT_PATH, out_name), y_att, SR)
        return jsonify({
            "filename": out_name,
            "s_time"  : round(s/SR, 2),
            "e_time"  : round(e/SR, 2),
            "duration": round(len(y)/SR, 2),
            "attack"  : attack,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_PATH, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)