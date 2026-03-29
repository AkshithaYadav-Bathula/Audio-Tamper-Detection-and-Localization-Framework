# # # import os
# # # import pickle
# # # import numpy as np
# # # from sklearn.linear_model import SGDClassifier
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# # # import warnings
# # # warnings.filterwarnings('ignore')

# # # OUTPUT_PATH = r"A:\audio_watermark\output"

# # # print("Loading dataset...")
# # # with open(os.path.join(OUTPUT_PATH, "dataset.pkl"), "rb") as f:
# # #     data = pickle.load(f)

# # # X = data["X"]
# # # y = data["y"]

# # # print(f"  Total frames : {len(X)}")
# # # print(f"  Authentic (1): {np.sum(y == 1)}")
# # # print(f"  Tampered  (0): {np.sum(y == 0)}")

# # # # --- Balance classes by undersampling authentic frames ---
# # # print("\nBalancing classes...")
# # # np.random.seed(42)
# # # idx_auth    = np.where(y == 1)[0]
# # # idx_tamp    = np.where(y == 0)[0]
# # # n_tamp      = len(idx_tamp)
# # # idx_auth_s  = np.random.choice(idx_auth, size=n_tamp, replace=False)
# # # idx_balanced = np.concatenate([idx_auth_s, idx_tamp])
# # # np.random.shuffle(idx_balanced)
# # # X_bal = X[idx_balanced]
# # # y_bal = y[idx_balanced]

# # # print(f"  Balanced authentic (1): {np.sum(y_bal == 1)}")
# # # print(f"  Balanced tampered  (0): {np.sum(y_bal == 0)}")

# # # # --- Train / test split 80/20 ---
# # # X_train, X_test, y_train, y_test = train_test_split(
# # #     X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
# # # )
# # # print(f"\nTrain size : {len(X_train)}")
# # # print(f"Test  size : {len(X_test)}")

# # # # --- Scale features ---
# # # print("\nScaling features...")
# # # scaler  = StandardScaler()
# # # X_train = scaler.fit_transform(X_train)
# # # X_test  = scaler.transform(X_test)

# # # # --- Train SGD classifier (SVM equivalent, fast on large data) ---
# # # print("\nTraining classifier...")
# # # # clf = SGDClassifier(
# # # #     loss="log_loss",
# # # #     max_iter=1000,
# # # #     tol=1e-3,
# # # #     random_state=42,
# # # #     class_weight="balanced",
# # # #     n_jobs=-1
# # # # )
# # # clf = SGDClassifier(
# # #     loss="hinge",
# # #     max_iter=1000,
# # #     tol=1e-3,
# # #     random_state=42,
# # #     class_weight="balanced",
# # #     early_stopping=True,
# # #     validation_fraction=0.1,
# # #     n_iter_no_change=5,
# # #     n_jobs=-1
# # # )
# # # clf.fit(X_train, y_train)
# # # print("Training done.")

# # # # --- Evaluate on test set ---
# # # print("\nEvaluating on test set...")
# # # y_pred = clf.predict(X_test)

# # # acc  = accuracy_score(y_test, y_pred)
# # # prec = precision_score(y_test, y_pred)
# # # rec  = recall_score(y_test, y_pred)
# # # f1   = f1_score(y_test, y_pred)
# # # cm   = confusion_matrix(y_test, y_pred)

# # # print(f"\n{'='*40}")
# # # print(f"  Accuracy  : {acc*100:.2f}%")
# # # print(f"  Precision : {prec*100:.2f}%")
# # # print(f"  Recall    : {rec*100:.2f}%")
# # # print(f"  F1 Score  : {f1*100:.2f}%")
# # # print(f"{'='*40}")
# # # print(f"\nConfusion Matrix:")
# # # print(f"  True Authentic, Predicted Authentic : {cm[1][1]}")
# # # print(f"  True Authentic, Predicted Tampered  : {cm[1][0]}")
# # # print(f"  True Tampered,  Predicted Tampered  : {cm[0][0]}")
# # # print(f"  True Tampered,  Predicted Authentic : {cm[0][1]}")

# # # # --- Save model and scaler ---
# # # model_path = os.path.join(OUTPUT_PATH, "svm_model.pkl")
# # # with open(model_path, "wb") as f:
# # #     pickle.dump({"model": clf, "scaler": scaler}, f)

# # # print(f"\nModel saved to: {model_path}")
# # # print("Step 2 complete. Run step3_register.py next.")

# # import os
# # import pickle
# # import numpy as np
# # from sklearn.linear_model import SGDClassifier
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# # import warnings
# # warnings.filterwarnings('ignore')

# # OUTPUT_PATH = r"A:\audio_watermark\output"

# # print("Loading dataset...")
# # with open(os.path.join(OUTPUT_PATH, "dataset.pkl"), "rb") as f:
# #     data = pickle.load(f)

# # X = data["X"]
# # y = data["y"]

# # print(f"  Total frames : {len(X)}")
# # print(f"  Authentic (1): {np.sum(y == 1)}")
# # print(f"  Tampered  (0): {np.sum(y == 0)}")

# # # --- Balance classes ---
# # print("\nBalancing classes...")
# # np.random.seed(42)
# # idx_auth     = np.where(y == 1)[0]
# # idx_tamp     = np.where(y == 0)[0]
# # n_tamp       = len(idx_tamp)
# # idx_auth_s   = np.random.choice(idx_auth, size=n_tamp, replace=False)
# # idx_balanced = np.concatenate([idx_auth_s, idx_tamp])
# # np.random.shuffle(idx_balanced)
# # X_bal = X[idx_balanced]
# # y_bal = y[idx_balanced]
# # print(f"  Each class : {n_tamp} frames")

# # # --- Train / test split 80/20 ---
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
# # )
# # print(f"\nTrain size : {len(X_train)}")
# # print(f"Test  size : {len(X_test)}")

# # # --- Scale features ---
# # print("\nScaling features...")
# # scaler  = StandardScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_test  = scaler.transform(X_test)

# # # --- Train SVM (hinge loss = true linear SVM) ---
# # print("\nTraining SVM (this takes 1-2 minutes)...")
# # clf = SGDClassifier(
# #     loss="hinge",
# #     alpha=0.0001,
# #     max_iter=2000,
# #     tol=1e-4,
# #     random_state=42,
# #     class_weight="balanced",
# #     learning_rate="optimal",
# #     early_stopping=True,
# #     validation_fraction=0.1,
# #     n_iter_no_change=10,
# #     n_jobs=-1
# # )
# # clf.fit(X_train, y_train)
# # print("Training done.")

# # # --- Evaluate ---
# # print("\nEvaluating...")
# # y_pred = clf.predict(X_test)

# # acc  = accuracy_score(y_test, y_pred)
# # prec = precision_score(y_test, y_pred)
# # rec  = recall_score(y_test, y_pred)
# # f1   = f1_score(y_test, y_pred)
# # cm   = confusion_matrix(y_test, y_pred)

# # print(f"\n{'='*45}")
# # print(f"  Accuracy  : {acc*100:.2f}%")
# # print(f"  Precision : {prec*100:.2f}%")
# # print(f"  Recall    : {rec*100:.2f}%")
# # print(f"  F1 Score  : {f1*100:.2f}%")
# # print(f"{'='*45}")
# # print(f"\nConfusion Matrix:")
# # print(f"  True Authentic, Predicted Authentic : {cm[1][1]}")
# # print(f"  True Authentic, Predicted Tampered  : {cm[1][0]}")
# # print(f"  True Tampered,  Predicted Tampered  : {cm[0][0]}")
# # print(f"  True Tampered,  Predicted Authentic : {cm[0][1]}")

# # # --- Save ---
# # model_path = os.path.join(OUTPUT_PATH, "svm_model.pkl")
# # with open(model_path, "wb") as f:
# #     pickle.dump({"model": clf, "scaler": scaler}, f)

# # print(f"\nModel saved to: {model_path}")
# # print("Step 2 complete. Run step3_register.py next.")
# import os
# import pickle
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import warnings
# warnings.filterwarnings('ignore')

# OUTPUT_PATH = r"A:\audio_watermark\output"

# print("Loading dataset...")
# with open(os.path.join(OUTPUT_PATH, "dataset.pkl"), "rb") as f:
#     data = pickle.load(f)

# X = data["X"]
# y = data["y"]

# print(f"  Total frames : {len(X)}")
# print(f"  Authentic (1): {np.sum(y == 1)}")
# print(f"  Tampered  (0): {np.sum(y == 0)}")

# # --- Balance classes ---
# print("\nBalancing classes...")
# np.random.seed(42)
# idx_auth     = np.where(y == 1)[0]
# idx_tamp     = np.where(y == 0)[0]
# n_tamp       = len(idx_tamp)
# idx_auth_s   = np.random.choice(idx_auth, size=n_tamp, replace=False)
# idx_balanced = np.concatenate([idx_auth_s, idx_tamp])
# np.random.shuffle(idx_balanced)
# X_bal = X[idx_balanced]
# y_bal = y[idx_balanced]
# print(f"  Each class : {n_tamp} frames")

# # --- Train / test split 80/20 ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
# )
# print(f"\nTrain size : {len(X_train)}")
# print(f"Test  size : {len(X_test)}")

# # --- Scale features ---
# print("\nScaling features...")
# scaler  = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test  = scaler.transform(X_test)

# # --- Train Random Forest ---
# print("\nTraining Random Forest (3-5 minutes)...")
# clf = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=20,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     class_weight="balanced",
#     random_state=42,
#     n_jobs=-1
# )
# clf.fit(X_train, y_train)
# print("Training done.")

# # --- Evaluate ---
# print("\nEvaluating...")
# y_pred = clf.predict(X_test)

# acc  = accuracy_score(y_test, y_pred)
# prec = precision_score(y_test, y_pred)
# rec  = recall_score(y_test, y_pred)
# f1   = f1_score(y_test, y_pred)
# cm   = confusion_matrix(y_test, y_pred)

# print(f"\n{'='*45}")
# print(f"  Accuracy  : {acc*100:.2f}%")
# print(f"  Precision : {prec*100:.2f}%")
# print(f"  Recall    : {rec*100:.2f}%")
# print(f"  F1 Score  : {f1*100:.2f}%")
# print(f"{'='*45}")
# print(f"\nConfusion Matrix:")
# print(f"  True Authentic, Predicted Authentic : {cm[1][1]}")
# print(f"  True Authentic, Predicted Tampered  : {cm[1][0]}")
# print(f"  True Tampered,  Predicted Tampered  : {cm[0][0]}")
# print(f"  True Tampered,  Predicted Authentic : {cm[0][1]}")

# # --- Feature importance (bonus insight for your paper) ---
# print(f"\nTop 5 most important features:")
# feat_names = (
#     [f"MFCC_{i+1}" for i in range(13)] +
#     [f"DMFCC_{i+1}" for i in range(13)] +
#     ["ZCR", "Pitch"]
# )
# importances = clf.feature_importances_
# top5_idx = np.argsort(importances)[::-1][:5]
# for i, idx in enumerate(top5_idx):
#     print(f"  {i+1}. {feat_names[idx]} : {importances[idx]*100:.2f}%")

# # --- Save ---
# model_path = os.path.join(OUTPUT_PATH, "svm_model.pkl")
# with open(model_path, "wb") as f:
#     pickle.dump({"model": clf, "scaler": scaler}, f)

# print(f"\nModel saved to: {model_path}")
# print("Step 2 complete. Run step3_register.py next.")
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

OUTPUT_PATH = r"A:\audio_watermark\output"

print("Loading dataset...")
with open(os.path.join(OUTPUT_PATH, "dataset.pkl"), "rb") as f:
    data = pickle.load(f)

X = data["X"]
y = data["y"]

print(f"  Total frames : {len(X)}")
print(f"  Authentic (1): {np.sum(y == 1)}")
print(f"  Tampered  (0): {np.sum(y == 0)}")

# --- Balance classes ---
print("\nBalancing classes...")
np.random.seed(42)
idx_auth     = np.where(y == 1)[0]
idx_tamp     = np.where(y == 0)[0]
n_tamp       = len(idx_tamp)
idx_auth_s   = np.random.choice(idx_auth, size=n_tamp, replace=False)
idx_balanced = np.concatenate([idx_auth_s, idx_tamp])
np.random.shuffle(idx_balanced)
X_bal = X[idx_balanced]
y_bal = y[idx_balanced]
print(f"  Each class : {n_tamp} frames")

# --- Train / test split 80/20 ---
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)
print(f"\nTrain size : {len(X_train)}")
print(f"Test  size : {len(X_test)}")

# --- Scale features ---
print("\nScaling features...")
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# --- Train Random Forest ---
print("\nTraining Random Forest (3-5 minutes)...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
print("Training done.")

# --- Evaluate ---
print("\nEvaluating...")
y_pred = clf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print(f"\n{'='*45}")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")
print(f"{'='*45}")
print(f"\nConfusion Matrix:")
print(f"  True Authentic, Predicted Authentic : {cm[1][1]}")
print(f"  True Authentic, Predicted Tampered  : {cm[1][0]}")
print(f"  True Tampered,  Predicted Tampered  : {cm[0][0]}")
print(f"  True Tampered,  Predicted Authentic : {cm[0][1]}")

# --- Feature importance ---
feat_names = (
    [f"MFCC_{i+1}"  for i in range(13)] +
    [f"DMFCC_{i+1}" for i in range(13)] +
    ["ZCR", "Pitch", "Rolloff", "RMS"]
)
importances = clf.feature_importances_
top5_idx = np.argsort(importances)[::-1][:5]
print(f"\nTop 5 most important features:")
for rank, idx in enumerate(top5_idx):
    print(f"  {rank+1}. {feat_names[idx]} : {importances[idx]*100:.2f}%")

# --- Save ---
model_path = os.path.join(OUTPUT_PATH, "svm_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump({"model": clf, "scaler": scaler}, f)

print(f"\nModel saved to: {model_path}")
print("Step 2 complete. Run step3_register.py next.")
