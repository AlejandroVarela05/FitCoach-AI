# posture_classifier_mediapipe.py


# This module trains and evaluates binary posture classifiers (correct vs incorrect)
# using joint angles extracted from MediaPipe keypoints. It compares three models:
# a Multi‑Layer Perceptron (MLP), Random Forest (RF), and Support Vector Machine (SVM).
#
# PURPOSE:
#   - Load the ExeCheck dataset preprocessed with MediaPipe (99 keypoints per frame).
#   - Convert the 99‑dimensional keypoint vectors into 12 biomechanical joint angles.
#   - Train an MLP classifier and compare its performance against RF and SVM baselines.
#   - Save the trained model and generate comparison plots and a confusion matrix.
#
# COURSE CONNECTION:
#   This module relates to "Advanced Machine Learning" (Unit I – Deep Feed‑Forward
#   Networks) and "Computer Vision" (Unit I – feature extraction). The MLP
#   architecture follows the design patterns discussed in the lectures, including
#   batch normalisation and dropout for regularisation.
#
# DECISIONS:
#   - I reduce the 99 keypoints to 12 angles because angles are more interpretable
#     and directly relate to biomechanical form. This also reduces the input
#     dimensionality and makes the model less prone to overfitting.
#   - I use batch normalisation in the MLP to stabilise training and allow a higher
#     learning rate.
#   - I compare against RF and SVM to establish a baseline and verify that the
#     problem benefits from a non‑linear neural network.
#   - The dataset is split 60/20/20 (train/validation/test) using stratification
#     to preserve the correct/incorrect balance.



import os
import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED
)

# I use a safe import block here because TensorFlow may not be installed in all environments.
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[PostureClassifier] TensorFlow not available")

# I also try to import scikit‑learn for the baseline models.
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# I define the two classes for binary posture classification.
POSTURE_CLASSES = ["incorrect", "correct"]
NUM_CLASSES = len(POSTURE_CLASSES)

# Hyperparameters for the MLP model.
MLP_CONFIG = {
    "input_size": 12,      # 12 joint angles
    "num_classes": NUM_CLASSES,
    "dropout": 0.3,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50,
}

# I define output directories for processed data, model checkpoints, plots, and reports.
PROCESSED_POSTURE_DIR = PROCESSED_DATA_DIR / "posture_classifier_mediapipe"
CKPT_DIR = MODELS_DIR / "checkpoints" / "posture_mediapipe"
PLOTS_DIR = MODELS_DIR / "plots" / "posture_mediapipe"
REPORTS_DIR = MODELS_DIR / "reports" / "posture_mediapipe"

for d in (CKPT_DIR, PLOTS_DIR, REPORTS_DIR, PROCESSED_POSTURE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# I define a helper function to compute the angle (in degrees) between three 3D points.
def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# I convert a 99‑dimensional MediaPipe keypoint vector into 12 joint angles.
# These angles are: left/right knee, left/right hip, spine, left/right shoulder,
# left/right elbow, neck, torso lean, head tilt.
def landmarks_99_to_angles(keypoints_99):
    kp = np.array(keypoints_99).reshape(-1, 3)
    def p(idx):
        return kp[idx]
    
    angles = []
    # Left and right knee angles.
    angles.append(compute_angle(p(23), p(25), p(27)))
    angles.append(compute_angle(p(24), p(26), p(28)))
    # Left and right hip angles.
    angles.append(compute_angle(p(11), p(23), p(25)))
    angles.append(compute_angle(p(12), p(24), p(26)))
    # Spine angle (deviation from vertical).
    left_shoulder = p(11)
    right_shoulder = p(12)
    left_hip = p(23)
    right_hip = p(24)
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    spine_vec = shoulder_mid - hip_mid
    vertical = np.array([0, 1, 0])
    spine_angle = np.degrees(np.arccos(np.clip(np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec)+1e-8), -1, 1)))
    angles.append(spine_angle)
    # Left and right shoulder angles.
    angles.append(compute_angle(p(13), p(11), p(23)))
    angles.append(compute_angle(p(14), p(12), p(24)))
    # Left and right elbow angles.
    angles.append(compute_angle(p(11), p(13), p(15)))
    angles.append(compute_angle(p(12), p(14), p(16)))
    # Neck angle.
    neck_vec = p(0) - shoulder_mid
    neck_angle = np.degrees(np.arccos(np.clip(np.dot(neck_vec, vertical) / (np.linalg.norm(neck_vec)+1e-8), -1, 1)))
    angles.append(neck_angle)
    # Torso lean (same as spine angle for consistency).
    angles.append(spine_angle)
    # Head tilt (same as neck angle for consistency).
    angles.append(neck_angle)
    
    return np.array(angles, dtype=np.float32)

# I load the MediaPipe‑processed ExeCheck dataset and convert it to angle vectors.
# Each sample becomes a 12‑dimensional feature vector (the mean over all frames).
def load_mediapipe_posture_data(seq_len=30, stride=10, force_reprocess=False,
                                max_samples_per_class=None):
    data_root = RAW_DATA_DIR / "ExeCheck_Dataset" / "processed_dataset_mediapipe"
    proc_x = PROCESSED_POSTURE_DIR / "X_posture_mediapipe.npy"
    proc_y = PROCESSED_POSTURE_DIR / "y_posture_mediapipe.npy"
    
    if not force_reprocess and proc_x.exists() and proc_y.exists():
        print("[PostureData] Loading cached MediaPipe posture dataset...")
        X = np.load(proc_x); y = np.load(proc_y)
        print(f"  X: {X.shape}, y: {y.shape}")
        return X, y, _class_weights(y)
    
    print("[PostureData] Loading MediaPipe dataset...")
    joint_file = data_root / "seg_data_joint_mediapipe.npy"
    label_file = data_root / "seg_label_mediapipe.pkl"
    if not joint_file.exists() or not label_file.exists():
        raise FileNotFoundError(f"Files not found in {data_root}. Run process_execheck_mediapipe.py first.")
    
    X_raw = np.load(joint_file)          # shape (N, 160, 99)
    with open(label_file, 'rb') as f:
        labels = pickle.load(f)          # list of (class_id, correctness)
    
    correctness = np.array([lbl[1] for lbl in labels], dtype=np.int64)  # 0=incorrect, 1=correct
    print(f"Correctness distribution: {np.bincount(correctness)}")
    
    N, T, F = X_raw.shape
    X_list = []
    for i in range(N):
        seq_angles = []
        for t in range(T):
            kp_99 = X_raw[i, t, :].tolist()
            angles = landmarks_99_to_angles(kp_99)
            seq_angles.append(angles)
        # I average the angles over all frames to get a single feature vector per sample.
        mean_angles = np.mean(seq_angles, axis=0)
        X_list.append(mean_angles)
    
    X = np.array(X_list, dtype=np.float32)
    y = correctness
    
    # Optionally limit the number of samples per class (e.g., for balanced experiments).
    if max_samples_per_class:
        new_X, new_y = [], []
        for c in range(NUM_CLASSES):
            idx = np.where(y == c)[0]
            if len(idx) > max_samples_per_class:
                idx = idx[:max_samples_per_class]
            new_X.append(X[idx])
            new_y.append(y[idx])
        X = np.concatenate(new_X, axis=0)
        y = np.concatenate(new_y, axis=0)
        print(f"  Limited to {max_samples_per_class} samples per class -> {X.shape[0]} total")
    
    # Shuffle the dataset.
    perm = np.random.default_rng(RANDOM_SEED).permutation(len(X))
    X, y = X[perm], y[perm]
    
    print(f"[PostureData] Final dataset: {X.shape}, classes: {np.unique(y, return_counts=True)}")
    PROCESSED_POSTURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(proc_x, X)
    np.save(proc_y, y)
    return X, y, _class_weights(y)

# I compute class weights to handle any imbalance in the dataset.
def _class_weights(y):
    counts = np.bincount(y)
    n_total = len(y)
    n_cls = NUM_CLASSES
    return {i: n_total / (n_cls * max(c, 1)) for i, c in enumerate(counts)}

# I build the MLP classifier for posture assessment.
# The architecture uses three hidden layers with batch normalisation and dropout.
def build_mlp_posture(input_size=12, num_classes=NUM_CLASSES, dropout=0.3):
    inputs = keras.Input(shape=(input_size,), name="angles_input")
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)
    model = keras.Model(inputs, outputs, name="MLP_Posture_MediaPipe")
    model.compile(optimizer=keras.optimizers.Adam(MLP_CONFIG['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# This function orchestrates the training of the MLP, Random Forest, and SVM.
# It then evaluates them on the test set and saves comparison plots.
def train_and_compare():
    print("[PostureClassifier] Starting model comparison: MLP vs RF vs SVM")
    
    # Load the dataset (force reprocess to ensure we use the latest data).
    X, y, class_weights = load_mediapipe_posture_data(force_reprocess=True, max_samples_per_class=None)
    from sklearn.model_selection import train_test_split
    # Split into 60% train, 20% validation, 20% test (stratified).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    results = {}
    
    # Train the MLP if TensorFlow is available.
    if TF_AVAILABLE:
        print("\n[1] Training MLP (Keras)...")
        model = build_mlp_posture()
        model.summary(print_fn=lambda x: print("  " + x))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ]
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=MLP_CONFIG['epochs'],
            batch_size=MLP_CONFIG['batch_size'],
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        report = classification_report(y_test, y_pred, target_names=POSTURE_CLASSES, output_dict=True, zero_division=0)
        results["MLP"] = {
            "accuracy":  report["accuracy"],
            "macro_f1":  report["macro avg"]["f1-score"],
            "report":    report
        }
        model.save(str(CKPT_DIR / "mlp_posture_mediapipe_best.keras"))
        print(f"  [MLP] Test accuracy: {report['accuracy']:.4f}")

    # Train Random Forest baseline if scikit‑learn is available.
    if SKLEARN_AVAILABLE:
        print("\n[2] Training Random Forest baseline...")
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf  = rf.predict(X_test)
        report_rf  = classification_report(y_test, y_pred_rf, target_names=POSTURE_CLASSES, output_dict=True, zero_division=0)
        results["RandomForest"] = {
            "accuracy":  report_rf["accuracy"],
            "macro_f1":  report_rf["macro avg"]["f1-score"],
            "report":    report_rf
        }
        print(f"  [RF] Test accuracy: {report_rf['accuracy']:.4f}")
        import joblib
        joblib.dump(rf, CKPT_DIR / "random_forest_posture_mediapipe.joblib")
        print(f"  [RF] Model saved to {CKPT_DIR / 'random_forest_posture_mediapipe.joblib'}")

    # Train SVM baseline if scikit‑learn is available.
    if SKLEARN_AVAILABLE:
        print("\n[3] Training SVM (RBF kernel) baseline...")
        svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        report_svm = classification_report(y_test, y_pred_svm, target_names=POSTURE_CLASSES, output_dict=True, zero_division=0)
        results["SVM"] = {
            "accuracy":  report_svm["accuracy"],
            "macro_f1":  report_svm["macro avg"]["f1-score"],
            "report":    report_svm
        }
        print(f"  [SVM] Test accuracy: {report_svm['accuracy']:.4f}")
        joblib.dump(svm, CKPT_DIR / "svm_posture_mediapipe.joblib")
        print(f"  [SVM] Model saved to {CKPT_DIR / 'svm_posture_mediapipe.joblib'}")

    print("\nComparison summary:")
    for name, res in results.items():
        print(f"  {name:12} -- Accuracy: {res['accuracy']:.4f}, Macro F1: {res['macro_f1']:.4f}")
    
    # Generate a bar chart comparing accuracy and macro F1 for all models.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names = list(results.keys())
        accs = [results[n]["accuracy"] for n in names]
        f1s = [results[n]["macro_f1"] for n in names]
        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(x - width/2, accs, width, label='Accuracy')
        ax.bar(x + width/2, f1s, width, label='Macro F1')
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
        ax.set_ylim(0,1); ax.set_ylabel('Score'); ax.set_title('Posture Classifier Comparison (MediaPipe)')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        for i, (a,f) in enumerate(zip(accs, f1s)):
            ax.text(i - width/2, a+0.01, f"{a:.3f}", ha='center', fontsize=8)
            ax.text(i + width/2, f+0.01, f"{f:.3f}", ha='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(str(PLOTS_DIR / "posture_comparison.png"), dpi=120)
        plt.close(fig)
        
        # If MLP was trained, also save its confusion matrix.
        if "MLP" in results:
            y_pred_best = np.argmax(model.predict(X_test, verbose=0), axis=1)
            cm = confusion_matrix(y_test, y_pred_best)
            fig, ax = plt.subplots(figsize=(8,6))
            im = ax.imshow(cm, cmap='Blues')
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
            ax.set_xticklabels(POSTURE_CLASSES)
            ax.set_yticklabels(POSTURE_CLASSES)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            ax.set_title('MLP — Confusion Matrix (MediaPipe)')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                            color='white' if cm[i,j] > cm.max()/2 else 'black')
            plt.tight_layout()
            plt.savefig(str(PLOTS_DIR / "mlp_confusion_matrix.png"), dpi=110)
            plt.close(fig)
    except Exception as e:
        print(f"Plotting error: {e}")
    
    return results

# I define a manager class to easily load the trained MLP model and use it for inference.
class PostureClassifierManager:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or str(CKPT_DIR / "mlp_posture_mediapipe_best.keras")
        self._load_model()
    
    def _load_model(self):
        if not TF_AVAILABLE:
            print("[PostureManager] TensorFlow not available")
            return
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                print(f"[PostureManager] Loaded from {self.model_path}")
            except Exception as e:
                print(f"[PostureManager] Error loading: {e}")
        else:
            print("[PostureManager] No trained model found. Run train_and_compare() first.")
    
    def classify_angles(self, angles_vector):
        if self.model is None or angles_vector is None:
            return {'class': 'unknown', 'confidence': 0.0, 'probabilities': {}}
        angles = np.array(angles_vector).reshape(1, -1).astype(np.float32)
        probs = self.model.predict(angles, verbose=0)[0]
        idx = np.argmax(probs)
        return {
            'class': POSTURE_CLASSES[idx],
            'confidence': float(probs[idx]),
            'probabilities': {c: float(p) for c, p in zip(POSTURE_CLASSES, probs)}
        }

# Script entry point: train models or run a quick inference demo.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Posture Quality Classifier (MediaPipe)")
    parser.add_argument("--train", action="store_true", help="Train and compare models")
    parser.add_argument("--demo",  action="store_true", help="Run a quick demo inference")
    args = parser.parse_args()

    if args.train:
        train_and_compare()
    elif args.demo:
        manager = PostureClassifierManager()
        dummy_angles = np.random.randn(12) * 20 + 120
        result = manager.classify_angles(dummy_angles)
        print(f"[Demo] Prediction: {result['class']} (confidence={result['confidence']:.2f})")
    else:
        print("Usage: python posture_classifier_mediapipe.py --train | --demo")