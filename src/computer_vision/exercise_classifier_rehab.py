# exercise_classifier_rehab.py


# This module trains and evaluates five neural network architectures for
# classifying 10 rehabilitation exercises from the ExeCheck dataset. The goal is
# to find the best model for recognising exercises like arm circles, lunges, and
# squats from sequences of 3D keypoints.
#
# PURPOSE:
#   - Load preprocessed keypoint sequences (MediaPipe or Kinect).
#   - Build and train MLP, CNN1D, SimpleRNN, LSTM, and BiLSTM (with Attention).
#   - Save checkpoints, plots, and performance reports.
#   - Provide a manager class for real‑time inference.
#
# COURSE CONNECTION:
#   This work applies concepts from "Computer Vision" (Unit I – pose estimation,
#   Unit III – deep learning for vision) and "Advanced Machine Learning"
#   (Unit I – deep feed‑forward networks, Unit III – heuristic search for
#   hyperparameters). The custom Attention layer and the five‑architecture
#   comparison directly reflect the experimental evaluation required in the
#   teaching‑learning contracts.
#
# DECISIONS:
#   - I use TensorFlow/Keras because it provides a clean API and good GPU support.
#   - I save a session state (SessionManager) so I can interrupt and resume
#     long training runs without losing progress.
#   - I implement five architectures to understand which temporal modelling
#     approach works best for rehabilitation movements (balanced dataset).
#   - The BiLSTM includes an Attention layer (Unit I, Session 4) to focus on
#     the most discriminative frames (e.g., deepest point of a squat).
#   - I use sparse categorical cross‑entropy because labels are integers.
#   - Early stopping and ReduceLROnPlateau prevent overfitting and help
#     convergence (as discussed in the Deep Learning unit).



import os
import sys
import json
import shutil
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

# I insert the project root into the system path so I can import my configuration module.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import (
    BILSTM_CONFIG, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
)

# I try to import TensorFlow. If it fails, I set a flag and continue without crashing.
# This allows the script to be imported even when TensorFlow is not installed.
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[ExerciseClassifierRehab] TensorFlow not available")

# I define directories to store processed data, model checkpoints, plots, and reports.
# They are built from the central configuration file.
PROCESSED_REHAB_DIR = PROCESSED_DATA_DIR / "exercise_classifier_rehab"
CKPT_DIR     = MODELS_DIR / "checkpoints" / "exercise_rehab"
PLOTS_DIR    = MODELS_DIR / "plots"        / "exercise_rehab"
REPORTS_DIR  = MODELS_DIR / "reports"      / "exercise_rehab"
SESSION_FILE = CKPT_DIR   / "session.json"

# I save a periodic checkpoint every 5 epochs. This is a balance between
# disk usage and safety – I will not lose more than 5 epochs of work if training stops.
PERIODIC_CKPT_EVERY = 5

# These are the 10 rehabilitation exercises from the ExeCheck dataset.
# I keep them as a list so I can map integer labels to human‑readable names.
REHAB_CLASSES = [
    "arm_circle", "forward_lunge", "high_knee_raise", "hip_abduction",
    "leg_extension", "shoulder_abduction", "shoulder_external_rotation",
    "shoulder_flexion", "side_step_squat", "squat"
]
NUM_CLASSES = len(REHAB_CLASSES)

# I update the global configuration with the number of classes and the
# input feature size. For MediaPipe, we have 33 landmarks × 3 coordinates = 99.
BILSTM_CONFIG["num_classes"] = NUM_CLASSES
BILSTM_CONFIG["input_size"] = 99  # default for MediaPipe (33x3)

# If TensorFlow is available, I define a custom Attention layer.
# Attention was covered in the Advanced Machine Learning course (Unit I – Deep Learning).
# It allows the model to weigh the importance of each frame when making a decision.
if TF_AVAILABLE:
    @keras.utils.register_keras_serializable(package="FitCoach")
    class AttentionLayer(layers.Layer):
        def __init__(self, units=None, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.W = None
        def build(self, input_shape):
            # I create a single Dense unit to compute a scalar score for each time step.
            self.W = layers.Dense(1)
            super().build(input_shape)
        def call(self, lstm_output):
            # lstm_output shape: (batch, timesteps, features)
            # I compute scores, then apply softmax over the time axis to get attention weights.
            scores = self.W(lstm_output)
            weights = tf.nn.softmax(scores, axis=1)
            # The context vector is the weighted sum of the LSTM outputs.
            context = tf.reduce_sum(lstm_output * weights, axis=1)
            return context, weights
        def get_config(self):
            cfg = super().get_config()
            cfg.update({"units": self.units})
            return cfg

# I define a function to build the MLP baseline.
# MLP flattens the 30×99 input into one long vector and uses two dense layers.
# It cannot model temporal order explicitly, so it serves as a control.
def build_mlp(input_size=None, sequence_length=30, num_classes=NUM_CLASSES, dropout=0.3, **_):
    input_size = input_size or BILSTM_CONFIG['input_size']
    inputs = keras.Input(shape=(sequence_length, input_size), name="kp_input")
    x = layers.Flatten()(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name="MLP_Baseline")
    model.compile(optimizer=keras.optimizers.Adam(BILSTM_CONFIG['learning_rate']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# I build a 1D Convolutional Neural Network. This model looks for local patterns over time,
# like the rhythmic up‑and‑down of a squat. It is much smaller than recurrent models,
# making it suitable for mobile deployment.
def build_cnn1d(input_size=None, sequence_length=30, num_classes=NUM_CLASSES, dropout=0.3, **_):
    input_size = input_size or BILSTM_CONFIG['input_size']
    inputs = keras.Input(shape=(sequence_length, input_size), name="kp_input")
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name="CNN1D")
    model.compile(optimizer=keras.optimizers.Adam(BILSTM_CONFIG['learning_rate']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# SimpleRNN processes the sequence step by step but suffers from vanishing gradients.
# I include it to see if a more complex recurrent model is really necessary.
def build_simple_rnn(input_size=None, sequence_length=30, num_classes=NUM_CLASSES,
                     hidden_size=128, dropout=0.3, **_):
    input_size = input_size or BILSTM_CONFIG['input_size']
    inputs = keras.Input(shape=(sequence_length, input_size), name="kp_input")
    x = layers.SimpleRNN(hidden_size, dropout=dropout)(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name="SimpleRNN")
    model.compile(optimizer=keras.optimizers.Adam(BILSTM_CONFIG['learning_rate']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# LSTM can remember information over longer periods, which helps with full exercise cycles.
# I stack two LSTM layers: the first returns sequences, the second returns the final state.
def build_lstm(input_size=None, sequence_length=30, num_classes=NUM_CLASSES,
               hidden_size=128, dropout=0.3, **_):
    input_size = input_size or BILSTM_CONFIG['input_size']
    inputs = keras.Input(shape=(sequence_length, input_size), name="kp_input")
    x = layers.LSTM(hidden_size, return_sequences=True, dropout=dropout)(inputs)
    x = layers.LSTM(hidden_size, dropout=dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name="LSTM")
    model.compile(optimizer=keras.optimizers.Adam(BILSTM_CONFIG['learning_rate']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# BiLSTM with Attention is the most advanced model. It reads the sequence forward
# and backward, and attention helps it focus on the most informative frames
# (e.g., the bottom of a squat). This architecture is expected to capture
# long‑range dependencies and subtle differences.
def build_bilstm(input_size=None, sequence_length=30, num_classes=NUM_CLASSES,
                 hidden_size=128, num_layers=2, dropout=0.3, **_):
    input_size = input_size or BILSTM_CONFIG['input_size']
    hidden_size = hidden_size or BILSTM_CONFIG['hidden_size']
    num_layers = num_layers or BILSTM_CONFIG['num_layers']
    num_classes = num_classes or NUM_CLASSES
    dropout = dropout or BILSTM_CONFIG['dropout']
    sequence_length = sequence_length or BILSTM_CONFIG['sequence_length']
    inputs = keras.Input(shape=(sequence_length, input_size), name="kp_input")
    x = inputs
    for i in range(num_layers):
        x = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True, dropout=dropout),
            name=f"bilstm_{i+1}"
        )(x)
        if i < num_layers - 1:
            x = layers.Dropout(dropout)(x)
    context, _ = AttentionLayer(hidden_size*2)(x)
    x = layers.Dense(64, activation='relu')(context)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name="BiLSTM_Attention")
    model.compile(optimizer=keras.optimizers.Adam(BILSTM_CONFIG['learning_rate']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# I create a dictionary mapping model names to their builder functions.
# This makes it easy to loop over all five architectures during training.
MODEL_BUILDERS = {
    "MLP": build_mlp,
    "CNN1D": build_cnn1d,
    "SimpleRNN": build_simple_rnn,
    "LSTM": build_lstm,
    "BiLSTM": build_bilstm,
}

# SessionManager saves the training progress of each model in a JSON file.
# This way, if training is interrupted, I can resume from the last checkpoint
# without starting over. It also tracks the best validation accuracy seen so far.
class SessionManager:
    def __init__(self):
        for d in (CKPT_DIR, PLOTS_DIR, REPORTS_DIR):
            d.mkdir(parents=True, exist_ok=True)
        self.state = self._load()
    def _load(self):
        if SESSION_FILE.exists():
            with open(SESSION_FILE) as f:
                return json.load(f)
        blank = {"status": "pending", "best_val_acc": 0.0, "best_epoch": 0, "last_epoch": 0, "best_ckpt": None}
        return {"models": {n: blank.copy() for n in MODEL_BUILDERS}, "dataset_shape": None, "data_progress": {}}
    def save(self):
        with open(SESSION_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    def model_state(self, name):
        return self.state["models"].setdefault(name, {"status": "pending", "best_val_acc": 0.0, "best_epoch": 0, "last_epoch": 0, "best_ckpt": None})
    def is_done(self, name):
        return self.model_state(name)["status"] == "done"
    def mark_in_progress(self, name, last_epoch):
        s = self.model_state(name)
        s["status"] = "in_progress"
        s["last_epoch"] = last_epoch
        self.save()
    def update_best(self, name, val_acc, epoch, ckpt_path):
        s = self.model_state(name)
        s["best_val_acc"] = float(val_acc)
        s["best_epoch"] = int(epoch)
        s["best_ckpt"] = str(ckpt_path)
        self.save()
    def mark_done(self, name, val_acc, epoch, ckpt_path):
        s = self.model_state(name)
        s["status"] = "done"
        s["best_val_acc"] = float(val_acc)
        s["best_epoch"] = int(epoch)
        s["best_ckpt"] = str(ckpt_path)
        self.save()
    def initial_epoch(self, name):
        return self.model_state(name).get("last_epoch", 0)
    def best_ckpt_path(self, name):
        return CKPT_DIR / f"{name}_best.keras"
    def periodic_ckpt_path(self, name, epoch):
        return CKPT_DIR / f"{name}_ep{epoch:03d}.keras"
    def clean_periodic_ckpts(self, name, keep_epoch=None):
        for p in CKPT_DIR.glob(f"{name}_ep*.keras"):
            try:
                ep = int(p.stem.split("_ep")[-1])
                if keep_epoch is None or ep != keep_epoch:
                    p.unlink(missing_ok=True)
            except: pass
    def refresh_plot(self, name):
        p = PLOTS_DIR / f"{name}_training.png"
        p.unlink(missing_ok=True); return p
    def refresh_report(self, name):
        p = REPORTS_DIR / f"{name}_metrics.json"
        p.unlink(missing_ok=True); return p
    def refresh_comparison(self):
        for p in list(REPORTS_DIR.glob("comparison_*.json")) + list(PLOTS_DIR.glob("comparison_*.png")):
            p.unlink(missing_ok=True)
    def summary(self):
        # Print a quick overview of which models are done and their best accuracy.
        print("[Session] Status summary (ExeCheck Rehab):")
        for name in MODEL_BUILDERS:
            s = self.model_state(name)
            status = s["status"]
            print(f"  {name:<12} {status:<12} best_val_acc={s['best_val_acc']:.3f} best_epoch={s['best_epoch']}")
        dprog = self.state.get("data_progress", {})
        if dprog:
            print(f"  Data explored: {dprog.get('explored', 0)}/{dprog.get('total', 0)} ({dprog.get('explored_pct', 0.0):.1f}%)")

# This custom Keras callback is used during training. At the end of each epoch,
# it updates the session state, saves the best model, and creates periodic checkpoints.
if TF_AVAILABLE:
    class SessionCheckpoint(keras.callbacks.Callback):
        def __init__(self, model_name, session, history_ref):
            super().__init__()
            self.name_ = model_name
            self.session = session
            self.history_ = history_ref
            self._best_acc = session.model_state(model_name).get("best_val_acc", 0.0)
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_acc = logs.get("val_accuracy", 0.0)
            abs_epoch = self.session.model_state(self.name_)["last_epoch"] + 1
            self.history_.append({
                "epoch": abs_epoch,
                "accuracy": logs.get("accuracy", 0.0),
                "loss": logs.get("loss", 0.0),
                "val_accuracy": val_acc,
                "val_loss": logs.get("val_loss", 0.0),
            })
            saved = False
            if val_acc > self._best_acc:
                self._best_acc = val_acc
                ckpt = self.session.best_ckpt_path(self.name_)
                self.model.save(str(ckpt))
                self.session.update_best(self.name_, val_acc, abs_epoch, ckpt)
                saved = True
            if (epoch + 1) % PERIODIC_CKPT_EVERY == 0:
                pckpt = self.session.periodic_ckpt_path(self.name_, abs_epoch)
                self.model.save(str(pckpt))
                self.session.clean_periodic_ckpts(self.name_, keep_epoch=abs_epoch)
                saved = True
            self.session.model_state(self.name_)["last_epoch"] = abs_epoch
            self.session.save()
            if saved:
                self._save_plot_and_metrics()
        def _save_plot_and_metrics(self):
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                out = self.session.refresh_plot(self.name_)
                epochs = [h["epoch"] for h in self.history_]
                tr_acc = [h["accuracy"] for h in self.history_]
                val_acc = [h["val_accuracy"] for h in self.history_]
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(epochs, tr_acc, label='Train')
                ax.plot(epochs, val_acc, label='Val', linestyle='--')
                ax.set_ylim(0,1); ax.legend(); ax.grid(alpha=0.3)
                plt.savefig(str(out), dpi=100)
                plt.close(fig)
            except: pass
            with open(self.session.refresh_report(self.name_), 'w') as f:
                json.dump({"history": self.history_, "best_val_acc": self._best_acc}, f, indent=2)

# This function loads the original Kinect skeleton data from the ExeCheck dataset.
# I keep it for comparison, but the main pipeline uses MediaPipe (99 values).
def load_kinect_data(seq_len=30, stride=10, force_reprocess=False,
                     max_sequences_per_class=None):
    data_root = RAW_DATA_DIR / "ExeCheck_Dataset" / "processed_dataset_full"
    proc_x = PROCESSED_REHAB_DIR / "X_kinect.npy"
    proc_y = PROCESSED_REHAB_DIR / "y_kinect.npy"
    
    if not force_reprocess and proc_x.exists() and proc_y.exists():
        print("[DataLoader] Loading cached Kinect dataset...")
        X = np.load(proc_x); y = np.load(proc_y)
        print(f"  X: {X.shape}, y: {y.shape}")
        return X, y, _class_weights(y)
    
    print("[DataLoader] Loading Kinect dataset from processed_dataset_full...")
    joint_file = data_root / "seg_data_joint.npy"
    label_file = data_root / "seg_label.pkl"
    
    if not joint_file.exists() or not label_file.exists():
        raise FileNotFoundError(f"Files not found in {data_root}. Run gendata_fixed.py first.")
    
    joints = np.load(joint_file, allow_pickle=True)   # shape (700, 3, 160, 21, 1)
    with open(label_file, 'rb') as f:
        labels_data = pickle.load(f)
    
    print(f"Raw joints shape: {joints.shape}")
    print(f"Labels type: {type(labels_data)}")
    
    if isinstance(labels_data, tuple) and len(labels_data) >= 2:
        label_list = labels_data[1]
        print(f"  Number of label entries: {len(label_list)}")
    else:
        raise TypeError("Unexpected label format")
    
    y_raw = np.array([lbl[0] for lbl in label_list], dtype=np.int64)
    print(f"Extracted labels shape: {y_raw.shape}, unique: {np.unique(y_raw)}")
    
    joints = joints.squeeze(-1)                     # (700, 3, 160, 21)
    joints = np.transpose(joints, (0, 2, 3, 1))     # (700, 160, 21, 3)
    N, T, J, C = joints.shape
    X_raw = joints.reshape(N, T, J*C).astype(np.float32)  # (700, 160, 63)
    
    if T > seq_len:
        X_list, y_list = [], []
        for start in range(0, T - seq_len + 1, stride):
            X_list.append(X_raw[:, start:start+seq_len, :])
            y_list.append(y_raw)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
    else:
        X = X_raw
        y = y_raw
    
    if max_sequences_per_class:
        new_X, new_y = [], []
        for c in range(NUM_CLASSES):
            idx = np.where(y == c)[0]
            if len(idx) > max_sequences_per_class:
                idx = idx[:max_sequences_per_class]
            new_X.append(X[idx])
            new_y.append(y[idx])
        X = np.concatenate(new_X, axis=0)
        y = np.concatenate(new_y, axis=0)
        print(f"  Limited to {max_sequences_per_class} sequences per class -> {X.shape[0]} total")
    
    perm = np.random.default_rng(RANDOM_SEED).permutation(len(X))
    X, y = X[perm], y[perm]
    print(f"[DataLoader] Final Kinect dataset: {X.shape}, classes: {np.unique(y)}")
    PROCESSED_REHAB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(proc_x, X)
    np.save(proc_y, y)
    return X, y, _class_weights(y)

# This is the primary data loader. It reads the ExeCheck dataset preprocessed with MediaPipe.
# The data consists of 160‑frame sequences of 99‑dimensional keypoint vectors.
def load_mediapipe_data(seq_len=30, stride=10, force_reprocess=False,
                        max_sequences_per_class=None):
    data_root = RAW_DATA_DIR / "ExeCheck_Dataset" / "processed_dataset_mediapipe"
    proc_x = PROCESSED_REHAB_DIR / "X_mediapipe.npy"
    proc_y = PROCESSED_REHAB_DIR / "y_mediapipe.npy"
    
    if not force_reprocess and proc_x.exists() and proc_y.exists():
        print("[DataLoader] Loading cached MediaPipe dataset...")
        X = np.load(proc_x); y = np.load(proc_y)
        print(f"  X: {X.shape}, y: {y.shape}")
        return X, y, _class_weights(y)
    
    print("[DataLoader] Loading MediaPipe dataset...")
    joint_file = data_root / "seg_data_joint_mediapipe.npy"
    label_file = data_root / "seg_label_mediapipe.pkl"
    if not joint_file.exists() or not label_file.exists():
        raise FileNotFoundError(f"Files not found in {data_root}. Run process_execheck_mediapipe.py first.")
    
    X_raw = np.load(joint_file)  # shape (N, 160, 99)
    with open(label_file, 'rb') as f:
        labels = pickle.load(f)  # list of (class_id, correctness)
    y_raw = np.array([l[0] for l in labels], dtype=np.int64)
    
    N, T, F = X_raw.shape
    if T > seq_len:
        X_list, y_list = [], []
        for start in range(0, T - seq_len + 1, stride):
            X_list.append(X_raw[:, start:start+seq_len, :])
            y_list.append(y_raw)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
    else:
        X = X_raw
        y = y_raw
    
    if max_sequences_per_class:
        new_X, new_y = [], []
        for c in range(NUM_CLASSES):
            idx = np.where(y == c)[0]
            if len(idx) > max_sequences_per_class:
                idx = idx[:max_sequences_per_class]
            new_X.append(X[idx])
            new_y.append(y[idx])
        X = np.concatenate(new_X, axis=0)
        y = np.concatenate(new_y, axis=0)
        print(f"  Limited to {max_sequences_per_class} sequences per class -> {X.shape[0]} total")
    
    perm = np.random.default_rng(RANDOM_SEED).permutation(len(X))
    X, y = X[perm], y[perm]
    print(f"[DataLoader] Final MediaPipe dataset: {X.shape}, classes: {np.unique(y)}")
    PROCESSED_REHAB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(proc_x, X)
    np.save(proc_y, y)
    return X, y, _class_weights(y)

# I compute class weights to handle any imbalance in the dataset.
# The formula weight_i = total_samples / (num_classes * count_i) gives higher weight
# to minority classes, which helps the model pay more attention to them.
def _class_weights(y):
    counts = np.bincount(y)
    n_total = len(y)
    n_cls = NUM_CLASSES
    return {i: n_total / (n_cls * max(c, 1)) for i, c in enumerate(counts)}

# I use online data augmentation by adding small Gaussian noise to the keypoints.
# This is a form of regularisation that helps the model become more robust to small
# variations in pose estimation (jitter).
def create_augmented_dataset(X, y, batch_size, augment_noise_std=0.02):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size)
    if augment_noise_std > 0:
        def add_noise(x, y):
            noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=augment_noise_std)
            x = x + noise
            return x, y
        dataset = dataset.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# This function handles the training of a single architecture. It can resume from a
# checkpoint and uses early stopping and learning rate reduction to improve convergence.
def train_one_model(model_name, X_tr, y_tr, X_val, y_val,
                    class_weights, session, epochs, input_size,
                    use_augmentation=False):
    print(f"[Train] Starting model: {model_name}")

    s = session.model_state(model_name)
    initial_epoch = s.get("last_epoch", 0)
    best_ckpt = session.best_ckpt_path(model_name)

    if best_ckpt.exists():
        print(f"  Resuming from checkpoint (epoch {initial_epoch})")
        model = keras.models.load_model(
            str(best_ckpt),
            custom_objects={"AttentionLayer": AttentionLayer})
    else:
        print(f"  Building new model")
        model = MODEL_BUILDERS[model_name](input_size=input_size)

    model.summary(print_fn=lambda x: print("  " + x))

    history_list = []
    mp = REPORTS_DIR / f"{model_name}_metrics.json"
    if mp.exists():
        with open(mp) as f:
            history_list = json.load(f).get("history", [])
        print(f"  Loaded {len(history_list)} previous epoch records")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        SessionCheckpoint(model_name, session, history_list),
    ]

    session.mark_in_progress(model_name, initial_epoch)

    if use_augmentation:
        print("  Using online augmentation (Gaussian noise, std=0.02)")
        train_dataset = create_augmented_dataset(X_tr, y_tr, BILSTM_CONFIG['batch_size'], augment_noise_std=0.02)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BILSTM_CONFIG['batch_size'])
        model.fit(
            train_dataset,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=val_dataset,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        model.fit(
            X_tr, y_tr,
            epochs=epochs,
            initial_epoch=initial_epoch,
            batch_size=BILSTM_CONFIG['batch_size'],
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

    best_val_acc = session.model_state(model_name)["best_val_acc"]
    best_epoch = session.model_state(model_name)["best_epoch"]
    session.mark_done(model_name, best_val_acc, best_epoch, best_ckpt)

    _save_training_plot(model_name, history_list, session)
    _save_model_metrics(model_name, history_list, session)

    print(f"[Train] {model_name} done -- best val_acc={best_val_acc:.4f} @ epoch {best_epoch}")
    return model

# I save a plot showing training and validation accuracy over epochs.
def _save_training_plot(model_name, history, session):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        out_path = session.refresh_plot(model_name)
        epochs = [h["epoch"] for h in history]
        tr_acc = [h["accuracy"] for h in history]
        val_acc = [h["val_accuracy"] for h in history]
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(epochs, tr_acc, label='Train')
        ax.plot(epochs, val_acc, label='Val', linestyle='--')
        ax.set_ylim(0,1); ax.legend(); ax.grid(alpha=0.3)
        plt.savefig(str(out_path), dpi=100)
        plt.close(fig)
    except: pass

# I save the training history and best validation accuracy as a JSON file.
def _save_model_metrics(model_name, history, session):
    out_path = session.refresh_report(model_name)
    with open(out_path, 'w') as f:
        json.dump({
            "model": model_name,
            "history": history,
            "best_val_acc": session.model_state(model_name)["best_val_acc"],
            "best_epoch": session.model_state(model_name)["best_epoch"],
        }, f, indent=2)

# This is the main orchestration function. It loads the data, splits it into
# train/validation/test sets (70/15/15 stratified), and then trains each architecture.
def train_all_models(reset_session=False, epochs=None, force_reprocess=False,
                     max_sequences_per_class=None, use_augmentation=False,
                     use_mediapipe=True):
    if not TF_AVAILABLE:
        print("ERROR: TensorFlow not installed.")
        return
    from sklearn.model_selection import train_test_split

    if use_mediapipe:
        BILSTM_CONFIG["input_size"] = 99
        load_func = load_mediapipe_data
        print("Using MediaPipe dataset (99 keypoints per frame)")
    else:
        BILSTM_CONFIG["input_size"] = 63
        load_func = load_kinect_data
        print("Using Kinect dataset (63 keypoints per frame)")

    if reset_session:
        print("[Session] Resetting all checkpoints...")
        for d in (CKPT_DIR, PLOTS_DIR, REPORTS_DIR):
            if d.exists(): shutil.rmtree(d)
        for p in (PROCESSED_REHAB_DIR / "X_mediapipe.npy", PROCESSED_REHAB_DIR / "y_mediapipe.npy",
                  PROCESSED_REHAB_DIR / "X_kinect.npy", PROCESSED_REHAB_DIR / "y_kinect.npy"):
            p.unlink(missing_ok=True)

    session = SessionManager()
    session.summary()

    X, y, cw = load_func(force_reprocess=force_reprocess,
                         max_sequences_per_class=max_sequences_per_class)
    session.state["dataset_shape"] = list(X.shape)
    session.state["data_progress"] = {"explored": len(X), "total": len(X), "explored_pct": 100.0}
    session.save()

    n_feat = X.shape[2]
    print(f"\n  Dataset: {X.shape}")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_tmp)

    split_file = CKPT_DIR / "test_split.npz"
    if not split_file.exists():
        np.savez(str(split_file), X_te=X_te, y_te=y_te)
    else:
        loaded = np.load(str(split_file))
        X_te, y_te = loaded['X_te'], loaded['y_te']

    print(f"  Train:{len(X_tr)}  Val:{len(X_val)}  Test:{len(X_te)}")
    n_epochs = epochs or BILSTM_CONFIG.get('epochs', 50)

    for model_name in MODEL_BUILDERS:
        if session.is_done(model_name):
            print(f"\n[Session] {model_name} already done — skipping")
            continue
        train_one_model(
            model_name=model_name,
            X_tr=X_tr, y_tr=y_tr,
            X_val=X_val, y_val=y_val,
            class_weights=cw,
            session=session,
            epochs=n_epochs,
            input_size=n_feat,
            use_augmentation=use_augmentation,
        )
        session.refresh_comparison()
        save_comparison_report(session, X_te, y_te)

    session.summary()
    print(f"\n[Done] Plots -> {PLOTS_DIR}")
    print(f"[Done] Reports -> {REPORTS_DIR}")
    print(f"[Done] Checkpts -> {CKPT_DIR}")

# I generate a final comparison report that includes test accuracy, macro F1,
# confusion matrices, and a bar chart comparing validation vs test accuracy.
def save_comparison_report(session, X_te, y_te):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        return
    session.refresh_comparison()
    summary = []
    for name in MODEL_BUILDERS:
        s = session.model_state(name)
        if s["status"] != "done" or not s["best_ckpt"]:
            continue
        model = keras.models.load_model(s["best_ckpt"], custom_objects={"AttentionLayer": AttentionLayer})
        y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
        unique_labels = np.unique(y_te)
        target_names_present = [REHAB_CLASSES[i] for i in unique_labels if i < len(REHAB_CLASSES)]
        report = classification_report(y_te, y_pred,
                                       labels=unique_labels,
                                       target_names=target_names_present,
                                       output_dict=True,
                                       zero_division=0)
        summary.append({
            "model": name,
            "best_val_acc": s["best_val_acc"],
            "test_acc": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "n_params": model.count_params(),
        })
        cm = confusion_matrix(y_te, y_pred, labels=unique_labels)
        fig, ax = plt.subplots(figsize=(max(6, len(unique_labels)*0.8), max(5, len(unique_labels)*0.7)))
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(unique_labels)))
        ax.set_yticks(range(len(unique_labels)))
        ax.set_xticklabels(target_names_present, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(target_names_present, fontsize=8)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'{name} -- Confusion Matrix')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=7)
        plt.tight_layout()
        plt.savefig(str(PLOTS_DIR / f"comparison_confusion_{name}.png"), dpi=110)
        plt.close(fig)
    if summary:
        names = [r["model"] for r in summary]
        val_acc = [r["best_val_acc"] for r in summary]
        t_acc = [r["test_acc"] for r in summary]
        fig, ax = plt.subplots(figsize=(10,6))
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width/2, val_acc, width, label='Best Val Acc')
        ax.bar(x + width/2, t_acc, width, label='Test Acc')
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
        ax.set_ylim(0,1); ax.set_ylabel('Accuracy'); ax.set_title('Model Comparison (ExeCheck)')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        for i, (v,t) in enumerate(zip(val_acc, t_acc)):
            ax.text(i - width/2, v+0.01, f"{v:.3f}", ha='center', fontsize=8)
            ax.text(i + width/2, t+0.01, f"{t:.3f}", ha='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(str(PLOTS_DIR / "comparison_accuracy.png"), dpi=120)
        plt.close(fig)
        with open(REPORTS_DIR / "comparison_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

# INCL is a list of landmark indices used to convert a 99‑value MediaPipe vector
# to a 63‑value Kinect‑style vector (21 joints × 3). This is only used if a Kinect
# model is loaded with MediaPipe input.
INCL = [0, 1, 2, 3, 5, 6, 7, 12, 13, 14, 18, 19, 20, 22, 23, 24, 27, 8, 15, 21, 25]

# The ExerciseClassifierManager class provides a simple interface for inference.
# I use it in the live demo to load the best trained model and predict exercises
# frame‑by‑frame from a webcam stream.
class ExerciseClassifierManager:
    def __init__(self, model_path=None):
        self.model_path = model_path or str(CKPT_DIR / "BiLSTM_best.keras")
        self.sequence_length = BILSTM_CONFIG['sequence_length']
        self.classes = REHAB_CLASSES
        self.keypoints_buffer = []
        self.model = self._load_model()
        self.expected_input_size = self.model.input_shape[-1] if self.model else 99

    def _convert_99_to_63(self, keypoints_99):
        # I reshape the 99‑vector to (33,3), select the 21 Kinect joints, and flatten back.
        kp = np.array(keypoints_99).reshape(-1, 3)
        selected = kp[INCL]
        return selected.flatten().tolist()

    def _load_model(self):
        if not TF_AVAILABLE:
            return None
        for p in [Path(self.model_path), CKPT_DIR / "BiLSTM_best.keras"]:
            if p.exists():
                try:
                    m = keras.models.load_model(
                        str(p),
                        custom_objects={"AttentionLayer": AttentionLayer}
                    )
                    print(f"[Manager] Loaded {p.name} (input_size={m.input_shape[-1]})")
                    return m
                except Exception as e:
                    print(f"[Manager] Load error: {e}")
        print("[Manager] No trained model — untrained demo mode")
        return build_bilstm()

    def add_frame(self, kv):
        if kv is None or self.model is None:
            return None, 0.0
        # If the model expects 63 features but I received 99, I convert.
        if self.expected_input_size == 63 and len(kv) == 99:
            kv = self._convert_99_to_63(kv)
        elif self.expected_input_size == 99 and len(kv) == 63:
            # I cannot convert from 63 to 99, so I just pass it through (should not happen).
            pass
        self.keypoints_buffer.append(kv)
        if len(self.keypoints_buffer) > self.sequence_length:
            self.keypoints_buffer = self.keypoints_buffer[-self.sequence_length:]
        if len(self.keypoints_buffer) < self.sequence_length:
            return None, 0.0
        window = np.array(self.keypoints_buffer[-self.sequence_length:], dtype=np.float32)
        return self.predict_exercise(window)

    def predict_exercise(self, keypoints_window):
        if self.model is None:
            return self.classes[0], 0.0
        x = np.expand_dims(keypoints_window, 0)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])

    def reset_buffer(self):
        self.keypoints_buffer = []

    def save_model(self, path=None):
        if self.model is None:
            return
        p = Path(path or self.model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(p))
        print(f"[Manager] Saved to {p}")

# This is the script entry point. It parses command‑line arguments and starts the training process.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ExeCheck Exercise Classifier (5 architectures)")
    parser.add_argument("--reset", action="store_true", help="Delete all checkpoints and start fresh")
    parser.add_argument("--reprocess", action="store_true", help="Force reload of raw data")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs per model")
    parser.add_argument("--max-sequences-per-class", type=int, default=None, help="Limit sequences per class")
    parser.add_argument("--augment", action="store_true", help="Use online Gaussian noise augmentation")
    parser.add_argument("--kinect", action="store_true", help="Use Kinect dataset (63 keypoints) instead of MediaPipe (99)")
    args = parser.parse_args()

    use_mediapipe = not args.kinect  # default to MediaPipe

    train_all_models(reset_session=args.reset,
                     epochs=args.epochs,
                     force_reprocess=args.reprocess,
                     max_sequences_per_class=args.max_sequences_per_class,
                     use_augmentation=args.augment,
                     use_mediapipe=use_mediapipe)