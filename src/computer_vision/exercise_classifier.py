# exercise_classifier.py


# This module trains and evaluates five neural network architectures for
# classifying 8 general fitness exercises using multiple public datasets.
#
# PURPOSE:
#   - Load and combine keypoint sequences from RiccardoRiccio CSV, QEVD-COACH NPY,
#     Gym/Workout videos, and QEVD-300k videos.
#   - Build and train MLP, CNN1D, SimpleRNN, LSTM, and BiLSTM (with Attention).
#   - Manage checkpoints, plots, and comparison reports.
#   - Provide a manager class for real‑time inference.
#
# COURSE CONNECTION:
#   This script applies concepts from "Computer Vision" (Unit I – pose estimation
#   and feature extraction) and "Advanced Machine Learning" (Unit I – deep
#   feed‑forward networks, recurrent models, attention). The five‑architecture
#   comparison directly fulfills the experimental evaluation required in the
#   teaching‑learning contract.
#
# DECISIONS:
#   - I use TensorFlow/Keras for a clean API and GPU acceleration.
#   - A SessionManager saves progress, allowing me to resume long training runs.
#   - I combine multiple datasets to increase diversity and robustness.
#   - The BiLSTM includes an Attention layer (Unit I, Session 4) to focus on
#     discriminative frames (e.g., bottom of a squat).
#   - Class weights and data augmentation handle the severe class imbalance
#     present in the general fitness dataset (e.g., plank vs. barbell curl).
#   - QEVD-300k processing is incremental to avoid memory overload.



import os
import sys
import json
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

# I insert the project root into the path to import my configuration module.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import (
    BILSTM_CONFIG, EXERCISE_CLASSES, MODELS_DIR,
    RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
)

# I try to import TensorFlow. If it is missing, I set a flag to continue without crashing.
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[ExerciseClassifier] TensorFlow not available — demo mode only")

# I define output directories for processed data, checkpoints, plots, and reports.
PROCESSED_EXERCISE_DIR = PROCESSED_DATA_DIR / "exercise_classifier"
CKPT_DIR     = MODELS_DIR / "checkpoints" / "exercise"
PLOTS_DIR    = MODELS_DIR / "plots"        / "exercise"
REPORTS_DIR  = MODELS_DIR / "reports"      / "exercise"
SESSION_FILE = CKPT_DIR   / "session.json"

# I cap the number of QEVD-300k videos per class to keep training manageable.
MAX_QEVD_PER_CLASS  = 300   # budget cap for QEVD-300k processing
# By default, I add 1800 new QEVD-300k videos per run (incremental loading).
DEFAULT_QEVD_BATCH_TOTAL = 1800  # videos added from QEVD-300k per run
# I save a periodic checkpoint every 5 epochs to avoid losing progress.
PERIODIC_CKPT_EVERY = 5     # save a periodic checkpoint every N epochs
MAX_VIDEO_PROCESS_WARNINGS = 5
QEVD_PROGRESS_FILE = PROCESSED_EXERCISE_DIR / "qevd_300k_progress.json"


def _count_sequences(seq_dict):
    """Count the total number of sequences stored in a dictionary of lists."""
    return int(sum(len(v) for v in seq_dict.values()))


def _print_source_summary(name, seq_dict):
    """Print a summary of how many sequences were loaded from a given source."""
    total = _count_sequences(seq_dict)
    if total == 0:
        print(f"  [{name}] 0 sequences")
        return
    print(f"  [{name}] {total} sequences")
    per_cls = [f"{cls}:{len(seq_dict.get(cls, []))}" for cls in EXERCISE_CLASSES]
    print("    " + " | ".join(per_cls))


def _progress_pct(explored, total):
    """Calculate the percentage of completion."""
    if total <= 0:
        return 100.0
    return (100.0 * explored) / total


def _print_progress_line(label, explored, total, unit):
    """Print a formatted progress line (e.g., 'Processed 150/200 videos (75.0%)')."""
    pct = _progress_pct(explored, total)
    print(f"  [Progress] {label}: {explored}/{total} {unit} ({pct:.1f}%)")


def _compute_overall_progress(source_stats):
    """Aggregate exploration statistics from multiple data sources."""
    total = int(sum(s.get("total", 0) for s in source_stats))
    explored = int(sum(s.get("explored", 0) for s in source_stats))
    return {
        "explored": explored,
        "total": total,
        "explored_pct": round(_progress_pct(explored, total), 2),
        "sources": source_stats,
    }

# This mapping translates folder names and video labels into the 8 canonical class names.
VIDEO_FOLDER_TO_CLASS = {
    "push-up": "push_up", "push_up": "push_up",
    "pushup": "push_up", "push up": "push_up",
    "push-ups": "push_up", "pushups": "push_up",
    "squat": "squat", "squats": "squat",
    "barbell squat": "squat", "goblet squat": "squat",
    "bodyweight squat": "squat",
    "shoulder press": "shoulder_press",
    "shoulder_press": "shoulder_press",
    "overhead press": "shoulder_press",
    "military press": "shoulder_press",
    "dumbbell shoulder press": "shoulder_press",
    "seated shoulder press": "shoulder_press",
    "barbell biceps curl": "barbell_biceps_curl",
    "barbell_biceps_curl": "barbell_biceps_curl",
    "bicep curl": "barbell_biceps_curl",
    "biceps curl": "barbell_biceps_curl",
    "dumbbell bicep curl": "barbell_biceps_curl",
    "hammer curl": "barbell_biceps_curl",
    "curl": "barbell_biceps_curl",
    "plank": "plank", "forearm plank": "plank", "side plank": "plank",
    "leg raises": "leg_raises", "leg_raises": "leg_raises",
    "leg raise": "leg_raises", "hanging leg raise": "leg_raises",
    "lying leg raise": "leg_raises",
    "lateral raise": "lateral_raise", "lateral_raise": "lateral_raise",
    "dumbbell lateral raise": "lateral_raise", "side raise": "lateral_raise",
    "deadlift": "deadlift", "romanian deadlift": "deadlift",
    "romanian_deadlift": "deadlift", "rdl": "deadlift",
    "sumo deadlift": "deadlift", "conventional deadlift": "deadlift",
}
# Video file extensions that I support when reading raw video files.
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv',
                    '.MOV', '.MP4', '.AVI', '.MKV'}


# I define a custom Attention layer. Attention was covered in the Advanced Machine Learning course (Unit I – Deep Learning).
# It allows the model to weigh the importance of each time step.
if TF_AVAILABLE:
    @keras.utils.register_keras_serializable(package="FitCoach")
    class AttentionLayer(layers.Layer):
        def __init__(self, units=None, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.W = None

        def build(self, input_shape):
            # I create a single Dense unit to compute a scalar score per time step.
            self.W = layers.Dense(1)
            super().build(input_shape)

        def call(self, lstm_output):
            # lstm_output shape: (batch, timesteps, features)
            scores  = self.W(lstm_output)                           # (B, T, 1)
            weights = tf.nn.softmax(scores, axis=1)                 # (B, T, 1)
            context = tf.reduce_sum(lstm_output * weights, axis=1)  # (B, H)
            return context, weights

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"units": self.units})
            return cfg


# I build a simple MLP baseline: it flattens the 30x99 input and uses two dense layers.
# It cannot model temporal order explicitly, so it serves as a control.
def build_mlp(input_size=99, sequence_length=30, num_classes=8,
              dropout=0.3, **_):
    inputs  = keras.Input(shape=(sequence_length, input_size), name="kp_input")
    x       = layers.Flatten(name="flatten")(inputs)
    x       = layers.Dense(256, activation='relu', name="dense_1")(x)
    x       = layers.Dropout(dropout, name="drop_1")(x)
    x       = layers.Dense(128, activation='relu', name="dense_2")(x)
    x       = layers.Dropout(dropout, name="drop_2")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)
    model   = keras.Model(inputs, outputs, name="MLP_Baseline")
    model.compile(optimizer=keras.optimizers.Adam(BILSTM_CONFIG['learning_rate']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# CNN1D applies 1D convolutions over the time dimension. This allows the model
# to detect local temporal patterns, like the rhythm of a squat.
def build_cnn1d(input_size=99, sequence_length=30, num_classes=8,
                dropout=0.3, **_):
    inputs = keras.Input(shape=(sequence_length, input_size), name="kp_input")
    x = layers.Conv1D(64,  3, activation='relu', padding='same', name="conv_1")(inputs)
    x = layers.MaxPooling1D(2, name="pool_1")(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same', name="conv_2")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(128, activation='relu', name="dense_1")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)
    model   = keras.Model(inputs, outputs, name="CNN1D")
    model.compile(optimizer=keras.optimizers.Adam(BILSTM_CONFIG['learning_rate']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# SimpleRNN processes the sequence step by step but suffers from vanishing gradients.
# I include it to see if a more complex recurrent model is really needed.
def build_simple_rnn(input_size=99, sequence_length=30, num_classes=8,
                     hidden_size=128, dropout=0.3, **_):
    inputs  = keras.Input(shape=(sequence_length, input_size), name="kp_input")
    x       = layers.SimpleRNN(hidden_size, dropout=dropout, name="rnn")(inputs)
    x       = layers.Dense(64, activation='relu', name="dense_1")(x)
    x       = layers.Dropout(dropout, name="drop")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)
    model   = keras.Model(inputs, outputs, name="SimpleRNN")
    model.compile(optimizer=keras.optimizers.Adam(BILSTM_CONFIG['learning_rate']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# LSTM can remember information over longer periods, which helps with full exercise cycles.
# I stack two LSTM layers: the first returns sequences, the second returns the final state.
def build_lstm(input_size=99, sequence_length=30, num_classes=8,
               hidden_size=128, dropout=0.3, **_):
    inputs = keras.Input(shape=(sequence_length, input_size), name="kp_input")
    x = layers.LSTM(hidden_size, return_sequences=True,
                    dropout=dropout, name="lstm_1")(inputs)
    x = layers.LSTM(hidden_size, dropout=dropout, name="lstm_2")(x)
    x = layers.Dense(64, activation='relu', name="dense_1")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)
    model   = keras.Model(inputs, outputs, name="LSTM")
    model.compile(optimizer=keras.optimizers.Adam(BILSTM_CONFIG['learning_rate']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# BiLSTM with Attention is the most advanced model. It reads the sequence forward
# and backward, and attention helps it focus on the most informative frames.
def build_bilstm(input_size=None, hidden_size=None, num_layers=None,
                 num_classes=None, dropout=None, sequence_length=None, **_):
    if not TF_AVAILABLE:
        return None

    input_size      = input_size      or BILSTM_CONFIG['input_size']
    hidden_size     = hidden_size     or BILSTM_CONFIG['hidden_size']
    num_layers      = num_layers      or BILSTM_CONFIG['num_layers']
    num_classes     = num_classes     or BILSTM_CONFIG['num_classes']
    dropout         = dropout         or BILSTM_CONFIG['dropout']
    sequence_length = sequence_length or BILSTM_CONFIG['sequence_length']

    inputs = keras.Input(shape=(sequence_length, input_size), name="kp_input")
    x = inputs
    for i in range(num_layers):
        x = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True,
                        dropout=dropout, name=f"lstm_{i+1}"),
            name=f"bilstm_{i+1}"
        )(x)
        if i < num_layers - 1:
            x = layers.Dropout(dropout, name=f"drop_lstm_{i+1}")(x)

    context, _ = AttentionLayer(hidden_size * 2, name="attention")(x)
    x          = layers.Dense(64, activation='relu', name="dense_1")(context)
    x          = layers.Dropout(dropout, name="drop_dense")(x)
    outputs    = layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Attention")
    model.compile(optimizer=keras.optimizers.Adam(BILSTM_CONFIG['learning_rate']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# This dictionary maps model names to their builder functions, making it easy to loop over them.
MODEL_BUILDERS = {
    "MLP":       build_mlp,
    "CNN1D":     build_cnn1d,
    "SimpleRNN": build_simple_rnn,
    "LSTM":      build_lstm,
    "BiLSTM":    build_bilstm,
}


# SessionManager saves the training progress of each model in a JSON file.
# It allows me to resume interrupted training and track the best validation accuracy.
class SessionManager:
    def __init__(self):
        for d in (CKPT_DIR, PLOTS_DIR, REPORTS_DIR):
            d.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self):
        if SESSION_FILE.exists():
            with open(SESSION_FILE) as f:
                state = json.load(f)
            print(f"[Session] Resumed from {SESSION_FILE}")
            return state
        blank = {"status": "pending", "best_val_acc": 0.0,
                 "best_epoch": 0, "last_epoch": 0, "best_ckpt": None}
        return {"models": {n: blank.copy() for n in MODEL_BUILDERS},
            "dataset_shape": None,
            "data_progress": {"explored": 0, "total": 0,
                      "explored_pct": 0.0, "sources": []}}

    def save(self):
        with open(SESSION_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def model_state(self, name):
        return self.state["models"].setdefault(
            name, {"status": "pending", "best_val_acc": 0.0,
                   "best_epoch": 0, "last_epoch": 0, "best_ckpt": None})

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
        s["best_epoch"]   = int(epoch)
        s["best_ckpt"]    = str(ckpt_path)
        self.save()

    def mark_done(self, name, val_acc, epoch, ckpt_path):
        s = self.model_state(name)
        s["status"]       = "done"
        s["best_val_acc"] = float(val_acc)
        s["best_epoch"]   = int(epoch)
        s["best_ckpt"]    = str(ckpt_path)
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
            except ValueError:
                pass

    def refresh_plot(self, name):
        p = PLOTS_DIR / f"{name}_training.png"
        p.unlink(missing_ok=True)
        return p

    def refresh_report(self, name):
        p = REPORTS_DIR / f"{name}_metrics.json"
        p.unlink(missing_ok=True)
        return p

    def refresh_comparison(self):
        for p in list(REPORTS_DIR.glob("comparison_*.json")) + \
                 list(PLOTS_DIR.glob("comparison_*.png")):
            p.unlink(missing_ok=True)

    def summary(self):
        icons = {"done": "✓", "in_progress": "►", "pending": "○"}
        print("SESSION SUMMARY")
        for name in MODEL_BUILDERS:
            s    = self.model_state(name)
            icon = icons.get(s["status"], "?")
            print(f"  {icon} {name:<12} {s['status']:<12} "
                  f"best_val_acc={s['best_val_acc']:.3f}  "
                  f"best_epoch={s['best_epoch']}")
        dprog = self.state.get("data_progress", {})
        if dprog:
            print(f"  Data explored: {dprog.get('explored', 0)}/"
                f"{dprog.get('total', 0)} "
                f"({float(dprog.get('explored_pct', 0.0)):.1f}%)")
        print("=" * 60 + "\n")


# This custom Keras callback saves checkpoints and updates the session state at the end of each epoch.
if TF_AVAILABLE:
    class SessionCheckpoint(keras.callbacks.Callback):
        def __init__(self, model_name, session, history_ref):
            super().__init__()
            self.name_    = model_name
            self.session  = session
            self.history_ = history_ref      # shared list, mutated in place
            self._best_acc = session.model_state(model_name).get(
                "best_val_acc", 0.0)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_acc  = float(logs.get("val_accuracy", 0.0))
            val_loss = float(logs.get("val_loss",     9999.0))
            tr_acc   = float(logs.get("accuracy",     0.0))
            tr_loss  = float(logs.get("loss",         9999.0))

            abs_epoch = self.session.model_state(self.name_)["last_epoch"] + 1

            self.history_.append({
                "epoch":        abs_epoch,
                "accuracy":     tr_acc,
                "loss":         tr_loss,
                "val_accuracy": val_acc,
                "val_loss":     val_loss,
            })

            saved_any = False

            if val_acc > self._best_acc:
                self._best_acc = val_acc
                ckpt = self.session.best_ckpt_path(self.name_)
                self.model.save(str(ckpt))
                self.session.update_best(self.name_, val_acc, abs_epoch, ckpt)

                dprog = self.session.state.get("data_progress", {})
                explored_pct = float(dprog.get("explored_pct", 0.0))
                model_pct = 100.0 * abs_epoch / max(1, int(self.params.get("epochs", 1)))
                print(f"\n  [✓ BEST] {self.name_}  ep{abs_epoch}"
                    f"  val_acc={val_acc:.4f}  → {ckpt.name}"
                    f"  | data={explored_pct:.1f}% explored"
                    f" | model={model_pct:.1f}%")
                saved_any = True

            if (epoch + 1) % PERIODIC_CKPT_EVERY == 0:
                pckpt = self.session.periodic_ckpt_path(self.name_, abs_epoch)
                self.model.save(str(pckpt))
                self.session.clean_periodic_ckpts(self.name_,
                                                   keep_epoch=abs_epoch)
                dprog = self.session.state.get("data_progress", {})
                explored_pct = float(dprog.get("explored_pct", 0.0))
                model_pct = 100.0 * abs_epoch / max(1, int(self.params.get("epochs", 1)))
                print(f"  [↓ CKPT] {self.name_}  ep{abs_epoch}  → {pckpt.name}"
                      f"  | data={explored_pct:.1f}% explored"
                      f" | model={model_pct:.1f}%")
                saved_any = True

            self.session.model_state(self.name_)["last_epoch"] = abs_epoch
            self.session.save()

            if saved_any:
                _save_training_plot(self.name_, self.history_, self.session)
                _save_model_metrics(self.name_, self.history_, self.session)


# It creates a side‑by‑side plot of accuracy and loss over epochs.
def _save_training_plot(model_name, history, session):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_path = session.refresh_plot(model_name)
    epochs   = [h["epoch"]        for h in history]
    tr_acc   = [h["accuracy"]     for h in history]
    val_acc  = [h["val_accuracy"] for h in history]
    tr_loss  = [h["loss"]         for h in history]
    val_loss = [h["val_loss"]     for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name} — Training Progress",
                 fontsize=14, fontweight='bold')

    ax1.plot(epochs, tr_acc,  lw=2,  label='Train')
    ax1.plot(epochs, val_acc, lw=2,  label='Validation', linestyle='--')
    ax1.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy",
            ylim=(0, 1)); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, tr_loss,  lw=2,  label='Train')
    ax2.plot(epochs, val_loss, lw=2,  label='Validation', linestyle='--')
    ax2.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close(fig)


def _save_model_metrics(model_name, history, session):
    out_path = session.refresh_report(model_name)
    with open(out_path, 'w') as f:
        json.dump({
            "model":        model_name,
            "history":      history,
            "best_val_acc": session.model_state(model_name)["best_val_acc"],
            "best_epoch":   session.model_state(model_name)["best_epoch"],
        }, f, indent=2)


# This function generates a comprehensive comparison report after all models are trained.
# It saves confusion matrices, a bar chart comparing validation vs test accuracy, and a JSON summary.
def save_comparison_report(session, X_te, y_te):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("[Report] matplotlib / sklearn not available — skipping")
        return

    session.refresh_comparison()

    summary = []
    for name in MODEL_BUILDERS:
        s = session.model_state(name)
        if s["status"] != "done" or not s["best_ckpt"]:
            continue
        try:
            model = keras.models.load_model(
                s["best_ckpt"],
                custom_objects={"AttentionLayer": AttentionLayer})
        except Exception as e:
            print(f"  [Report] Could not load {name}: {e}"); continue

        y_pred   = np.argmax(model.predict(X_te, verbose=0), axis=1)
        report   = classification_report(y_te, y_pred,
                                         target_names=EXERCISE_CLASSES,
                                         output_dict=True)
        summary.append({
            "model":        name,
            "best_val_acc": s["best_val_acc"],
            "test_acc":     float(report["accuracy"]),
            "macro_f1":     float(report["macro avg"]["f1-score"]),
            "n_params":     int(model.count_params()),
            "per_class_f1": {cls: report[cls]["f1-score"]
                             for cls in EXERCISE_CLASSES if cls in report},
        })

        cm  = confusion_matrix(y_te, y_pred)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set(xticks=range(len(EXERCISE_CLASSES)),
               yticks=range(len(EXERCISE_CLASSES)),
               xticklabels=EXERCISE_CLASSES,
               yticklabels=EXERCISE_CLASSES,
               xlabel='Predicted', ylabel='True',
               title=f'{name} — Confusion Matrix')
        plt.setp(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black',
                        fontsize=8)
        plt.tight_layout()
        plt.savefig(str(PLOTS_DIR / f"comparison_confusion_{name}.png"),
                    dpi=110, bbox_inches='tight')
        plt.close(fig)

    if not summary:
        return

    names   = [r["model"]       for r in summary]
    val_acc = [r["best_val_acc"] for r in summary]
    t_acc   = [r["test_acc"]    for r in summary]

    fig, ax = plt.subplots(figsize=(10, 6))
    x, w = np.arange(len(names)), 0.35
    ax.bar(x - w/2, val_acc, w, label='Best Val Acc', color='steelblue')
    ax.bar(x + w/2, t_acc,   w, label='Test Acc',     color='tomato')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.set_title("Model Comparison")
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for xi, (v, t) in enumerate(zip(val_acc, t_acc)):
        ax.text(xi - w/2, v + 0.01, f"{v:.3f}", ha='center', fontsize=8)
        ax.text(xi + w/2, t + 0.01, f"{t:.3f}", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "comparison_accuracy.png"),
                dpi=120, bbox_inches='tight')
    plt.close(fig)

    with open(REPORTS_DIR / "comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n[Report] Comparison saved:")
    for r in summary:
        print(f"  {r['model']:<12} val={r['best_val_acc']:.3f}  "
              f"test={r['test_acc']:.3f}  "
              f"f1={r['macro_f1']:.3f}  "
              f"params={r['n_params']:,}")


# QEVD metadata comes in different JSON formats. This function extracts a mapping
# from video stem to canonical class name.
def _load_qevd_labels(json_path):
    if not Path(json_path).exists():
        return {}
    with open(json_path) as f:
        data = json.load(f)

    def _candidate_texts(info):
        texts = []
        for key in ('label', 'exercise', 'exercise_name', 'class', 'activity'):
            val = info.get(key)
            if isinstance(val, str):
                texts.append(val)
            elif isinstance(val, list):
                texts.extend(v for v in val if isinstance(v, str))
        for key in ('labels', 'labels_descriptive'):
            val = info.get(key)
            if isinstance(val, list):
                texts.extend(v for v in val if isinstance(v, str))
            elif isinstance(val, str):
                texts.append(val)
        return texts

    def _to_class(raw_text):
        raw = str(raw_text).lower().strip()
        if not raw:
            return None
        base = raw.split(' - ')[0].strip()
        variants = [raw, base]
        for v in variants:
            cls = VIDEO_FOLDER_TO_CLASS.get(v)
            if cls in EXERCISE_CLASSES:
                return cls
        for key, cls in VIDEO_FOLDER_TO_CLASS.items():
            if cls in EXERCISE_CLASSES and (base.startswith(key) or raw.startswith(key)):
                return cls
        return None

    mapping = {}

    if isinstance(data, dict):
        iterable = data.items()
    elif isinstance(data, list):
        iterable = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            vid = (entry.get('video_path') or entry.get('video')
                   or entry.get('clip_path') or entry.get('video_id'))
            if vid is None:
                continue
            iterable.append((str(vid), entry))
    else:
        return {}

    for vid_id, info in iterable:
        if not isinstance(info, dict):
            continue
        stem = Path(str(vid_id)).stem
        cls = None
        for txt in _candidate_texts(info):
            cls = _to_class(txt)
            if cls:
                break
        if cls:
            mapping[stem] = cls
    return mapping


# I use MediaPipe to extract 33 landmarks (99 values) from a video file.
# This function is called for every raw video in Gym/Workout and QEVD-300k.
def _extract_keypoints_from_video(video_path):
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        return None

    if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'pose'):
        return None

    pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        pose.close()
        return None

    if not hasattr(_extract_keypoints_from_video, "_warn_count"):
        _extract_keypoints_from_video._warn_count = 0

    frames = []
    try:
        while True:
            try:
                ret, frame = cap.read()
            except cv2.error as e:
                if _extract_keypoints_from_video._warn_count < MAX_VIDEO_PROCESS_WARNINGS:
                    print(f"  [VideoSkip] OpenCV read error in {Path(video_path).name}: {e}")
                    _extract_keypoints_from_video._warn_count += 1
                break

            if not ret or frame is None:
                break

            try:
                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except (cv2.error, MemoryError) as e:
                if _extract_keypoints_from_video._warn_count < MAX_VIDEO_PROCESS_WARNINGS:
                    print(f"  [VideoSkip] Pose processing error in {Path(video_path).name}: {e}")
                    _extract_keypoints_from_video._warn_count += 1
                break

            if res.pose_landmarks:
                kps = []
                for lm in res.pose_landmarks.landmark:
                    kps.extend([lm.x, lm.y, lm.z])
                frames.append(np.array(kps, np.float32))
    finally:
        cap.release()
        pose.close()

    return np.array(frames, np.float32) if len(frames) >= 10 else None


# RiccardoRiccio dataset provides pre‑extracted keypoints in CSV files.
# This function reads those CSVs and creates sequence windows.
def _load_riccardo_data(raw_dir, seq_len=30, stride=10):
    try:
        import pandas as pd
    except ImportError:
        return {}, {"name": "Riccardo CSV", "explored": 0, "total": 0, "unit": "csv"}
    root = Path(raw_dir) / "Real-Time_Exercise_Recognition_Dataset"
    if not root.exists():
        return {}, {"name": "Riccardo CSV", "explored": 0, "total": 0, "unit": "csv"}
    all_seqs = defaultdict(list)
    csv_files = [p for p in root.rglob("*.csv") if p.stat().st_size > 10_000]
    explored = 0
    for csv_path in csv_files:
        explored += 1
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        lc = next((c for c in ['label','class','exercise','activity']
                   if c in df.columns), None)
        if not lc:
            continue
        ic = next((c for c in ['video_id','vid_id','video','clip_id','id']
                   if c in df.columns), None)
        skip = {lc} | ({ic} if ic else set())
        fc = [c for c in df.columns
              if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
        if not fc:
            continue
        if ic:
            groups = df.groupby(ic)
        else:
            df['_g'] = (df[lc] != df[lc].shift()).cumsum()
            groups   = df.groupby('_g')
        for _, grp in groups:
            cls = VIDEO_FOLDER_TO_CLASS.get(
                str(grp[lc].iloc[0]).lower().strip())
            if cls not in EXERCISE_CLASSES:
                continue
            frames = grp[fc].values.astype(np.float32)
            if len(frames) < seq_len:
                continue
            for s in range(0, len(frames) - seq_len + 1, stride):
                all_seqs[cls].append(frames[s: s + seq_len])
    return dict(all_seqs), {
        "name": "Riccardo CSV",
        "explored": explored,
        "total": len(csv_files),
        "unit": "csv",
    }


# QEVD-COACH provides pre‑extracted keypoints as .npy files.
# This function loads them directly without running MediaPipe again.
def _load_qevd_coach_npy(raw_dir, seq_len=30, stride=10):
    qevd_root = Path(raw_dir) / "QEVD"
    if not qevd_root.exists():
        return {}, {"name": "QEVD-COACH NPY", "explored": 0, "total": 0, "unit": "npy"}
    subdirs = [
        qevd_root / "QEVD-FIT-COACH"           / "QEVD-FIT-COACH",
        qevd_root / "QEVD-FIT-COACH-Benchmark" / "QEVD-FIT-COACH-Benchmark",
        qevd_root / "QEVD-FIT-COACH-Competition-CVPR2025"
                  / "QEVD-FIT-COACH-Competition-CVPR2025",
    ]
    all_seqs = defaultdict(list)
    total_npy = 0
    explored = 0
    for subdir in subdirs:
        if not subdir.exists():
            continue
        lj = next((subdir / c for c in
                   ('fine_grained_labels.json', 'feedbacks_long_range.json',
                    'fine_grained_labels_with_worker_ids.json')
                   if (subdir / c).exists()), None)
        if not lj:
            continue
        s2c = _load_qevd_labels(lj)
        for vdir in ['long_range_videos', 'short_clips', '.']:
            nd = subdir / vdir
            if not nd.exists():
                continue
            npy_files = list(nd.glob("*.npy"))
            total_npy += len(npy_files)
            for npy_path in npy_files:
                explored += 1
                cls = s2c.get(npy_path.stem)
                if cls not in EXERCISE_CLASSES:
                    continue
                try:
                    kps = np.load(str(npy_path)).astype(np.float32)
                except Exception:
                    continue
                if kps.ndim == 3:
                    kps = kps.reshape(kps.shape[0], -1)
                elif kps.ndim == 2 and kps.shape[0] < kps.shape[1]:
                    kps = kps.T
                if kps.ndim != 2 or kps.shape[0] < seq_len:
                    continue
                for s in range(0, kps.shape[0] - seq_len + 1, stride):
                    all_seqs[cls].append(kps[s: s + seq_len])
    return dict(all_seqs), {
        "name": "QEVD-COACH NPY",
        "explored": explored,
        "total": total_npy,
        "unit": "npy",
    }


# Gym/Workout videos are raw .mp4 files. I run MediaPipe on them to extract keypoints.
# To avoid extremely long processing, I can limit the number of videos per class.
def _process_video_folders(raw_dir, seq_len=30, stride=10, max_videos_per_class=None):
    raw_dir  = Path(raw_dir)
    all_seqs = defaultdict(list)
    candidates = []
    allowed_ext = {e.lower() for e in VIDEO_EXTENSIONS}

    for root in [raw_dir / "Gym_WorkoutExercises_Video",
                 raw_dir / "WorkoutExercises_Video"]:
        if not root.exists():
            continue
        for folder in root.rglob("*"):
            if not folder.is_dir():
                continue
            cls = VIDEO_FOLDER_TO_CLASS.get(folder.name.lower().strip())
            if cls not in EXERCISE_CLASSES:
                continue
            for f in folder.iterdir():
                if f.suffix.lower() not in allowed_ext:
                    continue
                candidates.append((cls, f))

    if max_videos_per_class and max_videos_per_class > 0:
        class_videos = defaultdict(list)
        for cls, path in candidates:
            class_videos[cls].append((cls, path))
        limited = []
        for cls, lst in class_videos.items():
            limited.extend(lst[:max_videos_per_class])
        candidates = limited
        print(f"  [Gym/Workout] limited to {max_videos_per_class} videos per class → {len(candidates)} total")

    total_videos = len(candidates)
    print(f"  [Gym/Workout] candidates to process: {total_videos} videos")
    explored = 0
    skipped = 0
    for cls, f in candidates:
        explored += 1
        if explored % 25 == 0 or explored == total_videos:
            _print_progress_line("Gym/Workout videos", explored, total_videos, "videos")
        kps = _extract_keypoints_from_video(f)
        if kps is None:
            skipped += 1
            continue
        for s in range(0, len(kps) - seq_len + 1, stride):
            all_seqs[cls].append(kps[s: s + seq_len])
    if skipped:
        print(f"  [Gym/Workout] skipped videos: {skipped}/{total_videos}")
    return dict(all_seqs), {
        "name": "Gym/Workout videos",
        "explored": explored,
        "total": total_videos,
        "unit": "videos",
    }


# This helper builds a mapping from class to video paths for QEVD-300k,
# applying a per‑class cap to avoid loading millions of videos.
def _build_qevd_300k_class_paths(raw_dir, max_per_class=300):
    qevd_root = Path(raw_dir) / "QEVD"
    if not qevd_root.exists():
        return {}, {}

    part_dirs = [p for p in [
        qevd_root / "QEVD-FIT-300k-Part-1" / "QEVD-FIT-300k-Part-1",
        qevd_root / "QEVD-FIT-300k-Part-2" / "QEVD-FIT-300k-Part-2",
        qevd_root / "QEVD-FIT-300k-Part-3" / "QEVD-FIT-300k-Part-3",
        qevd_root / "QEVD-FIT-300k-Part-4" / "QEVD-FIT-300k-Part-4",
    ] if p.exists()]

    stem_to_path = {}
    stem_to_class = {}
    for part in part_dirs:
        part_labels = _load_qevd_labels(part / "fine_grained_labels.json")
        stem_to_class.update(part_labels)
        for f in part.iterdir():
            if f.suffix.lower() in {e.lower() for e in VIDEO_EXTENSIONS}:
                stem_to_path[f.stem] = f

    class_paths = defaultdict(list)
    for stem, cls in stem_to_class.items():
        if cls in EXERCISE_CLASSES and stem in stem_to_path:
            class_paths[cls].append(stem_to_path[stem])

    rng = np.random.default_rng(RANDOM_SEED)
    limited = {}
    for cls in EXERCISE_CLASSES:
        paths = class_paths.get(cls, [])
        if not paths:
            limited[cls] = []
            continue
        order = [paths[i] for i in rng.permutation(len(paths))]
        limited[cls] = order if max_per_class is None else order[:max_per_class]

    totals = {cls: len(limited.get(cls, [])) for cls in EXERCISE_CLASSES}
    return limited, totals


# I persist the number of QEVD-300k videos already processed so I can resume later.
def _load_qevd_progress(class_totals):
    if not QEVD_PROGRESS_FILE.exists():
        return {cls: 0 for cls in EXERCISE_CLASSES}
    try:
        with open(QEVD_PROGRESS_FILE) as f:
            state = json.load(f)
    except Exception:
        return {cls: 0 for cls in EXERCISE_CLASSES}

    offsets = state.get("offsets", {}) if isinstance(state, dict) else {}
    clean = {}
    for cls in EXERCISE_CLASSES:
        off = int(offsets.get(cls, 0))
        clean[cls] = max(0, min(off, int(class_totals.get(cls, 0))))
    return clean


def _save_qevd_progress(offsets, class_totals):
    QEVD_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QEVD_PROGRESS_FILE, "w") as f:
        json.dump({
            "offsets": {cls: int(offsets.get(cls, 0)) for cls in EXERCISE_CLASSES},
            "class_totals": {cls: int(class_totals.get(cls, 0)) for cls in EXERCISE_CLASSES},
            "processed_total": int(sum(offsets.get(cls, 0) for cls in EXERCISE_CLASSES)),
            "available_total": int(sum(class_totals.get(cls, 0) for cls in EXERCISE_CLASSES)),
        }, f, indent=2)


# I load QEVD-300k videos incrementally. Each run adds a batch of videos,
# processes them with MediaPipe, and saves the progress.
def _load_qevd_300k(raw_dir, seq_len=30, stride=15, max_per_class=300,
                    batch_total=None, reset_progress=False):
    class_paths, class_totals = _build_qevd_300k_class_paths(raw_dir, max_per_class=max_per_class)
    total_available = int(sum(class_totals.values()))
    if total_available == 0:
        return {}, {"name": "QEVD-300k videos", "explored": 0, "total": 0, "unit": "videos"}

    if reset_progress:
        QEVD_PROGRESS_FILE.unlink(missing_ok=True)

    offsets = _load_qevd_progress(class_totals)
    already_done = int(sum(offsets.values()))
    remaining = max(0, total_available - already_done)
    if remaining == 0:
        print("  [QEVD-300k] all capped videos already consumed in previous runs.")
        return {}, {
            "name": "QEVD-300k videos",
            "explored": 0,
            "total": 0,
            "unit": "videos",
            "overall_explored": already_done,
            "overall_total": total_available,
        }

    if batch_total is None or batch_total <= 0:
        batch_total = remaining
    target_batch = min(int(batch_total), remaining)

    selected = []
    selected_by_class = {cls: 0 for cls in EXERCISE_CLASSES}
    while len(selected) < target_batch:
        progressed = False
        for cls in EXERCISE_CLASSES:
            idx = offsets.get(cls, 0)
            paths = class_paths.get(cls, [])
            if idx < len(paths):
                selected.append((cls, paths[idx]))
                offsets[cls] = idx + 1
                selected_by_class[cls] += 1
                progressed = True
                if len(selected) >= target_batch:
                    break
        if not progressed:
            break

    print(f"  [QEVD-300k] run chunk: {len(selected)} videos (remaining after this run: {remaining - len(selected)})")
    for cls in EXERCISE_CLASSES:
        total_cls = class_totals.get(cls, 0)
        if total_cls == 0 and selected_by_class.get(cls, 0) == 0:
            print(f"  [QEVD-300k] '{cls}': processing 0/0...")
            continue
        done_cls = offsets.get(cls, 0)
        chunk_cls = selected_by_class.get(cls, 0)
        print(f"  [QEVD-300k] '{cls}': this run {chunk_cls} | consumed {done_cls}/{total_cls}")

    all_seqs = defaultdict(list)
    explored = 0
    total_selected = len(selected)
    for cls, path in selected:
        explored += 1
        if explored % 25 == 0 or explored == total_selected:
            _print_progress_line("QEVD-300k videos", explored, total_selected, "videos")
        kps = _extract_keypoints_from_video(path)
        if kps is None:
            continue
        for s in range(0, len(kps) - seq_len + 1, stride):
            all_seqs[cls].append(kps[s: s + seq_len])

    _save_qevd_progress(offsets, class_totals)
    return dict(all_seqs), {
        "name": "QEVD-300k videos",
        "explored": explored,
        "total": total_selected,
        "unit": "videos",
        "overall_explored": int(sum(offsets.values())),
        "overall_total": total_available,
    }


# Different sources may produce sequences with varying numbers of features.
# I normalise them all to 99 dimensions (33 landmarks × 3).
def _normalise_feat(seq_dict, target=99):
    out = {}
    for cls, seqs in seq_dict.items():
        norm = []
        for seq in seqs:
            T, F = seq.shape
            if F == target:
                norm.append(seq)
            elif F < target:
                norm.append(np.hstack([seq,
                            np.zeros((T, target-F), np.float32)]))
            else:
                norm.append(seq[:, :target])
        out[cls] = norm
    return out


# I augment minority classes to reduce the severe imbalance.
# Two techniques are used: horizontal flip (inverting x coordinates) and temporal scaling.
def _augment(sequences, target_count):
    aug     = list(sequences)
    rng     = np.random.default_rng(RANDOM_SEED)
    n_feat  = sequences[0].shape[1]
    seq_len = sequences[0].shape[0]
    while len(aug) < target_count:
        orig = sequences[rng.integers(0, len(sequences))]
        if rng.integers(0, 2) == 0:
            fl = orig.copy()
            for j in range(0, n_feat, 3):
                fl[:, j] = 1.0 - fl[:, j]
            aug.append(fl.astype(np.float32))
        else:
            factor = rng.uniform(0.8, 1.2)
            n_src  = max(2, int(seq_len * factor))
            src_i  = np.linspace(0, seq_len-1, n_src)
            dst_i  = np.linspace(0, n_src-1,   seq_len)
            rs     = np.zeros((seq_len, n_feat), np.float32)
            for f in range(n_feat):
                col = np.interp(src_i, np.arange(seq_len), orig[:, f])
                rs[:, f] = np.interp(dst_i, np.arange(n_src), col)
            aug.append(rs)
    return aug[:target_count]


# If no real data is available for a class (e.g., during debugging), I can generate
# synthetic sinusoidal patterns as a fallback. This is only used with --allow-synthetic-fallback.
def _synthetic(class_list, n_per=50, seq_len=30, n_feat=99):
    out = {}
    for k, cls in enumerate(class_list):
        t    = np.linspace(0, 2*np.pi, seq_len)
        base = np.sin((0.5 + k*0.4)*t + k*np.pi/3)
        seqs = []
        for _ in range(n_per):
            seq = np.zeros((seq_len, n_feat), np.float32)
            for j in range(n_feat):
                seq[:, j] = (base * (0.3 + 0.1*np.sin(j*0.5))
                             + np.random.normal(0, 0.05, seq_len) + 0.5)
            seqs.append(seq)
        out[cls] = seqs
    return out


# I compute class weights using the standard formula: weight = total / (num_classes * count).
# This gives higher weight to minority classes, mitigating imbalance.
def _class_weights(y):
    counts  = np.bincount(y.astype(int))
    n_total = len(y)
    n_cls   = len(EXERCISE_CLASSES)
    return {i: n_total / (n_cls * max(c, 1)) for i, c in enumerate(counts)}


def generate_demo_data(n=200, seq_len=30, n_feat=99, n_cls=8):
    synth = _synthetic(EXERCISE_CLASSES[:n_cls], n_per=n//n_cls,
                       seq_len=seq_len, n_feat=n_feat)
    X, y = [], []
    for i, cls in enumerate(EXERCISE_CLASSES[:n_cls]):
        for seq in synth[cls]:
            X.append(seq); y.append(i)
    return np.array(X, np.float32), np.array(y, np.int64)


# This is the main data loading function. It orchestrates all sources, caches the result,
# and applies augmentation if requested.
def load_dataset(raw_dir=None, force_reprocess=False,
                 seq_len=30, stride=10, augment_minority=True,
                 use_qevd_300k=True, qevd_max=None,
                 allow_synthetic_fallback=False,
                 qevd_batch_total=DEFAULT_QEVD_BATCH_TOTAL,
                 max_videos_per_class=None, max_sequences_per_class=None):
    raw_dir = Path(raw_dir or RAW_DATA_DIR)
    proc_x  = PROCESSED_EXERCISE_DIR / "X.npy"
    proc_y  = PROCESSED_EXERCISE_DIR / "y.npy"

    if force_reprocess:
        QEVD_PROGRESS_FILE.unlink(missing_ok=True)

    if not force_reprocess and proc_x.exists() and proc_y.exists():
        print("[DataLoader] Loading cached dataset...")
        X = np.load(proc_x); y = np.load(proc_y)
        print(f"  X: {X.shape}  y: {y.shape}")
        if use_qevd_300k and qevd_batch_total and qevd_batch_total > 0:
            print(f"[DataLoader] Incremental QEVD-300k chunk: {qevd_batch_total} videos/run")
            max_300k = qevd_max if (qevd_max is not None and qevd_max > 0) else MAX_QEVD_PER_CLASS
            if qevd_max == 0:
                max_300k = None
            seqs_300k, q300k_stats = _load_qevd_300k(
                raw_dir, seq_len, stride=15,
                max_per_class=max_300k,
                batch_total=qevd_batch_total,
                reset_progress=False)
            added_X, added_y = [], []
            for i, cls in enumerate(EXERCISE_CLASSES):
                for seq in seqs_300k.get(cls, []):
                    added_X.append(seq)
                    added_y.append(i)
            if added_X:
                X_new = np.array(added_X, np.float32)
                y_new = np.array(added_y, np.int64)
                X = np.concatenate([X, X_new], axis=0)
                y = np.concatenate([y, y_new], axis=0)
                perm = np.random.default_rng(RANDOM_SEED).permutation(len(X))
                X, y = X[perm], y[perm]
                np.save(proc_x, X)
                np.save(proc_y, y)
                print(f"  [DataLoader] Added QEVD sequences this run: {len(X_new)}")
                if "overall_explored" in q300k_stats:
                    _print_progress_line(
                        "QEVD-300k consumed",
                        q300k_stats["overall_explored"],
                        q300k_stats["overall_total"],
                        "videos")
            else:
                print("  [DataLoader] No new QEVD-300k videos left in the configured cap.")
        cached_progress = {
            "explored": int(len(X)),
            "total": int(len(X)),
            "explored_pct": 100.0,
            "sources": [{"name": "cached", "explored": int(len(X)),
                         "total": int(len(X)), "unit": "samples"}],
        }
        return X, y, _class_weights(y), cached_progress

    print("[DataLoader] Building from raw sources...")
    merged = defaultdict(list)
    source_stats = []

    print("\n[1/4] RiccardoRiccio CSVs...")
    seqs_ric, ric_stats = _load_riccardo_data(raw_dir, seq_len, stride)
    _print_source_summary("Riccardo", seqs_ric)
    _print_progress_line(ric_stats["name"], ric_stats["explored"], ric_stats["total"], ric_stats["unit"])
    source_stats.append(ric_stats)
    for cls, seqs in seqs_ric.items():
        merged[cls].extend(seqs)

    print("\n[2/4] QEVD-COACH NPY (pre-extracted)...")
    seqs_coach, coach_stats = _load_qevd_coach_npy(raw_dir, seq_len, stride)
    _print_source_summary("QEVD-COACH", seqs_coach)
    _print_progress_line(coach_stats["name"], coach_stats["explored"], coach_stats["total"], coach_stats["unit"])
    source_stats.append(coach_stats)
    for cls, seqs in seqs_coach.items():
        merged[cls].extend(seqs)

    print("\n[3/4] Gym + WorkoutExercises (MediaPipe)...")
    seqs_vid, vid_stats = _process_video_folders(raw_dir, seq_len, stride, max_videos_per_class=max_videos_per_class)
    _print_source_summary("Gym/Workout videos", seqs_vid)
    _print_progress_line(vid_stats["name"], vid_stats["explored"], vid_stats["total"], vid_stats["unit"])
    source_stats.append(vid_stats)
    for cls, seqs in seqs_vid.items():
        ex = len(merged.get(cls, []))
        merged[cls].extend(seqs if ex < 50 else seqs[:max(len(seqs)//2, 50)])

    if use_qevd_300k:
        print("\n[4/4] QEVD-300k sampled videos (MediaPipe)...")
        max_300k = qevd_max if (qevd_max is not None and qevd_max > 0) else MAX_QEVD_PER_CLASS
        if qevd_max == 0:
            max_300k = None  # 0 means: use all available videos per class
        seqs_300k, q300k_stats = _load_qevd_300k(
                raw_dir, seq_len, stride=15,
                max_per_class=max_300k,
                batch_total=qevd_batch_total,
                reset_progress=force_reprocess)
        _print_source_summary("QEVD-300k", seqs_300k)
        _print_progress_line(q300k_stats["name"], q300k_stats["explored"], q300k_stats["total"], q300k_stats["unit"])
        if "overall_explored" in q300k_stats:
            _print_progress_line("QEVD-300k consumed", q300k_stats["overall_explored"], q300k_stats["overall_total"], "videos")
        source_stats.append(q300k_stats)
        for cls, seqs in seqs_300k.items():
            merged[cls].extend(seqs)
    else:
        print("\n[4/4] QEVD-300k skipped.")
        source_stats.append({
            "name": "QEVD-300k videos",
            "explored": 0,
            "total": 0,
            "unit": "videos",
        })

    missing = [c for c in EXERCISE_CLASSES if not merged.get(c)]
    if missing:
        if allow_synthetic_fallback:
            print(f"  WARNING: no real data for {missing} — synthetic fallback")
            for cls, seqs in _synthetic(missing, n_per=50, seq_len=seq_len).items():
                merged[cls] = seqs
        else:
            raise RuntimeError(
                "No real sequences found for classes: "
                f"{missing}. Use --allow-synthetic-fallback if you want "
                "to proceed with synthetic data, or run in an environment "
                "that can extract MediaPipe pose from videos."
            )

    merged = _normalise_feat(dict(merged), target=99)

    if max_sequences_per_class and max_sequences_per_class > 0:
        for cls in list(merged.keys()):
            if len(merged[cls]) > max_sequences_per_class:
                merged[cls] = merged[cls][:max_sequences_per_class]
                print(f"  Truncated '{cls}' to {max_sequences_per_class} sequences")

    if augment_minority:
        max_n  = max(len(v) for v in merged.values())
        target = max(50, int(max_n * 0.8))
        for cls in EXERCISE_CLASSES:
            n = len(merged.get(cls, []))
            if 0 < n < target:
                print(f"  Augmenting '{cls}': {n} → {target}")
                merged[cls] = _augment(merged[cls], target)

    X_list, y_list = [], []
    for i, cls in enumerate(EXERCISE_CLASSES):
        for seq in merged.get(cls, []):
            X_list.append(seq); y_list.append(i)

    X = np.array(X_list, np.float32)
    y = np.array(y_list, np.int64)
    perm = np.random.default_rng(RANDOM_SEED).permutation(len(X))
    X, y = X[perm], y[perm]

    print(f"\n[DataLoader] Final: {X.shape}")
    overall_progress = _compute_overall_progress(source_stats)
    _print_progress_line("Dataset total", overall_progress["explored"], overall_progress["total"], "items")
    PROCESSED_EXERCISE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(proc_x, X); np.save(proc_y, y)
    return X, y, _class_weights(y), overall_progress


# It trains a single model architecture, handling checkpoint resuming and evaluation.
def train_one_model(model_name, X_tr, y_tr, X_val, y_val,
                    class_weights, session, epochs, input_size):
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    s             = session.model_state(model_name)
    initial_epoch = s.get("last_epoch", 0)
    best_ckpt     = session.best_ckpt_path(model_name)

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
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3,
            min_lr=1e-6, verbose=1),
        SessionCheckpoint(model_name, session, history_list),
    ]

    session.mark_in_progress(model_name, initial_epoch)
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
    best_epoch   = session.model_state(model_name)["best_epoch"]
    session.mark_done(model_name, best_val_acc, best_epoch, best_ckpt)

    _save_training_plot(model_name, history_list, session)
    _save_model_metrics(model_name, history_list, session)

    print(f"\n  ✓ {model_name} done — "
          f"best val_acc={best_val_acc:.4f} @ epoch {best_epoch}")
    return model


# This is the main entry point for training. It loads the data, splits it,
# and then trains each architecture sequentially.
def train_all_models(use_real_data=True, force_reprocess=False,
                     epochs=None, use_qevd_300k=True,
                     qevd_max=None, reset_session=False,
                     allow_synthetic_fallback=False,
                     qevd_batch_total=DEFAULT_QEVD_BATCH_TOTAL,
                     max_videos_per_class=None,
                     no_augment=False,
                     max_sequences_per_class=None):
    if not TF_AVAILABLE:
        print("ERROR: TensorFlow not installed.")
        return

    from sklearn.model_selection import train_test_split

    if reset_session:
        print("[Session] Deleting all checkpoints, plots, reports, and cached dataset...")
        for d in (CKPT_DIR, PLOTS_DIR, REPORTS_DIR):
            if d.exists():
                shutil.rmtree(d)
        for p in (PROCESSED_EXERCISE_DIR / "X.npy",
                  PROCESSED_EXERCISE_DIR / "y.npy"):
            p.unlink(missing_ok=True)
        QEVD_PROGRESS_FILE.unlink(missing_ok=True)

    session = SessionManager()
    session.summary()

    if use_real_data:
        X, y, cw, data_progress = load_dataset(
            force_reprocess=force_reprocess,
            use_qevd_300k=use_qevd_300k,
            qevd_max=qevd_max,
            allow_synthetic_fallback=allow_synthetic_fallback,
            qevd_batch_total=qevd_batch_total,
            augment_minority=not no_augment,
            max_videos_per_class=max_videos_per_class,
            max_sequences_per_class=max_sequences_per_class)
    else:
        print("[Training] Synthetic demo data")
        X, y = generate_demo_data(n=800)
        cw   = _class_weights(y)
        data_progress = {
            "explored": int(len(X)),
            "total": int(len(X)),
            "explored_pct": 100.0,
            "sources": [{"name": "synthetic", "explored": int(len(X)),
                         "total": int(len(X)), "unit": "samples"}],
        }

    session.state["dataset_shape"] = list(X.shape)
    session.state["data_progress"] = data_progress
    session.save()

    n_feat = X.shape[2]
    print(f"\n  Dataset: {X.shape}")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_tmp)

    split_file = CKPT_DIR / "test_split.npz"
    if not split_file.exists():
        np.savez(str(split_file), X_te=X_te, y_te=y_te)
    else:
        loaded  = np.load(str(split_file))
        X_te, y_te = loaded['X_te'], loaded['y_te']

    print(f"  Train:{len(X_tr)}  Val:{len(X_val)}  Test:{len(X_te)}")

    n_epochs = epochs or BILSTM_CONFIG.get('epochs', 50)

    for model_name in MODEL_BUILDERS:
        if session.is_done(model_name):
            print(f"\n[Session] {model_name} already complete — skipping")
            continue
        train_one_model(
            model_name=model_name,
            X_tr=X_tr, y_tr=y_tr,
            X_val=X_val, y_val=y_val,
            class_weights=cw,
            session=session,
            epochs=n_epochs,
            input_size=n_feat,
        )
        session.refresh_comparison()
        save_comparison_report(session, X_te, y_te)

    session.summary()
    print(f"\n[Done] Plots    → {PLOTS_DIR}")
    print(f"[Done] Reports  → {REPORTS_DIR}")
    print(f"[Done] Checkpts → {CKPT_DIR}")


# This manager is used in the live demo to load a trained model and make predictions.
class ExerciseClassifierManager:
    def __init__(self, model_path=None):
        self.model_path       = (str(CKPT_DIR / "BiLSTM_best.keras")
                                 if model_path is None else model_path)
        self.sequence_length  = BILSTM_CONFIG['sequence_length']
        self.classes          = EXERCISE_CLASSES
        self.keypoints_buffer = []
        self.model            = self._load_model()

    def _load_model(self):
        if not TF_AVAILABLE:
            return None
        for p in [Path(self.model_path),
                  MODELS_DIR / "exercise_classifier.keras"]:
            if p.exists():
                try:
                    m = keras.models.load_model(
                        str(p),
                        custom_objects={"AttentionLayer": AttentionLayer})
                    print(f"[Manager] Loaded {p.name}")
                    return m
                except Exception as e:
                    print(f"[Manager] Load error: {e}")
        print("[Manager] No trained model — untrained demo mode")
        return build_bilstm()

    def add_frame(self, kv):
        if kv is None or self.model is None:
            return None, 0.0
        self.keypoints_buffer.append(kv)
        if len(self.keypoints_buffer) > self.sequence_length:
            self.keypoints_buffer = self.keypoints_buffer[-self.sequence_length:]
        if len(self.keypoints_buffer) < self.sequence_length:
            return None, 0.0
        window = np.array(self.keypoints_buffer[-self.sequence_length:],
                          np.float32)
        return self.predict_exercise(window)

    def predict_exercise(self, keypoints_window):
        if self.model is None:
            return EXERCISE_CLASSES[0], 0.0
        x     = np.expand_dims(keypoints_window.astype(np.float32), 0)
        probs = self.model.predict(x, verbose=0)[0]
        idx   = int(np.argmax(probs))
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


# It checks that the Python version and MediaPipe installation are correct.
def _validate_runtime_for_real_training(args):
    if args.demo:
        return

    py_path = Path(sys.executable)
    if sys.version_info[:2] != (3, 12):
        print("ERROR: Python 3.12 is required for full training.")
        print(f"  Current: {py_path}")
        print(f"  Version: {sys.version.split()[0]}")
        print("  Use a Python 3.12 environment, for example:")
        print("  .\\.venv312\\Scripts\\python.exe "
              "src/computer_vision/exercise_classifier.py --reset --reprocess --qevd-max 0")
        sys.exit(1)

    try:
        import mediapipe as mp
    except ImportError:
        print("ERROR: mediapipe is not installed in this environment.")
        print("  Run: .\\scripts\\setup_training_env.ps1")
        sys.exit(1)

    if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'pose'):
        print("ERROR: MediaPipe legacy Pose API is unavailable in this environment.")
        print(f"  mediapipe version: {getattr(mp, '__version__', 'unknown')}")
        print("  Run: .\\scripts\\setup_training_env.ps1")
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FitCoach AI — 5-Architecture Exercise Classifier")
    parser.add_argument("--demo",          action="store_true",
                        help="Synthetic data, no datasets required")
    parser.add_argument("--reprocess",     action="store_true",
                        help="Ignore NPY cache, re-run MediaPipe")
    parser.add_argument("--reset",         action="store_true",
                        help="Delete all checkpoints and start fresh")
    parser.add_argument("--no-qevd-300k",  action="store_true",
                        help="Skip QEVD-300k (faster first run)")
    parser.add_argument("--qevd-max",      type=int, default=None,
                        help=f"Max QEVD-300k videos/class "
                            f"(default {MAX_QEVD_PER_CLASS}; use 0 for all)")
    parser.add_argument("--qevd-batch-total", type=int,
                        default=DEFAULT_QEVD_BATCH_TOTAL,
                        help=("QEVD-300k videos to add per run before training "
                              f"(default {DEFAULT_QEVD_BATCH_TOTAL}; use 0 for all remaining)"))
    parser.add_argument("--allow-synthetic-fallback", action="store_true",
                        help="Allow synthetic fallback when some classes have no real data")
    parser.add_argument("--epochs",        type=int, default=None,
                        help="Max epochs per model (default from config)")
    
    parser.add_argument("--max-videos-per-class", type=int, default=None,
                        help="Limit videos per class in Gym/Workout and QEVD-300k (fast training)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable minority class augmentation (keeps dataset smaller)")
    parser.add_argument("--max-sequences-per-class", type=int, default=None,
                        help="Hard limit on sequences per class after extraction")
    
    args = parser.parse_args()

    if not TF_AVAILABLE:
        print("ERROR: TensorFlow not installed.  Run: pip install tensorflow")
        sys.exit(1)

    _validate_runtime_for_real_training(args)

    train_all_models(
        use_real_data   = not args.demo,
        force_reprocess = args.reprocess,
        epochs          = args.epochs,
        use_qevd_300k   = not args.no_qevd_300k,
        qevd_max        = args.qevd_max,
        reset_session   = args.reset,
        allow_synthetic_fallback = args.allow_synthetic_fallback,
        qevd_batch_total = args.qevd_batch_total,
        max_videos_per_class = args.max_videos_per_class,
        no_augment = args.no_augment,
        max_sequences_per_class = args.max_sequences_per_class,
    )