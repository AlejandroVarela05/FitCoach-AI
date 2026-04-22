# shap_explainer.py


# This module applies explainable AI (XAI) techniques to the posture classifier.
# It compares four methods: SHAP Kernel, SHAP Gradient, LIME, and a custom
# permutation‑based explanation. The goal is to understand which joint angles
# most influence the model's decision and to measure the consistency and speed
# of each explanation method.
#
# PURPOSE:
#   - Load the trained MLP posture classifier and the test dataset.
#   - Generate local explanations for individual samples.
#   - Produce global feature importance using permutation importance.
#   - Visualise and compare the results to select the best method for real‑time
#     feedback.
#
# COURSE CONNECTION:
#   This work relates to "Advanced Machine Learning" (Unit I – Green, Explainable
#   and Safe AI). The teaching‑learning contract explicitly requires the use of
#   SHAP and LIME to explain model decisions. I also evaluate the methods in
#   terms of speed and consistency, as discussed in the lectures on XAI evaluation.
#
# DECISIONS:
#   - I use the MLP model trained on 12 joint angles because it is the best
#     performer and its input features are interpretable.
#   - I compare four methods to understand the trade‑off between accuracy
#     (SHAP Kernel), speed (SHAP Gradient), and model‑agnostic simplicity (LIME).
#   - I measure consistency via correlation between importance vectors and speed
#     via average execution time per sample.
#   - SHAP Gradient is chosen for real‑time use because it is fast and correlates
#     well with the more accurate Kernel SHAP.



import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED

# I use safe import blocks because SHAP and LIME are optional dependencies.
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[Explainability] TensorFlow not available")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[Explainability] SHAP not available")

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("[Explainability] LIME not available")

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier  # for comparison (not used directly here)

# The two posture classes.
POSTURE_CLASSES = ["incorrect", "correct"]
# The 12 joint angles used as features.
ANGLE_FEATURE_NAMES = [
    "Left Knee", "Right Knee", "Left Hip", "Right Hip", "Spine",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Neck", "Torso Lean", "Head Tilt"
]

# Paths to the trained model and the processed dataset.
MODEL_PATH = MODELS_DIR / "checkpoints" / "posture_mediapipe" / "mlp_posture_mediapipe_best.keras"
DATA_PATH = PROCESSED_DATA_DIR / "posture_classifier_mediapipe" / "X_posture_mediapipe.npy"
LABEL_PATH = PROCESSED_DATA_DIR / "posture_classifier_mediapipe" / "y_posture_mediapipe.npy"

# Output directory for all plots.
OUTPUT_DIR = MODELS_DIR / "plots" / "explainability_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# I load the trained Keras model and the test data. If the data is missing,
# I generate synthetic data so the script can still run for demonstration.
def load_model_and_data():
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not installed")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run posture_classifier_mediapipe.py --train first.")
    model = keras.models.load_model(MODEL_PATH)
    print(f"[Explainability] Loaded model from {MODEL_PATH}")

    if DATA_PATH.exists() and LABEL_PATH.exists():
        X = np.load(DATA_PATH)
        y = np.load(LABEL_PATH)
        print(f"[Explainability] Loaded data: X {X.shape}, y {y.shape}")
    else:
        # If the real data is not found, I create dummy data to allow the script to run.
        print("[Explainability] Data not found, generating synthetic background...")
        X = np.random.randn(500, 12) * 15 + [140,140,160,160,170,170,45,45,165,165,175,100]
        y = (X[:,0] < 90).astype(int)  # dummy label based on left knee angle
    return model, X, y

# I define a helper function to get the predicted probabilities for both classes.
def predict_proba(model, X):
    # I check this condition to make sure I only run this path when the state is valid.
    if isinstance(X, np.ndarray) and X.ndim == 1:
        X = X.reshape(1, -1)
    probs = model.predict(X, verbose=0)
    return probs  # shape (n_samples, 2)

# This class encapsulates the four explanation methods. It initialises the
# explainers once and provides a uniform interface to explain a single sample.
class ExplainabilityComparison:
    def __init__(self, model, background_X):
        self.model = model
        self.background_X = background_X[:100]  # use 100 samples as background for SHAP
        self.predict_fn = lambda x: predict_proba(model, x)
        self.shap_kernel = None
        self.shap_gradient = None
        self.lime_explainer = None

        if SHAP_AVAILABLE:
            print("[SHAP] Initialising KernelExplainer...")
            # KernelExplainer is model‑agnostic but slower.
            self.shap_kernel = shap.KernelExplainer(self.predict_fn, self.background_X)
            print("[SHAP] Initialising GradientExplainer...")
            # GradientExplainer is specific to neural networks and much faster.
            self.shap_gradient = shap.GradientExplainer(model, self.background_X)

        if LIME_AVAILABLE:
            print("[LIME] Initialising LimeTabularExplainer...")
            self.lime_explainer = LimeTabularExplainer(
                self.background_X,
                feature_names=ANGLE_FEATURE_NAMES,
                class_names=POSTURE_CLASSES,
                discretize_continuous=True,
                random_state=RANDOM_SEED
            )

    def explain_shap_kernel(self, x):
        if self.shap_kernel is None:
            return None
        start = time.time()
        shap_values = self.shap_kernel.shap_values(x.reshape(1, -1), nsamples=100)
        elapsed = time.time() - start
        # SHAP returns a list for each class. I take the absolute values for the "correct" class.
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                imp = np.abs(shap_values[1][0])  # class 1 (correct)
            else:
                imp = np.abs(shap_values[0][0])  # only one class output
        else:
            imp = np.abs(shap_values[0])
        return imp, elapsed

    def explain_shap_gradient(self, x):
        if self.shap_gradient is None:
            return None
        start = time.time()
        shap_values = self.shap_gradient.shap_values(x.reshape(1, -1))
        elapsed = time.time() - start
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                imp = np.abs(shap_values[1][0])
            else:
                imp = np.abs(shap_values[0][0])
        else:
            imp = np.abs(shap_values[0])
        return imp, elapsed

    def explain_lime(self, x):
        if self.lime_explainer is None:
            return None
        start = time.time()
        exp = self.lime_explainer.explain_instance(x, self.predict_fn, num_features=12, num_samples=500)
        elapsed = time.time() - start
        # LIME returns a list of (feature, weight). I convert this to an array of absolute weights.
        imp = np.zeros(12)
        for feat, weight in exp.as_list():
            idx = ANGLE_FEATURE_NAMES.index(feat)
            imp[idx] = abs(weight)
        return imp, elapsed

    def explain_permutation(self, x, baseline_pred=None):
        # This is a simple custom method: I perturb each feature and measure the change in prediction.
        start = time.time()
        x_orig = x.copy()
        prob_orig = self.predict_fn(x_orig.reshape(1, -1))[0, 1]  # probability of "correct"
        imp = np.zeros(12)
        for i in range(12):
            x_pert = x_orig.copy()
            # Add Gaussian noise to one feature at a time.
            x_pert[i] = x_pert[i] + np.random.normal(0, 5)
            prob_pert = self.predict_fn(x_pert.reshape(1, -1))[0, 1]
            imp[i] = abs(prob_orig - prob_pert)
        elapsed = time.time() - start
        return imp, elapsed

# I run all four methods on a few test samples and collect the importance vectors and timings.
def compare_methods_on_samples(model, X_test, y_test, num_samples=5):
    explainer = ExplainabilityComparison(model, X_test[:100])
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    results = defaultdict(list)  # method name -> list of (importance array, time, true label)

    for idx in indices:
        x = X_test[idx]
        true_label = y_test[idx]
        print(f"\n[Sample {idx}] True label: {POSTURE_CLASSES[true_label]}")

        for method_name, method in [
            ("SHAP Kernel", explainer.explain_shap_kernel),
            ("SHAP Gradient", explainer.explain_shap_gradient),
            ("LIME", explainer.explain_lime),
            ("Permutation", explainer.explain_permutation),
        ]:
            res = method(x)
            if res is not None:
                imp, t = res
                results[method_name].append((imp, t, true_label))
                print(f"  {method_name}: time={t:.3f}s")

    return results

# I create a grouped bar chart comparing the feature importance values from all methods.
def plot_feature_importance_comparison(results, sample_idx=0, save=True):
    if not results:
        return
    methods = list(results.keys())
    importance_dict = {}
    for method in methods:
        imp, _, _ = results[method][0]
        if imp is None:
            continue
        # If the importance array has two dimensions (features × classes), I take the "correct" class.
        if imp.ndim == 2:
            imp = imp[:, 1]   # second column corresponds to "correct" class
        imp = np.squeeze(imp)
        if imp.shape[0] != len(ANGLE_FEATURE_NAMES):
            print(f"[Warning] {method} importance shape {imp.shape} does not match features")
            continue
        importance_dict[method] = imp

    if not importance_dict:
        print("No valid importance data to plot.")
        return

    features = ANGLE_FEATURE_NAMES
    x = np.arange(len(features))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (method, imp) in enumerate(importance_dict.items()):
        ax.bar(x + i*width, imp, width, label=method, color=colors[i % len(colors)])

    ax.set_xticks(x + width * (len(importance_dict)-1)/2)
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Feature Importance (absolute)')
    ax.set_title('Comparison of Explainability Methods (Sample 0)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "feature_importance_comparison.png", dpi=120)
        print(f"[Plot] Saved to {OUTPUT_DIR / 'feature_importance_comparison.png'}")
    plt.show()

# I plot the average execution time for each method.
def plot_speed_comparison(results, save=True):
    methods = list(results.keys())
    times = []
    for method in methods:
        avg_time = np.mean([t for _, t, _ in results[method]])
        times.append(avg_time)
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(methods, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Average time (seconds)')
    ax.set_title('Explainability Method Speed Comparison')
    ax.grid(axis='y', alpha=0.3)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{t:.3f}s', ha='center', fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "speed_comparison.png", dpi=120)
        print(f"[Plot] Saved to {OUTPUT_DIR / 'speed_comparison.png'}")
    plt.show()

# I compute the correlation between the importance vectors produced by different methods.
# A high correlation indicates that the methods agree on which features are important.
def plot_consistency_heatmap(results, save=True):
    methods = list(results.keys())
    n_methods = len(methods)
    corr_matrix = np.ones((n_methods, n_methods))
    importance_vectors = {m: [] for m in methods}
    for method in methods:
        for imp, _, _ in results[method]:
            if imp is None:
                continue
            if imp.ndim == 2:
                imp = imp[:, 1]   # take class 1 (correct)
            imp = np.squeeze(imp)
            if imp.shape[0] == len(ANGLE_FEATURE_NAMES):
                importance_vectors[method].append(imp)
    # Compute pairwise Pearson correlation.
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                continue
            corrs = []
            for imp1, imp2 in zip(importance_vectors[m1], importance_vectors[m2]):
                if np.std(imp1) > 0 and np.std(imp2) > 0:
                    corr = np.corrcoef(imp1, imp2)[0,1]
                    if not np.isnan(corr):
                        corrs.append(corr)
            corr_matrix[i,j] = np.mean(corrs) if corrs else 0
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(n_methods))
    ax.set_yticks(range(n_methods))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticklabels(methods)
    ax.set_title('Correlation between Methods (Feature Importance)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "consistency_heatmap.png", dpi=120)
        print(f"[Plot] Saved to {OUTPUT_DIR / 'consistency_heatmap.png'}")
    plt.show()

# I compute global feature importance using permutation importance.
# This measures the drop in accuracy when a feature's values are randomly shuffled.
def global_permutation_importance(model, X_test, y_test):
    from sklearn.metrics import accuracy_score
    def scorer(model, X, y):
        pred = np.argmax(model.predict(X, verbose=0), axis=1)
        return accuracy_score(y, pred)
    result = permutation_importance(model, X_test, y_test, scoring=scorer,
                                    n_repeats=5, random_state=RANDOM_SEED, n_jobs=1)
    return result.importances_mean, result.importances_std

def plot_global_importance(importances, stds, save=True):
    fig, ax = plt.subplots(figsize=(10,6))
    y_pos = np.arange(len(ANGLE_FEATURE_NAMES))
    ax.barh(y_pos, importances, xerr=stds, align='center', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ANGLE_FEATURE_NAMES)
    ax.invert_yaxis()
    ax.set_xlabel('Permutation Importance (drop in accuracy)')
    ax.set_title('Global Feature Importance (Permutation)')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "global_permutation_importance.png", dpi=120)
        print(f"[Plot] Saved to {OUTPUT_DIR / 'global_permutation_importance.png'}")
    plt.show()

# The main function orchestrates the full comparison: loads data, runs the four
# methods, generates all plots, and prints a summary table.
def main():
    print("[Explainability] Starting comparison: SHAP Kernel vs SHAP Gradient vs LIME vs Permutation")

    model, X, y = load_model_and_data()
    # I use the first 200 samples for speed.
    X_test = X[:200]
    y_test = y[:200]

    print("\nRunning explainability methods on 3 test samples...")
    results = compare_methods_on_samples(model, X_test, y_test, num_samples=3)

    # Generate the comparison plots.
    plot_feature_importance_comparison(results, save=True)
    plot_speed_comparison(results, save=True)
    plot_consistency_heatmap(results, save=True)

    print("\nComputing global permutation importance...")
    imp_mean, imp_std = global_permutation_importance(model, X_test, y_test)
    plot_global_importance(imp_mean, imp_std, save=True)

    # Print a summary table.
    print("\nExplainability summary table:")
    methods = list(results.keys())
    avg_times = [np.mean([t for _, t, _ in results[m]]) for m in methods]
    print(f"{'Method':<20} {'Avg time (s)':<15} {'Top feature (sample 0)':<25}")
    for m, t in zip(methods, avg_times):
        imp, _, _ = results[m][0]
        if imp.ndim == 2:
            imp = imp[:, 1]
        imp = np.squeeze(imp)
        top_idx = np.argmax(imp)
        top_feat = ANGLE_FEATURE_NAMES[top_idx]
        print(f"{m:<20} {t:<15.4f} {top_feat}")

    print("\n[Done] All plots saved to", OUTPUT_DIR)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Explainability Comparison for Posture Classifier")
    parser.add_argument("--demo", action="store_true", help="Run full comparison")
    parser.add_argument("--sample", type=int, default=0, help="Explain a specific test sample index")
    args = parser.parse_args()

    if args.demo:
        main()
    elif args.sample is not None:
        # Explain a single sample and print the results.
        model, X, y = load_model_and_data()
        explainer = ExplainabilityComparison(model, X[:100])
        x_sample = X[args.sample]
        print(f"Sample {args.sample} (true class: {POSTURE_CLASSES[y[args.sample]]})")
        for name, method in [("SHAP Kernel", explainer.explain_shap_kernel),
                             ("SHAP Gradient", explainer.explain_shap_gradient),
                             ("LIME", explainer.explain_lime),
                             ("Permutation", explainer.explain_permutation)]:
            res = method(x_sample)
            if res is not None:
                imp, t = res
                top = ANGLE_FEATURE_NAMES[np.argmax(imp)]
                print(f"  {name}: top feature = {top}, time = {t:.3f}s")
    else:
        print("Usage: python shap_explainer.py --demo   # run full comparison")
        print("       python shap_explainer.py --sample 42   # explain one sample")