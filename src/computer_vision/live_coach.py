# live_coach.py


# This module provides a real‑time interactive coaching demo using a webcam.
# It combines exercise classification, posture evaluation, and repetition counting
# into a single live interface.
#
# PURPOSE:
#   - Load a trained exercise classifier (general or rehab) selected by the user.
#   - Run MediaPipe Pose Landmarker on the webcam feed to extract 3D keypoints.
#   - Classify the current exercise every 30 frames.
#   - Evaluate posture quality using either a rule‑based Knowledge Base (for general
#     exercises) or a trained MLP classifier (for rehab exercises).
#   - Count repetitions with one of three methods: PeakDetection, BackgroundSub, or LSTM.
#   - Display the results as overlays on the video feed.
#
# COURSE CONNECTION:
#   This script integrates all four modules of the FitCoach AI system:
#   - Computer Vision (pose extraction and exercise classification).
#   - Intelligent Systems (Knowledge Base for rule‑based posture feedback).
#   - Advanced Machine Learning (MLP posture classifier, LSTM counter).
#   - Speech & NLP (the foundation for the voice coach, though voice is handled
#     in a separate module).
#
# DECISIONS:
#   - I disable CUDA (`CUDA_VISIBLE_DEVICES='-1'`) to avoid GPU memory conflicts
#     when multiple TensorFlow models are loaded.
#   - I let the user choose the classifier type and model at runtime. The best
#     model (according to validation accuracy) is suggested first.
#   - For general exercises, posture is evaluated with the Knowledge Base because
#     the rules are transparent and easy to explain. For rehab, I use the MLP
#     classifier trained on ExeCheck.
#   - The LSTM counter requires a buffer of keypoints and predicts the count every
#     30 frames to balance speed and accuracy.



import sys
import os
import json
from pathlib import Path
from collections import deque

# I disable CUDA to prevent GPU memory issues when loading multiple TensorFlow models.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# I define the project root and add it to the path so I can import my custom modules.
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import cv2
import mediapipe as mp
import numpy as np

# I import the Knowledge Base for rule‑based posture feedback.
from src.intelligent_systems.knowledge_base import KnowledgeBase, GENERAL_EXERCISES

# I import the three repetition counters I implemented in the rep_counter module.
from src.computer_vision.rep_counter import (
    PeakDetectionCounter,
    BackgroundSubCounter,
    LSTMCounter,
)

# I define a helper function to load the MediaPipe Pose Landmarker (lite version).
def load_mediapipe_pose():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import urllib.request

    # I store the model in C:/temp to avoid path issues with spaces or Unicode.
    model_path = Path("C:/temp/pose_lite.task")

    if not model_path.exists() or model_path.stat().st_size == 0:
        print("[MediaPipe] Downloading Pose Landmarker (lite) to C:\\temp...")
        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(model_url, str(model_path))
        print(f"[MediaPipe] Model downloaded to {model_path} ({model_path.stat().st_size} bytes)")
    else:
        print(f"[MediaPipe] Using existing model at {model_path}")

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.PoseLandmarker.create_from_options(options), mp

# I define a function to compute the angle between three 3D points.
def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# I convert the 33 MediaPipe landmarks into a vector of 12 joint angles used for posture evaluation.
def landmarks_to_angles(landmarks):
    # Helper to get the (x,y,z) coordinates of a landmark by index.
    def p(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])
    
    left_knee = compute_angle(p(23), p(25), p(27))
    right_knee = compute_angle(p(24), p(26), p(28))
    left_hip = compute_angle(p(11), p(23), p(25))
    right_hip = compute_angle(p(12), p(24), p(26))
    left_shoulder = p(11)
    right_shoulder = p(12)
    left_hip_mp = p(23)
    right_hip_mp = p(24)
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip_mp + right_hip_mp) / 2
    spine_vec = shoulder_mid - hip_mid
    vertical = np.array([0, 1, 0])
    spine_angle = np.degrees(np.arccos(np.clip(np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec)+1e-8), -1, 1)))
    left_shoulder_angle = compute_angle(p(13), p(11), p(23))
    right_shoulder_angle = compute_angle(p(14), p(12), p(24))
    left_elbow = compute_angle(p(11), p(13), p(15))
    right_elbow = compute_angle(p(12), p(14), p(16))
    neck_vec = p(0) - shoulder_mid
    neck_angle = np.degrees(np.arccos(np.clip(np.dot(neck_vec, vertical) / (np.linalg.norm(neck_vec)+1e-8), -1, 1)))
    torso_lean = spine_angle
    head_tilt = neck_angle
    
    angles = np.array([left_knee, right_knee, left_hip, right_hip, spine_angle,
                       left_shoulder_angle, right_shoulder_angle, left_elbow, right_elbow,
                       neck_angle, torso_lean, head_tilt], dtype=np.float32)
    return angles

# I read the comparison summary JSON to get the model names sorted by best validation accuracy.
def load_model_ranking(classifier_type):
    if classifier_type == "general":
        report_dir = project_root / "models" / "reports" / "exercise"
        default_models = ["MLP", "CNN1D", "SimpleRNN", "LSTM", "BiLSTM"]
    else:
        report_dir = project_root / "models" / "reports" / "exercise_rehab"
        default_models = ["MLP", "CNN1D", "SimpleRNN", "LSTM", "BiLSTM"]
    summary_path = report_dir / "comparison_summary.json"
    if not summary_path.exists():
        print(f"  [Warning] No comparison report found at {summary_path}")
        return default_models
    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)
        sorted_models = sorted(data, key=lambda x: x.get("best_val_acc", 0), reverse=True)
        return [entry["model"] for entry in sorted_models]
    except Exception:
        return default_models

# This class wraps the trained MLP posture classifier for rehabilitation exercises.
class PostureClassifierManager:
    def __init__(self, model_path=None):
        self.model = None
        if model_path is None:
            model_path = project_root / "models" / "checkpoints" / "posture_mediapipe" / "mlp_posture_mediapipe_best.keras"
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        try:
            import tensorflow as tf
            from tensorflow import keras
            if self.model_path.exists():
                self.model = keras.models.load_model(self.model_path)
                print(f"[Posture] Loaded MLP model from {self.model_path}")
            else:
                print(f"[Posture] Model not found at {self.model_path}. Posture evaluation disabled.")
        except Exception as e:
            print(f"[Posture] Error loading model: {e}")
    
    def classify_angles(self, angles_vector):
        if self.model is None:
            return {'class': 'unknown', 'confidence': 0.0}
        angles = angles_vector.reshape(1, -1).astype(np.float32)
        probs = self.model.predict(angles, verbose=0)[0]  # [incorrect, correct]
        pred_class = "correct" if probs[1] > 0.5 else "incorrect"
        confidence = probs[1] if pred_class == "correct" else probs[0]
        return {'class': pred_class, 'confidence': float(confidence)}

# I draw the skeleton and keypoints on the video frame using OpenCV.
def draw_landmarks_on_frame(frame, landmarks, mp_module=None):
    # These are the connections I want to draw between landmarks.
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ]
    h, w = frame.shape[:2]
    # I draw lines for each connection.
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        start_lm = landmarks[start_idx]
        end_lm = landmarks[end_idx]
        start_point = (int(start_lm.x * w), int(start_lm.y * h))
        end_point = (int(end_lm.x * w), int(end_lm.y * h))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    # I draw circles for each landmark.
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

# The main function sets up the webcam, loads the models, and runs the interactive loop.
def main():
    parser = argparse.ArgumentParser(description="Live Coach: Exercise + Posture + Counting")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--exercise-classifier", type=str, choices=["general", "rehab"], default=None)
    parser.add_argument("--exercise-model", type=str, default=None)
    parser.add_argument("--posture", action="store_true", help="Enable posture evaluation")
    parser.add_argument("--count", action="store_true", help="Enable repetition counter")
    parser.add_argument("--counting-method", type=str, choices=["peak", "bgsub", "lstm"], default=None,
                        help="Counting method: peak (PeakDetection), bgsub (BackgroundSub), lstm (LSTM)")
    args = parser.parse_args()

    # If no classifier is specified via command line, I ask the user interactively.
    classifier_type = args.exercise_classifier
    if classifier_type is None:
        print("\n=== Live Coach - Exercise & Posture Classification ===")
        print("Choose exercise classifier:")
        print("  general  - 8 fitness exercises (push_up, squat, ...)")
        print("  rehab    - 10 rehabilitation exercises (arm_circle, ...)")
        classifier_type = input("Enter 'general' or 'rehab' [default: general]: ").strip().lower()
        if classifier_type not in ["general", "rehab"]:
            classifier_type = "general"
        print()

    # I get the list of models sorted by validation accuracy and let the user choose one.
    ranked_models = load_model_ranking(classifier_type)
    model_name = args.exercise_model
    if model_name is None:
        print(f"Available models (sorted by best validation accuracy, best first):")
        for i, m in enumerate(ranked_models, 1):
            suffix = " * BEST" if i == 1 else (" X WORST" if i == len(ranked_models) else "")
            print(f"  {i}. {m}{suffix}")
        choice = input("\nEnter model name or number [default: 1 for best]: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            model_name = ranked_models[idx] if 0 <= idx < len(ranked_models) else ranked_models[0]
        elif choice in ranked_models:
            model_name = choice
        else:
            model_name = ranked_models[0]
        print()

    # I ask whether to enable posture evaluation.
    enable_posture = args.posture
    if not enable_posture:
        posture_choice = input("Evaluate posture? (y/n) [default: n]: ").strip().lower()
        enable_posture = posture_choice in ['y', 'yes']
        print()

    # I ask whether to enable repetition counting.
    enable_counting = args.count
    if not enable_counting:
        count_choice = input("Count repetitions? (y/n) [default: n]: ").strip().lower()
        enable_counting = count_choice in ['y', 'yes']
        print()

    # I ask which counting method to use.
    counting_method = args.counting_method
    if enable_counting and counting_method is None:
        print("Choose counting method:")
        print("  1. PeakDetection (based on keypoints) - faster")
        print("  2. BackgroundSub (based on motion) - more robust in some scenarios")
        print("  3. LSTM (deep learning) - most accurate if model available")
        method_choice = input("Enter 1, 2 or 3 [default: 1]: ").strip()
        if method_choice == "2":
            counting_method = "bgsub"
        elif method_choice == "3":
            counting_method = "lstm"
        else:
            counting_method = "peak"
        print()

    # I load the appropriate exercise classifier manager.
    if classifier_type == "general":
        from src.computer_vision.exercise_classifier import ExerciseClassifierManager as ExerciseManager
        ckpt_dir = project_root / "models" / "checkpoints" / "exercise"
    else:
        from src.computer_vision.exercise_classifier_rehab import ExerciseClassifierManager as ExerciseManager
        ckpt_dir = project_root / "models" / "checkpoints" / "exercise_rehab"

    exercise_model_path = ckpt_dir / f"{model_name}_best.keras"
    if not exercise_model_path.exists():
        print(f"ERROR: Exercise model checkpoint not found: {exercise_model_path}")
        return

    exercise_manager = ExerciseManager(model_path=str(exercise_model_path))
    print(f"\nLoaded exercise classifier: {classifier_type} | Model: {model_name}")
    print(f"Exercise classes: {exercise_manager.classes}")

    # I initialise the posture evaluation component.
    posture_manager = None
    kbs = None
    if enable_posture:
        if classifier_type == "general":
            kbs = KnowledgeBase(include_rehab=False)
            print("[Posture] Using Knowledge Base (rule-based) for general exercises")
        else:
            posture_manager = PostureClassifierManager()
            if posture_manager.model is None:
                print("WARNING: Posture model not loaded. Continuing without posture evaluation.")
                enable_posture = False
            else:
                print("[Posture] Using MLP classifier for rehab exercises")

    # I initialise the repetition counter.
    rep_counter = None
    current_exercise = None
    counting_method_name = ""
    lstm_buffer = []          # I store keypoints here for the LSTM counter.
    lstm_pred_interval = 30   # I predict the count every 30 frames.
    frame_counter = 0

    if enable_counting:
        if counting_method == "peak":
            counting_method_name = "PeakDetection"
            print("[Counter] Using PeakDetection method (based on keypoints)")
        elif counting_method == "bgsub":
            counting_method_name = "BackgroundSub"
            print("[Counter] Using BackgroundSub method (based on motion)")
        elif counting_method == "lstm":
            counting_method_name = "LSTM"
            lstm_model_path = Path("models/weights/lstm_counter.h5")
            keras_model_path = Path("models/weights/lstm_counter.keras")
            if lstm_model_path.exists() or keras_model_path.exists():
                print("[Counter] Using LSTM Counter (deep learning)")
            else:
                print("[Counter] LSTM model not found. Falling back to PeakDetection.")
                counting_method = "peak"
                counting_method_name = "PeakDetection"

    # I load MediaPipe Pose.
    pose_landmarker, mp_module = load_mediapipe_pose()
    print("[MediaPipe] Pose Landmarker loaded (Tasks API)")

    # I open the webcam.
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        pose_landmarker.close()
        return

    print("\nWebcam running. Press 'q' to quit.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from webcam.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
            detection_result = pose_landmarker.detect(mp_image)

            exercise_label = None
            exercise_conf = 0.0
            posture_result = None
            angles = None
            rep_count = 0

            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                # I draw the skeleton on the frame.
                draw_landmarks_on_frame(frame, landmarks, mp_module)
                
                # I extract the 99 keypoint values for classification.
                kp = []
                for lm in landmarks:
                    kp.extend([lm.x, lm.y, lm.z])
                kp_array = np.array(kp, dtype=np.float32)
                
                # I get the exercise prediction.
                exercise_label, exercise_conf = exercise_manager.add_frame(kp)
                # I compute the 12 joint angles for posture evaluation.
                angles = landmarks_to_angles(landmarks)

                # I evaluate posture if enabled.
                if enable_posture:
                    if classifier_type == "general" and kbs is not None and exercise_label in GENERAL_EXERCISES:
                        analysis = kbs.analyze(exercise_label, angles)
                        is_correct = analysis["is_correct"]
                        feedback_list = analysis["feedback"]
                        posture_result = {
                            'class': 'correct' if is_correct else 'incorrect',
                            'confidence': 0.0,
                            'feedback': feedback_list
                        }
                    elif classifier_type == "rehab" and posture_manager is not None:
                        result_mlp = posture_manager.classify_angles(angles)
                        posture_result = {
                            'class': result_mlp['class'],
                            'confidence': result_mlp['confidence'],
                            'feedback': []
                        }

                # I count repetitions if enabled.
                if enable_counting and exercise_label is not None:
                    # When the exercise changes, I reset the counter.
                    if current_exercise != exercise_label:
                        if counting_method == "peak":
                            rep_counter = PeakDetectionCounter(exercise_label)
                        elif counting_method == "bgsub":
                            rep_counter = BackgroundSubCounter(exercise_label)
                        elif counting_method == "lstm":
                            rep_counter = LSTMCounter(model_path="models/weights/lstm_counter.keras")
                            lstm_buffer = []
                            frame_counter = 0
                        current_exercise = exercise_label

                    if counting_method == "peak":
                        rep_count = rep_counter.update(kp_array)
                    elif counting_method == "bgsub":
                        rep_count = rep_counter.update(frame)
                    elif counting_method == "lstm":
                        lstm_buffer.append(kp_array)
                        frame_counter += 1
                        if frame_counter % lstm_pred_interval == 0 and len(lstm_buffer) >= rep_counter.sequence_length:
                            seq = np.array(lstm_buffer[-rep_counter.sequence_length:])
                            rep_counter.buffer = deque(seq, maxlen=rep_counter.sequence_length)
                            rep_count = rep_counter.predict_count()
                        else:
                            rep_count = rep_counter.counter if rep_counter else 0

            # I build the text overlay lines.
            lines = []
            if exercise_label is not None:
                lines.append(f"Exercise: {exercise_label} ({exercise_conf:.2f})")
            else:
                buf_len = len(exercise_manager.keypoints_buffer)
                needed = exercise_manager.sequence_length
                lines.append(f"Collecting frames: {buf_len}/{needed}")

            if posture_result is not None:
                posture_class = posture_result['class']
                if classifier_type == "rehab":
                    confidence = posture_result['confidence']
                    lines.append(f"Posture: {posture_class} ({confidence:.2f})")
                else:
                    lines.append(f"Posture: {posture_class}")
                    if posture_result['feedback']:
                        lines.append(f"  -> {posture_result['feedback'][0][:50]}")
                        if len(posture_result['feedback']) > 1:
                            lines.append(f"  -> {posture_result['feedback'][1][:50]}")

            if enable_counting and rep_counter is not None:
                lines.append(f"Reps: {rep_count}  [{counting_method_name}]")

            # I draw the text lines on the frame.
            y_offset = 35
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (20, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Live Coach", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pose_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()