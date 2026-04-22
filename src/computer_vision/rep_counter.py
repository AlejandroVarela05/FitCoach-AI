# rep_counter.py


# This module is the heart of the repetition counting evaluation for FitCoach AI.
# I implemented seven different counting methods to compare classical signal
# processing techniques against a deep learning LSTM approach. The goal is to find
# the most accurate and robust way to count exercise repetitions from pose keypoints
# extracted by MediaPipe.
#
# I evaluate all methods on the RepCountA public benchmark dataset and report
# Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Off-By-One
# accuracy (OBO). I also generate visual plots of the raw signal and filtered
# signal to help understand why certain methods work better than others.
#
# COURSE CONNECTION:
# This module applies concepts from:
# - "Computer Vision" Unit II (motion analysis, optical flow, feature tracking).
# - "Computer Vision" Unit III (deep learning for vision, LSTM networks).
# - "Advanced Machine Learning" Unit III (heuristic methods, signal filtering).
#
# The HybridCounter I designed combines a Butterworth low-pass filter with peak
# detection, which is a perfect example of using domain knowledge (signal
# processing) to outperform a pure deep learning model when data is limited.



# I import all the libraries I need for video processing, keypoint extraction,
# signal filtering, evaluation metrics, and visualisation.
import cv2
import numpy as np
import argparse
import csv
import json
import pickle
from collections import deque
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib
matplotlib.use('Agg')  # I use the non-interactive backend because this script runs headlessly in CI.
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# I try to import TensorFlow for the LSTM counter. If it is not available,
# I set a flag to disable that counter and continue with the others.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[LSTMCounter] TensorFlow not installed — LSTMCounter will be disabled.")

# I define the list of exercises for both general fitness and rehabilitation.
# This helps the counters know which signal to extract for each exercise.
GENERAL_EXERCISES = [
    "push_up", "squat", "shoulder_press", "barbell_biceps_curl",
    "plank", "leg_raises", "lateral_raise", "deadlift"
]

REHAB_EXERCISES = [
    "arm_circle", "forward_lunge", "high_knee_raise", "hip_abduction",
    "leg_extension", "shoulder_abduction", "shoulder_external_rotation",
    "shoulder_flexion", "side_step_squat"
]

# SIGNAL_MAP tells each counter which feature to extract for a given exercise.
# For example, for a squat I use the knee angle, for a push-up I use the Y
# coordinate of the shoulder. The boolean indicates whether the signal should be
# inverted when searching for peaks.
SIGNAL_MAP = {
    'squat': ('knee_angle', True),
    'push_up': ('shoulder_y', False),
    'shoulder_press': ('wrist_y', False),
    'barbell_biceps_curl': ('wrist_y', False),
    'plank': ('spine_angle', True),
    'leg_raises': ('ankle_y', False),
    'lateral_raise': ('wrist_y', False),
    'deadlift': ('hip_y', False),
    'arm_circle': ('shoulder_angle', True),
    'forward_lunge': ('knee_angle', True),
    'high_knee_raise': ('hip_y', False),
    'hip_abduction': ('hip_angle', True),
    'leg_extension': ('knee_angle', True),
    'shoulder_abduction': ('shoulder_angle', True),
    'shoulder_external_rotation': ('shoulder_angle', True),
    'shoulder_flexion': ('shoulder_angle', True),
    'side_step_squat': ('knee_angle', True),
}

# ------------------------------------------------------------------------------
# MEDIAPIPE POSE INITIALISATION
# ------------------------------------------------------------------------------
# I define a helper function to load the MediaPipe Pose Landmarker model.
# I use the lite version because it is faster and good enough for our purposes.
# I store the model in C:/temp to avoid path issues with spaces or Unicode.
def load_mediapipe_pose(use_gpu=True):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import urllib.request

    model_path = Path("C:/temp/pose_lite.task")

    if not model_path.exists() or model_path.stat().st_size == 0:
        print("[MediaPipe] Model not found locally — downloading pose_landmarker_lite...")
        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(model_url, str(model_path))
        print(f"[MediaPipe] Model downloaded to {model_path} ({model_path.stat().st_size} bytes)")
    else:
        print(f"[MediaPipe] Using cached model at {model_path}")

    # I force CPU usage because GPU delegate caused problems on my Windows machine.
    delegate = python.BaseOptions.Delegate.CPU
    print("[MediaPipe] Using CPU delegate (GPU not supported on this platform).")

    base_options = python.BaseOptions(model_asset_path=str(model_path), delegate=delegate)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.PoseLandmarker.create_from_options(options), mp

# ------------------------------------------------------------------------------
# KALMAN FILTER FOR SMOOTHING
# ------------------------------------------------------------------------------
# I implement a simple Kalman filter to smooth the keypoint coordinates (x, y).
# This reduces jitter from MediaPipe and makes peak detection more stable.
class KalmanFilter:
    def __init__(self):
        # I initialise a 4-state Kalman filter (x, y, vx, vy).
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False

    def update(self, x, y):
        # The first time, I set the initial state. After that, I correct and predict.
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
        self.kf.correct(np.array([[x], [y]], np.float32))
        prediction = self.kf.predict()
        return prediction[0, 0], prediction[1, 0]

# ------------------------------------------------------------------------------
# BUTTERWORTH LOW-PASS FILTER
# ------------------------------------------------------------------------------
# I use a 4th-order Butterworth filter to remove high-frequency noise from the
# extracted signal. The cutoff frequency is 2.5 Hz because most exercises are
# performed at a rate slower than 2.5 repetitions per second (150 reps/min).
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)  # filtfilt applies the filter forward and backward to avoid phase shift.
    return y

# ------------------------------------------------------------------------------
# SIGNAL EXTRACTION FROM KEYPOINTS
# ------------------------------------------------------------------------------
# Given a 99-dimensional keypoint vector (33 landmarks * 3 coordinates) and a
# signal type, this function returns a scalar value (e.g., knee angle or Y coord).
def extract_signal(kp99, sig_type):
    kp = kp99.reshape(-1, 3)  # Reshape to (33, 3) for easier indexing.
    if sig_type == 'knee_angle':
        hip = kp[23]; knee = kp[25]; ankle = kp[27]
        v1 = hip - knee; v2 = ankle - knee
        cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))
    elif sig_type == 'shoulder_angle':
        elbow = kp[13]; shoulder = kp[11]; hip = kp[23]
        v1 = elbow - shoulder; v2 = hip - shoulder
        cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))
    elif sig_type == 'hip_angle':
        shoulder = kp[11]; hip = kp[23]; knee = kp[25]
        v1 = shoulder - hip; v2 = knee - hip
        cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))
    elif sig_type == 'spine_angle':
        chest = kp[2]; pelvis = kp[0]
        vertical = np.array([0,1,0])
        spine_vec = chest - pelvis
        cos = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec)+1e-8)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))
    elif sig_type == 'shoulder_y':
        return kp[11][1]
    elif sig_type == 'wrist_y':
        return kp[15][1]
    elif sig_type == 'hip_y':
        return kp[23][1]
    elif sig_type == 'ankle_y':
        return kp[27][1]
    else:
        return 0.0

# ==============================================================================
# REPETITION COUNTER IMPLEMENTATIONS
# ==============================================================================
# I implemented seven different counting methods to understand the trade-offs
# between simplicity, speed, and accuracy. Each method is a separate class with
# update() and reset() methods.

# ------------------------------------------------------------------------------
# 1. Optical Flow Counter
#    Uses dense optical flow to detect motion. It counts a repetition when the
#    average motion magnitude goes above a threshold and then drops back down.
#    It does not require knowing the exercise type, but is sensitive to camera
#    shake and background movement.
# ------------------------------------------------------------------------------
class OpticalFlowCounter:
    def __init__(self, exercise):
        self.prev_gray = None
        self.flow_mag = deque(maxlen=60)
        self.counter = 0
        self.phase = "idle"
        self.adaptative_threshold = 1.0
    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))
            self.flow_mag.append(mag)
            if len(self.flow_mag) > 30:
                self.adaptative_threshold = np.mean(self.flow_mag) * 0.5
            if len(self.flow_mag) > 10:
                if self.phase == "idle" and mag > self.adaptative_threshold:
                    self.phase = "moving"
                elif self.phase == "moving" and mag < self.adaptative_threshold * 0.5:
                    self.counter += 1
                    self.phase = "idle"
        self.prev_gray = gray
        return self.counter
    def reset(self):
        self.counter = 0
        self.prev_gray = None
        self.flow_mag.clear()
        self.phase = "idle"

# ------------------------------------------------------------------------------
# 2. Angle Threshold Counter
#    Counts a repetition each time the knee angle drops below a dynamically
#    computed threshold (min angle in recent buffer + 15 degrees).
# ------------------------------------------------------------------------------
class AngleThresholdCounter:
    def __init__(self, threshold=None):
        self.threshold = threshold if threshold is not None else 90
        self.was_below = False
        self.counter = 0
        self.angle_buffer = deque(maxlen=30)
    def update(self, knee_angle):
        self.angle_buffer.append(knee_angle)
        if len(self.angle_buffer) > 10:
            dynamic_threshold = np.min(self.angle_buffer) + 15
            is_below = knee_angle < dynamic_threshold
            if not self.was_below and is_below:
                self.counter += 1
            self.was_below = is_below
        return self.counter
    def reset(self):
        self.counter = 0
        self.was_below = False
        self.angle_buffer.clear()

# ------------------------------------------------------------------------------
# 3. Peak Detection Counter
#    Detects peaks in the Y-coordinate of a chosen joint (e.g., hip for squats).
#    It uses a simple state machine (low/high) and a lockout period to avoid
#    double counting.
# ------------------------------------------------------------------------------
class PeakDetectionCounter:
    def __init__(self, exercise):
        self.joint_idx = {'squat': 23, 'push_up': 11}.get(exercise, 23)
        self.buffer = deque(maxlen=150)          # history of Y values
        self.counter = 0
        self.phase = 'low'                       # 'low' or 'high'
        self.lockout_counter = 0                 # remaining lockout frames
        self.lockout_period = 20                 # frames to wait after a peak before counting again

    def update(self, kp99):
        y = kp99[self.joint_idx * 3 + 1]
        self.buffer.append(y)

        if self.lockout_counter > 0:
            self.lockout_counter -= 1

        if len(self.buffer) < 30:
            return self.counter

        arr = np.array(self.buffer)
        min_val = np.min(arr)
        max_val = np.max(arr)
        amplitude = max_val - min_val

        # I ignore very small movements (amplitude < 0.03) to avoid false counts.
        if amplitude < 0.03:
            return self.counter

        low_thresh = min_val + 0.3 * amplitude
        high_thresh = min_val + 0.7 * amplitude

        current_val = arr[-1]

        if self.phase == 'low':
            if current_val > high_thresh and self.lockout_counter == 0:
                self.phase = 'high'
        elif self.phase == 'high':
            if current_val < low_thresh:
                self.counter += 1
                self.phase = 'low'
                self.lockout_counter = self.lockout_period

        return self.counter

    def reset(self):
        self.counter = 0
        self.buffer.clear()
        self.phase = 'low'
        self.lockout_counter = 0

# ------------------------------------------------------------------------------
# 4. Feature Tracking Counter
#    Uses ORB feature detection and homography to track the person's bounding
#    box. It counts a repetition when the vertical position of the box changes
#    significantly.
# ------------------------------------------------------------------------------
class FeatureTrackingCounter:
    def __init__(self, exercise):
        self.exercise = exercise
        self.orb = cv2.ORB_create(nfeatures=100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_desc = None
        self.counter = 0
        self.phase = "idle"
        self.tracked_points = None
        self.bbox_history = deque(maxlen=10)
    def update(self, frame, kp99=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_desc is None:
            kp, desc = self.orb.detectAndCompute(gray, None)
            if desc is not None:
                self.prev_kp = kp
                self.prev_desc = desc
            return self.counter
        kp, desc = self.orb.detectAndCompute(gray, None)
        if desc is not None and self.prev_desc is not None:
            matches = self.bf.match(self.prev_desc, desc)
            matches = sorted(matches, key=lambda x: x.distance)[:50]
            if len(matches) > 10:
                src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w = frame.shape[:2]
                    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts, M)
                    self.tracked_points = dst
                    center = np.mean(dst, axis=0)[0]
                    self.bbox_history.append(center[1])
                    if len(self.bbox_history) > 10:
                        smoothed_y = np.mean(self.bbox_history)
                        if self.phase == "idle":
                            self.phase = "moving"
                            self.start_y = smoothed_y
                        elif self.phase == "moving":
                            if abs(smoothed_y - self.start_y) > 50:
                                self.counter += 1
                                self.phase = "idle"
        self.prev_kp = kp
        self.prev_desc = desc
        return self.counter
    def reset(self):
        self.counter = 0
        self.prev_kp = None
        self.prev_desc = None
        self.phase = "idle"
        self.tracked_points = None
        self.bbox_history.clear()

# ------------------------------------------------------------------------------
# 5. Background Subtraction Counter
#    Uses MOG2 background subtractor to detect motion. It counts a repetition
#    when the percentage of moving pixels goes above a threshold and then drops.
# ------------------------------------------------------------------------------
class BackgroundSubCounter:
    def __init__(self, exercise):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.motion_history = deque(maxlen=90)      # ~3 seconds at 30 fps
        self.counter = 0
        self.phase = "idle"
        self.motion_threshold = 0.01   # initial threshold before adaptation
        self.hysteresis_low = 0.005
        self.frame_count = 0

    def update(self, frame):
        self.frame_count += 1
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel)
        moving = np.count_nonzero(fgmask) / (frame.shape[0] * frame.shape[1])
        self.motion_history.append(moving)

        if len(self.motion_history) >= 30:
            mean_motion = np.mean(self.motion_history)
            std_motion = np.std(self.motion_history)
            high_thresh = mean_motion + 1.5 * std_motion
            low_thresh  = mean_motion + 0.5 * std_motion
            high_thresh = max(high_thresh, 0.008)
            low_thresh  = max(low_thresh,  0.003)
        else:
            high_thresh = self.motion_threshold
            low_thresh  = self.hysteresis_low

        if self.phase == "idle":
            if moving > high_thresh:
                self.phase = "moving"
        elif self.phase == "moving":
            if moving < low_thresh:
                self.counter += 1
                self.phase = "idle"

        return self.counter

    def reset(self):
        self.counter = 0
        self.phase = "idle"
        self.motion_history.clear()
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

# ------------------------------------------------------------------------------
# 6. Peak-Valley Counter
#    Similar to PeakDetectionCounter, but uses a Kalman filter to smooth the
#    signal first. This makes it more robust to jitter.
# ------------------------------------------------------------------------------
class PeakValleyCounter:
    def __init__(self, exercise):
        self.joint_idx = {'squat':23, 'push_up':11}.get(exercise, 23)
        self.y_positions = deque(maxlen=120)
        self.counter = 0
        self.phase = "idle"
        self.last_peak = -30
        self.kalman = KalmanFilter()
    def update(self, kp99):
        raw_y = kp99[self.joint_idx*3+1]
        filtered_y, _ = self.kalman.update(raw_y, raw_y)
        self.y_positions.append(filtered_y)
        if len(self.y_positions) < 20:
            return 0
        smoothed = np.mean(list(self.y_positions)[-5:])
        recent_std = np.std(list(self.y_positions)[-30:])
        threshold = max(0.02, recent_std * 0.5)
        if self.phase == "idle":
            self.local_max = smoothed
            self.phase = "descending"
        elif self.phase == "descending":
            if smoothed < self.local_max - threshold:
                self.phase = "ascending"
                self.local_min = smoothed
        elif self.phase == "ascending":
            if smoothed > self.local_min + threshold:
                if len(self.y_positions) - self.last_peak > 25:
                    self.counter += 1
                    self.last_peak = len(self.y_positions)
                self.phase = "descending"
                self.local_max = smoothed
        return self.counter
    def reset(self):
        self.counter = 0
        self.phase = "idle"
        self.y_positions.clear()
        self.last_peak = -30

# ------------------------------------------------------------------------------
# 7. HybridCounter (my custom method)
#    This is the method I designed for the final system. It first extracts the
#    appropriate signal (e.g., knee angle for squats), applies a Kalman filter
#    for real-time smoothing, and then at the end of the video applies a
#    Butterworth low-pass filter to the entire signal. Peaks and valleys are
#    detected on the filtered signal. The combination of Kalman + Butterworth
#    makes it very robust to MediaPipe jitter.
# ------------------------------------------------------------------------------
class HybridCounter:
    def __init__(self, exercise, fps=30):
        self.sig_type, _ = SIGNAL_MAP.get(exercise, ('hip_y', False))
        self.buffer = deque(maxlen=300)
        self.kalman = KalmanFilter()
        self.counter = 0
        self.last_peak_idx = -30
        self.frame = 0
        self.fps = fps
        self.exercise = exercise
    def update(self, kp99):
        self.frame += 1
        val = extract_signal(kp99, self.sig_type)
        filtered_val, _ = self.kalman.update(val, val)
        self.buffer.append(filtered_val)
        return self.counter

    def final_count(self):
        # I only count if I have at least 20 frames of data.
        if len(self.buffer) < 20:
            return 0
        signal = np.array(self.buffer)
        cutoff = 2.5  # Hz
        filtered = butter_lowpass_filter(signal, cutoff, self.fps)
        # I normalise the filtered signal to zero mean and unit variance.
        norm = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
        # I adjust peak detection parameters based on the exercise.
        if self.exercise in ['squat', 'push_up']:
            height = 0.3
            distance = int(self.fps * 0.8)
        else:
            height = 0.4
            distance = int(self.fps * 1.0)
        peaks,   _ = find_peaks( norm, height=height, distance=distance)
        valleys, _ = find_peaks(-norm, height=height, distance=distance)
        # I merge peaks and valleys and filter out extrema that are too close.
        all_extrema = sorted(list(peaks) + list(valleys))
        min_dist = int(self.fps * 0.5)
        filtered_extrema = []
        for idx in all_extrema:
            if not filtered_extrema or (idx - filtered_extrema[-1] >= min_dist):
                filtered_extrema.append(idx)
        # Each repetition consists of one peak and one valley, so I divide by 2.
        self.counter = len(filtered_extrema) // 2
        return self.counter

    def reset(self):
        self.counter = 0
        self.buffer.clear()
        self.frame = 0

# ------------------------------------------------------------------------------
# 8. LSTM Counter (deep learning baseline)
#    This counter uses a bidirectional LSTM trained on keypoint sequences to
#    predict the repetition count directly. It requires a trained model.
# ------------------------------------------------------------------------------
class LSTMCounter:
    def __init__(self, model_path=None, sequence_length=90, feature_dim=99):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.buffer = deque(maxlen=sequence_length)
        self.counter = 0
        if TF_AVAILABLE:
            if model_path and Path(model_path).exists():
                self.model = self._load_compatible_model(model_path)
            else:
                self.model = self._build_model()
        else:
            print("[LSTMCounter] TensorFlow not available — counter disabled.")

    def _load_compatible_model(self, model_path):
        model_path = str(model_path)
        # I try several loading options because .h5 and .keras formats can be tricky.
        load_attempts = [
            {"compile": False},
            {"compile": False, "safe_mode": False},
            {
                "compile": False,
                "custom_objects": {
                    "mse": tf.keras.losses.MeanSquaredError(),
                    "mae": tf.keras.metrics.MeanAbsoluteError(),
                },
            },
        ]

        last_error = None
        for kwargs in load_attempts:
            try:
                return load_model(model_path, **kwargs)
            except TypeError:
                try:
                    filtered = {k: v for k, v in kwargs.items() if k in {"compile", "custom_objects"}}
                    return load_model(model_path, **filtered)
                except Exception as e:
                    last_error = e
            except Exception as e:
                last_error = e

        print(f"[LSTMCounter] Could not load trained model '{model_path}': {last_error}")
        print("[LSTMCounter] Falling back to an untrained architecture. Re-export weights in .keras format.")
        return self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.sequence_length, self.feature_dim)),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def update(self, kp99):
        self.buffer.append(kp99)
        return self.counter  # the prediction is made at the end

    def predict_count(self):
        if self.model is None or len(self.buffer) < self.sequence_length:
            return 0
        X = np.array([list(self.buffer)])  # (1, seq_len, features)
        pred = self.model.predict(X, verbose=0)[0][0]
        self.counter = max(0, int(round(pred)))
        return self.counter

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, save_path='models/weights/lstm_counter.h5'):
        if not TF_AVAILABLE:
            return
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(save_path, save_best_only=True)
        ]
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
        self.model.save(save_path)
        print(f"[LSTMCounter] Model saved to {save_path}")

        keras_copy_path = save_path.with_suffix('.keras')
        try:
            self.model.save(keras_copy_path)
            print(f"[LSTMCounter] .keras copy saved to {keras_copy_path}")
        except Exception as e:
            print(f"[LSTMCounter] Could not save .keras copy: {e}")

    def reset(self):
        self.buffer.clear()
        self.counter = 0

# ------------------------------------------------------------------------------
# DATA PREPARATION FOR LSTM
# ------------------------------------------------------------------------------
# I read the RepCountA CSV files, extract keypoints with MediaPipe, and create
# training sequences for the LSTM counter.
def prepare_lstm_data(csv_file, video_dir, sequence_length=90, max_videos=None, use_gpu=True, resize_width=640):
    rows = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            parsed = None
            try:
                video_name = row[0].strip()
                start = int(float(row[1]))
                end = int(float(row[2]))
                count = int(float(row[3]))
                if video_name:
                    parsed = (video_name, start, end, count)
            except (ValueError, TypeError):
                pass

            if parsed is None:
                try:
                    video_name = row[2].strip()
                    count_raw = row[3].strip() if len(row) > 3 else ""
                    if not video_name or not count_raw:
                        continue
                    count = int(float(count_raw))
                    l_columns = row[4:]
                    starts_ends = []
                    for i in range(0, len(l_columns), 2):
                        if i + 1 < len(l_columns) and l_columns[i].strip() and l_columns[i+1].strip():
                            try:
                                start_val = int(float(l_columns[i]))
                                end_val = int(float(l_columns[i+1]))
                                starts_ends.append((start_val, end_val))
                            except ValueError:
                                pass
                    if starts_ends:
                        start = starts_ends[0][0]
                        end = starts_ends[-1][1]
                    else:
                        start = 0
                        end = None
                    parsed = (video_name, start, end, count)
                except (ValueError, TypeError, IndexError):
                    continue

            if parsed is not None:
                rows.append(parsed)

    if max_videos:
        rows = rows[:max_videos]

    pose_landmarker, mp_module = load_mediapipe_pose(use_gpu=use_gpu)

    X, y = [], []
    for video_name, start, end, count in tqdm(rows, desc="Preparing LSTM data"):
        video_path = video_dir / video_name
        if not video_path.exists():
            found = list(video_dir.rglob(video_name))
            if found:
                video_path = found[0]
            else:
                continue
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames_kp = []
        end_frame = end
        if end_frame is None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            end_frame = total_frames
        for _ in range(start, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            if resize_width:
                h, w = frame.shape[:2]
                new_h = int(h * resize_width / w)
                frame = cv2.resize(frame, (resize_width, new_h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
            detection_result = pose_landmarker.detect(mp_image)
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                kp = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
            else:
                kp = [0.0]*99
            frames_kp.append(np.array(kp, dtype=np.float32))
        cap.release()

        if len(frames_kp) >= sequence_length:
            for i in range(0, len(frames_kp) - sequence_length + 1, sequence_length // 2):
                seq = frames_kp[i:i + sequence_length]
                X.append(seq)
                y.append(count)
        else:
            seq = frames_kp + [np.zeros(99, dtype=np.float32)] * (sequence_length - len(frames_kp))
            X.append(seq)
            y.append(count)

    pose_landmarker.close()
    return np.array(X), np.array(y)

# ------------------------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ------------------------------------------------------------------------------
# This function runs all counting methods on the specified split of RepCountA,
# computes metrics, and saves the results.
def evaluate_on_repcounta(split='test', num_videos=None, train_lstm=False, save_plots=True,
                          use_gpu=True, resize_width=640, skip_slow_methods=False):
    project_root = Path(__file__).resolve().parent.parent.parent
    annotation_candidates = list(project_root.rglob("annotation"))
    base_path = None
    for ann_dir in annotation_candidates:
        if (ann_dir / f"{split}.csv").exists():
            base_path = ann_dir.parent
            break
    if base_path is None:
        print(f"[RepCounter] ERROR: 'annotation' folder with {split}.csv not found under {project_root}")
        return

    csv_file  = base_path / "annotation" / f"{split}.csv"
    video_dir = base_path / "video" / split

    print(f"[RepCounter] Annotations CSV : {csv_file}")
    print(f"[RepCounter] Video directory : {video_dir}")

    if not csv_file.exists() or not video_dir.exists():
        print("[RepCounter] ERROR: CSV or video directory does not exist.")
        return

    rows = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row
        for row in reader:
            if len(row) < 4:
                continue
            exercise_class = row[1].strip()
            video_name = row[2].strip()
            gt_raw = row[3].strip() if len(row) > 3 else ""
            if not video_name or not gt_raw:
                continue
            try:
                gt_count = int(float(gt_raw))
            except (ValueError, TypeError):
                continue
            l_columns = row[4:]
            starts_ends = []
            for i in range(0, len(l_columns), 2):
                if i+1 < len(l_columns) and l_columns[i].strip() and l_columns[i+1].strip():
                    try:
                        start_val = int(float(l_columns[i]))
                        end_val = int(float(l_columns[i+1]))
                        starts_ends.append((start_val, end_val))
                    except ValueError:
                        pass
            if not starts_ends:
                print(f"[RepCounter] Warning: {video_name} has no L annotations — using full video.")
                start = 0
                end = None
            else:
                start = starts_ends[0][0]
                end   = starts_ends[-1][1]
            rows.append((video_name, start, end, gt_count, exercise_class))

    if num_videos:
        rows = rows[:num_videos]

    print(f"[RepCounter] Evaluating {len(rows)} video segments...")

    pose_landmarker, mp_module = load_mediapipe_pose(use_gpu=use_gpu)

    counters = {
        "OpticalFlow": OpticalFlowCounter,
        "AngleThreshold": AngleThresholdCounter,
        "PeakDetection": PeakDetectionCounter,
        "PeakValley": PeakValleyCounter,
        "HybridCounter": HybridCounter,
    }
    if not skip_slow_methods:
        counters.update({
            "FeatureTracking": FeatureTrackingCounter,
            "BackgroundSub": BackgroundSubCounter,
        })
    if TF_AVAILABLE:
        counters["LSTMCounter"] = LSTMCounter

    if train_lstm and TF_AVAILABLE:
        print("I train the LSTM model with training data...")
        train_csv = base_path / "annotation" / "train.csv"
        train_video_dir = base_path / "video" / "train"
        X_train, y_train = prepare_lstm_data(train_csv, train_video_dir, sequence_length=90, max_videos=200,
                                             use_gpu=use_gpu, resize_width=resize_width)
        val_csv = base_path / "annotation" / "valid.csv"
        val_video_dir = base_path / "video" / "valid"
        X_val, y_val = prepare_lstm_data(val_csv, val_video_dir, sequence_length=90, max_videos=50,
                                         use_gpu=use_gpu, resize_width=resize_width)
        temp_lstm = LSTMCounter()
        temp_lstm.train(X_train, y_train, X_val, y_val)

    results = {name: {'errors': [], 'preds': [], 'gts': []} for name in counters}
    class_results = {}

    for video_name, start, end, gt_count, ex_class in tqdm(rows, desc=f"Processing {split}"):
        video_path = video_dir / video_name
        if not video_path.exists():
            found = list(video_dir.rglob(video_name))
            if found:
                video_path = found[0]
            else:
                continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        frame_count = end if end is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(start, frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if resize_width:
                h, w = frame.shape[:2]
                new_h = int(h * resize_width / w)
                frame = cv2.resize(frame, (resize_width, new_h))
            frames.append(frame)
        cap.release()
        if len(frames) < 10:
            continue

        all_kps = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
            detection_result = pose_landmarker.detect(mp_image)
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                kp = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
            else:
                kp = [0.0]*99
            all_kps.append(np.array(kp, dtype=np.float32))

        for method_name, CounterClass in counters.items():
            if method_name == "AngleThreshold":
                counter = CounterClass()
                for kp in all_kps:
                    angle = extract_signal(kp, 'knee_angle')
                    counter.update(angle)
                pred = counter.counter
            elif method_name in ("OpticalFlow", "BackgroundSub", "FeatureTracking"):
                counter = CounterClass(exercise=ex_class)
                for frame in frames:
                    counter.update(frame)
                pred = counter.counter
            elif method_name == "HybridCounter":
                counter = CounterClass(exercise=ex_class, fps=30)
                for kp in all_kps:
                    counter.update(kp)
                pred = counter.final_count()
            elif method_name == "LSTMCounter":
                counter = CounterClass()
                for kp in all_kps:
                    counter.update(kp)
                pred = counter.predict_count()
            else:
                counter = CounterClass(exercise=ex_class)
                for kp in all_kps:
                    counter.update(kp)
                pred = counter.counter

            error = abs(pred - gt_count)
            results[method_name]['errors'].append(error)
            results[method_name]['preds'].append(pred)
            results[method_name]['gts'].append(gt_count)

            if ex_class not in class_results:
                class_results[ex_class] = {m: {'errors': [], 'preds': [], 'gts': []} for m in counters}
            class_results[ex_class][method_name]['errors'].append(error)
            class_results[ex_class][method_name]['preds'].append(pred)
            class_results[ex_class][method_name]['gts'].append(gt_count)

    pose_landmarker.close()

    print(f"RESULTS IN RepCountA - Split: {split.upper()}")
    print(f"{'Method':<18} {'MAE':<8} {'RMSE':<8} {'OBO (%)':<10} {'Pearson r':<10} {'Samples':<8}")
    print("-"*70)
    metrics_summary = {}
    for method_name, data in results.items():
        if not data['errors']:
            continue
        mae = np.mean(data['errors'])
        rmse = np.sqrt(mean_squared_error(data['gts'], data['preds']))
        obo = np.mean([1 if e <= 1 else 0 for e in data['errors']]) * 100
        corr = np.corrcoef(data['preds'], data['gts'])[0,1] if len(data['preds']) > 1 else 0
        print(f"{method_name:<18} {mae:<8.2f} {rmse:<8.2f} {obo:<9.1f}% {corr:<10.3f} {len(data['errors']):<8}")
        metrics_summary[method_name] = {'MAE': mae, 'RMSE': rmse, 'OBO': obo, 'Pearson': corr}

    print("\n--- Results by exercise class (Top 5) ---")
    class_counts = {cls: len(class_results[cls][list(counters.keys())[0]]['errors']) for cls in class_results}
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for ex_class, _ in top_classes:
        print(f"\nClass: {ex_class} (n={class_counts[ex_class]})")
        for method_name in counters:
            if method_name not in class_results[ex_class]:
                continue
            errs = class_results[ex_class][method_name]['errors']
            if not errs:
                continue
            mae = np.mean(errs)
            obo = np.mean([1 if e <= 1 else 0 for e in errs]) * 100
            print(f"  {method_name:<16} MAE={mae:.2f}, OBO={obo:.1f}%")

    report_path = Path("models/reports") / f"repcounta_{split}_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nMetrics saved to {report_path}")

    if save_plots:
        plot_signal_analysis(rows[:5], video_dir, counters, save_dir="models/plots/signal_analysis",
                             use_gpu=use_gpu, resize_width=resize_width)

    return results, class_results

# ------------------------------------------------------------------------------
# SIGNAL ANALYSIS PLOTS
# ------------------------------------------------------------------------------
# I generate figures that show the raw signal, the filtered signal, and the
# detected peaks and valleys. This helps me understand why a method succeeds or
# fails on a particular video.
def plot_signal_analysis(sample_rows, video_dir, counters, save_dir="models/plots/signal_analysis",
                         use_gpu=True, resize_width=640):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pose_landmarker, mp_module = load_mediapipe_pose(use_gpu=use_gpu)

    for video_name, start, end, gt_count, ex_class in sample_rows:
        video_path = video_dir / video_name
        if not video_path.exists():
            found = list(video_dir.rglob(video_name))
            if found:
                video_path = found[0]
            else:
                continue
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        frame_count = end if end is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(start, frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if resize_width:
                h, w = frame.shape[:2]
                new_h = int(h * resize_width / w)
                frame = cv2.resize(frame, (resize_width, new_h))
            frames.append(frame)
        cap.release()
        if not frames:
            continue

        all_kps = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
            detection_result = pose_landmarker.detect(mp_image)
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                kp = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
            else:
                kp = [0.0]*99
            all_kps.append(np.array(kp, dtype=np.float32))

        hybrid = HybridCounter(exercise=ex_class, fps=30)
        signal_vals = []
        for kp in all_kps:
            val = extract_signal(kp, hybrid.sig_type)
            signal_vals.append(val)
            hybrid.update(kp)
        final_count = hybrid.final_count()

        plt.figure(figsize=(12, 6))
        plt.plot(signal_vals, 'b-', alpha=0.7, label='Raw signal')
        filtered = butter_lowpass_filter(np.array(signal_vals), 2.5, 30)
        plt.plot(filtered, 'g-', linewidth=2, label='Filtered (Butterworth)')
        norm = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)
        peaks, _ = find_peaks(norm, height=0.3, distance=int(30*0.8))
        valleys, _ = find_peaks(-norm, height=0.3, distance=int(30*0.8))
        plt.plot(peaks, filtered[peaks], "ro", label='Peaks')
        plt.plot(valleys, filtered[valleys], "go", label='Valleys')
        plt.title(f"{video_name} | GT: {gt_count} | Pred: {final_count}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"{video_name}_analysis.png", dpi=150)
        plt.close()

    pose_landmarker.close()
    print(f"Signal plots saved to {save_dir}")

# ------------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ------------------------------------------------------------------------------
# When run from the command line, this script evaluates all counters on the
# specified split of RepCountA. Command-line arguments allow me to control
# whether to train the LSTM, save plots, limit the number of videos, etc.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-repcounta", action="store_true", help="Evaluate on RepCountA")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--num-videos", type=int, default=None, help="Limit number of videos")
    parser.add_argument("--train-lstm", action="store_true", help="Train LSTM model before evaluation")
    parser.add_argument("--save-plots", action="store_true", help="Generate signal plots")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--resize", type=int, default=640, help="Width to resize frames (0 to disable)")
    parser.add_argument("--skip-slow", action="store_true", help="Skip FeatureTracking and BackgroundSub (faster)")
    args = parser.parse_args()

    if args.evaluate_repcounta:
        use_gpu = not args.no_gpu
        resize_width = args.resize if args.resize > 0 else None
        evaluate_on_repcounta(split=args.split, num_videos=args.num_videos,
                              train_lstm=args.train_lstm, save_plots=args.save_plots,
                              use_gpu=use_gpu, resize_width=resize_width,
                              skip_slow_methods=args.skip_slow)
    else:
        print("Usage: python rep_counter.py --evaluate-repcounta [--split {train,valid,test}] [--num-videos N] [--train-lstm] [--save-plots] [--no-gpu] [--resize WIDTH] [--skip-slow]")