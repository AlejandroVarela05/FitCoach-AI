# pose_pipeline.py


# This module implements the real‑time pose detection pipeline for FitCoach AI.
# It combines YOLOv8 person detection, MediaPipe pose landmark extraction,
# exercise classification, posture evaluation, and repetition counting.
#
# PURPOSE:
#   - Provide a unified interface to process webcam frames.
#   - Apply smoothing (moving average or Kalman) to reduce landmark jitter.
#   - Normalise keypoints to make the system invariant to camera distance.
#   - Log session data (angles, predictions, rep counts) to CSV for analysis.
#   - Optionally record video output and save benchmark timings.
#
# COURSE CONNECTION:
#   This script integrates concepts from "Computer Vision" (Unit I – image
#   processing, Unit II – camera models and feature tracking, Unit III – deep
#   learning for vision) and "Intelligent Systems" (real‑time perception).
#   The modular design follows the teaching‑learning contract requirement to
#   build a complete perception pipeline.
#
# DECISIONS:
#   - I use YOLOv8 to isolate the person because MediaPipe can be distracted by
#     multiple people or cluttered backgrounds.
#   - I normalise keypoints relative to the pelvis and spine length. This makes
#     the system robust to different camera distances and body sizes.
#   - Smoothing methods (moving average / Kalman) reduce the high‑frequency
#     jitter that MediaPipe sometimes produces.
#   - The pipeline can switch between fitness (8 exercises) and rehabilitation
#     (10 exercises) modes dynamically.
#   - Benchmark timers help me evaluate performance bottlenecks (YOLO,
#     MediaPipe, post‑processing).



import cv2
import numpy as np
import os
import sys
import time
import json
import csv
from collections import deque
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# I add the project root to the path so I can import my custom modules.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.angle_calculator import calculate_all_angles, angles_to_vector
from src.utils.config import PROJECT_ROOT, MODELS_DIR, REPORTS_DIR, PLOTS_DIR, RAW_DATA_DIR

# I use safe import blocks here because the classifiers may not be available.
# If a classifier is missing, I disable that feature but the pipeline still runs.
try:
    from src.computer_vision.exercise_classifier import ExerciseClassifierManager
    FITNESS_CLASSIFIER_AVAILABLE = True
except ImportError:
    FITNESS_CLASSIFIER_AVAILABLE = False

try:
    from src.computer_vision.exercise_classifier_rehab import RehabExerciseClassifier
    REHAB_CLASSIFIER_AVAILABLE = True
except ImportError:
    REHAB_CLASSIFIER_AVAILABLE = False

try:
    from src.computer_vision.posture_classifier import PostureClassifier
    POSTURE_AVAILABLE = True
except ImportError:
    POSTURE_AVAILABLE = False

try:
    from src.computer_vision.rep_counter import RepCounter
    REP_COUNTER_AVAILABLE = True
except ImportError:
    REP_COUNTER_AVAILABLE = False


class KalmanFilter1D:
    # I implement a simple 1D Kalman filter to smooth a single coordinate (e.g., x, y, or z).
    # This is used later inside KalmanFilterLandmarks to smooth each coordinate independently.
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2):
        self.x = 0.0        # estimated state (the smoothed value)
        self.P = 1.0        # estimation error covariance
        self.Q = process_noise
        self.R = measurement_noise

    def update(self, measurement):
        # Prediction step: increase uncertainty because of process noise.
        self.P += self.Q
        # Compute Kalman gain.
        K = self.P / (self.P + self.R)
        # Update state estimate with the measurement.
        self.x += K * (measurement - self.x)
        # Update error covariance.
        self.P *= (1 - K)
        return self.x


class KalmanFilterLandmarks:
    # I create 33 independent Kalman filters (one per landmark), each with 3 coordinates (x, y, z).
    # This class smooths all 33 MediaPipe landmarks at once.
    def __init__(self):
        self.filters = [[KalmanFilter1D() for _ in range(3)] for _ in range(33)]

    def smooth(self, landmarks):
        smoothed = []
        for i, lm in enumerate(landmarks):
            sx = self.filters[i][0].update(lm.x)
            sy = self.filters[i][1].update(lm.y)
            sz = self.filters[i][2].update(lm.z)
            # I create a lightweight object with x,y,z attributes to mimic MediaPipe's landmark structure.
            smoothed.append(type('Landmark', (), {'x': sx, 'y': sy, 'z': sz})())
        return smoothed


class PoseDetectionPipeline:
    # This is the main pipeline class. It orchestrates person detection, pose estimation,
    # classification, posture evaluation, and counting.

    def __init__(self, use_yolo=True, smooth_method='moving_avg', smooth_window=5,
                 normalise_keypoints=True, enable_posture_feedback=True,
                 enable_rep_counter=True, enable_benchmark=True,
                 record_video=False, video_output_path=None,
                 mode='fitness'):
        # Configuration parameters.
        self.use_yolo = use_yolo
        self.smooth_method = smooth_method
        self.smooth_window = smooth_window
        self.normalise_keypoints = normalise_keypoints
        self.enable_posture_feedback = enable_posture_feedback and POSTURE_AVAILABLE
        self.enable_rep_counter = enable_rep_counter and REP_COUNTER_AVAILABLE
        self.enable_benchmark = enable_benchmark
        self.record_video = record_video
        self.mode = mode

        # Smoothing state.
        self.landmark_history = deque(maxlen=smooth_window) if smooth_method == 'moving_avg' else None
        self.kalman_filter = KalmanFilterLandmarks() if smooth_method == 'kalman' else None

        # Models and classifiers (lazy initialisation).
        self.yolo_model = None
        self.pose_landmarker = None
        self.posture_classifier = None
        self.rep_counter = None
        self.exercise_classifier = None
        self._keypoints_buffer = deque(maxlen=30)  # collect 30 frames for the classifier window

        # Performance monitoring.
        self.prev_time = time.time()
        self.fps = 0.0
        self.benchmark_times = {'yolo': 0.0, 'mediapipe': 0.0, 'postprocess': 0.0}

        # Video recording.
        self.video_writer = None
        self.video_output_path = video_output_path

        # CSV logging.
        self.csv_file = None
        self.csv_writer = None
        self.frame_count = 0

        # Session metrics for final report.
        self.session_metrics = {
            'frames_processed': 0,
            'person_detected_frames': 0,
            'avg_fps': 0.0,
            'rep_counts': {},
            'posture_feedback_counts': {},
            'exercise_predictions': [],
        }

        # I initialise YOLOv8 (nano version) for fast person detection.
        if self.use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO("yolov8n.pt")
                print("[PosePipeline] YOLOv8n loaded")
            except Exception as e:
                print(f"[PosePipeline] YOLO failed: {e} — falling back to full frame")
                self.use_yolo = False

        # I load MediaPipe Pose Landmarker (heavy model for better accuracy).
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request

            self.mp = mp
            mp_dir = Path(mp.__file__).parent
            model_path = mp_dir / 'modules' / 'pose_landmarker' / 'pose_landmarker_heavy.task'
            if not model_path.exists():
                print("[PosePipeline] Model not found locally, downloading...")
                model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
                model_path = mp_dir / 'pose_landmarker_heavy.task'
                urllib.request.urlretrieve(model_url, str(model_path))
                print(f"[PosePipeline] Model downloaded to {model_path}")

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.7,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            print("[PosePipeline] MediaPipe Pose Landmarker loaded (Tasks API)")
        except Exception as e:
            print(f"[PosePipeline] MediaPipe failed: {e}")
            raise

        # I load the appropriate exercise classifier based on the selected mode.
        if self.mode == 'fitness' and FITNESS_CLASSIFIER_AVAILABLE:
            try:
                self.exercise_classifier = ExerciseClassifierManager()
                print("[PosePipeline] Fitness exercise classifier loaded")
            except Exception as e:
                print(f"[PosePipeline] Fitness classifier failed: {e}")
        elif self.mode == 'rehab' and REHAB_CLASSIFIER_AVAILABLE:
            try:
                self.exercise_classifier = RehabExerciseClassifier()
                print("[PosePipeline] Rehabilitation exercise classifier loaded")
            except Exception as e:
                print(f"[PosePipeline] Rehab classifier failed: {e}")

        # I load the posture quality classifier if enabled.
        if self.enable_posture_feedback:
            try:
                self.posture_classifier = PostureClassifier()
                print("[PosePipeline] Posture classifier loaded")
            except Exception as e:
                print(f"[PosePipeline] Posture classifier failed: {e}")
                self.enable_posture_feedback = False

        # I load the repetition counter if enabled.
        if self.enable_rep_counter:
            try:
                self.rep_counter = RepCounter()
                print("[PosePipeline] Rep counter loaded")
            except Exception as e:
                print(f"[PosePipeline] Rep counter failed: {e}")
                self.enable_rep_counter = False

        # Start CSV logging for this session.
        self._init_csv_logging()

    def _init_csv_logging(self):
        # I create a unique filename with the current timestamp.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = REPORTS_DIR / f"pose_session_{timestamp}.csv"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        header = ['frame', 'timestamp', 'person_detected']
        for ang in ['left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_elbow', 'right_elbow']:
            header.append(ang)
        header.extend(['posture_quality', 'posture_feedback', 'rep_count', 'exercise', 'exercise_conf'])
        self.csv_writer.writerow(header)
        print(f"[PosePipeline] CSV logging to {csv_path}")

    def _init_video_writer(self, frame_width, frame_height, fps=20.0):
        if self.video_output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_dir = PROJECT_ROOT / "outputs" / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            self.video_output_path = video_dir / f"pose_session_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(self.video_output_path), fourcc, fps, (frame_width, frame_height))
        print(f"[PosePipeline] Recording video to {self.video_output_path}")

    def _smooth_landmarks(self, landmarks):
        # Apply the selected smoothing method to reduce jitter.
        if self.smooth_method == 'none':
            return landmarks
        if self.smooth_method == 'kalman' and self.kalman_filter:
            return self.kalman_filter.smooth(landmarks)
        if self.landmark_history is None:
            return landmarks
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        self.landmark_history.append(coords)
        if len(self.landmark_history) < self.smooth_window:
            return landmarks
        smoothed = np.mean(self.landmark_history, axis=0)
        return [type('Landmark', (), {'x': float(c[0]), 'y': float(c[1]), 'z': float(c[2])})() for c in smoothed]

    def _normalise_keypoints(self, keypoints_99):
        # Normalise keypoints to make them invariant to camera distance and body size.
        # I translate the origin to the pelvis and divide by spine length.
        if not self.normalise_keypoints:
            return keypoints_99
        kp = keypoints_99.reshape(-1, 3)
        pelvis = kp[23].copy()
        neck = kp[3]
        spine_len = np.linalg.norm(neck - pelvis) + 1e-8
        kp = (kp - pelvis) / spine_len
        return kp.flatten()

    def _detect_person_yolo(self, frame):
        # Use YOLOv8 to find the most prominent person in the frame.
        # I select the detection with the largest area, with a slight bias towards the center.
        yolo_results = self.yolo_model(frame, classes=[0], verbose=False)
        if len(yolo_results[0].boxes) == 0:
            return None, None
        boxes = yolo_results[0].boxes
        h, w = frame.shape[:2]
        frame_center = np.array([w/2, h/2])

        best_idx = 0
        best_score = -1
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            area = (x2 - x1) * (y2 - y1)
            center = np.array([(x1+x2)/2, (y1+y2)/2])
            score = area * 0.7 + (1.0 - np.linalg.norm(center - frame_center) / (w/2)) * 0.3
            if score > best_score:
                best_score = score
                best_idx = i

        best_box = boxes[best_idx]
        xyxy = best_box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)

    def process_frame(self, frame):
        # Main processing routine called for every video frame.
        frame_start = time.time()
        self.frame_count += 1
        self.session_metrics['frames_processed'] += 1

        h, w = frame.shape[:2]
        result = {
            'landmarks': None,
            'angles': {},
            'angles_vector': None,
            'bbox': None,
            'annotated_frame': frame.copy(),
            'person_detected': False,
            'keypoints_sequence': None,
            'fps': self.fps,
            'posture_quality': None,
            'posture_feedback': None,
            'rep_count': 0,
            'exercise': None,
            'exercise_confidence': 0.0,
            'benchmark': {},
        }

        # Step 1: Person detection (optional YOLO).
        t0 = time.time()
        if self.use_yolo and self.yolo_model:
            roi, bbox = self._detect_person_yolo(frame)
            result['bbox'] = bbox
        else:
            roi = frame
            bbox = (0, 0, w, h)
            result['bbox'] = bbox
        result['benchmark']['yolo'] = (time.time() - t0) * 1000

        if roi is None:
            self._update_fps()
            return result

        # Step 2: MediaPipe pose estimation.
        t1 = time.time()
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=roi_rgb)
        detection_result = self.pose_landmarker.detect(mp_image)
        result['benchmark']['mediapipe'] = (time.time() - t1) * 1000

        if not detection_result.pose_landmarks:
            self._update_fps()
            return result

        result['person_detected'] = True
        self.session_metrics['person_detected_frames'] += 1

        pose_landmarks = detection_result.pose_landmarks[0]
        # Convert normalised ROI coordinates back to full‑frame normalised coordinates.
        landmarks = []
        for lm in pose_landmarks:
            rx = lm.x / roi.shape[1] if roi.shape[1] > 0 else 0.0
            ry = lm.y / roi.shape[0] if roi.shape[0] > 0 else 0.0
            rz = lm.z / roi.shape[1] if roi.shape[1] > 0 else 0.0
            landmarks.append(type('Landmark', (), {'x': rx, 'y': ry, 'z': rz})())

        # Step 3: Smoothing and normalisation.
        t2 = time.time()
        smoothed = self._smooth_landmarks(landmarks)
        result['landmarks'] = smoothed
        keypoints_99 = np.array([coord for lm in smoothed for coord in (lm.x, lm.y, lm.z)], dtype=np.float32)
        if self.normalise_keypoints:
            keypoints_99 = self._normalise_keypoints(keypoints_99)
        result['keypoints_sequence'] = keypoints_99

        # Step 4: Calculate joint angles.
        try:
            angles = calculate_all_angles(result['landmarks'])
            result['angles'] = angles
            result['angles_vector'] = angles_to_vector(angles)
        except Exception as e:
            print(f"[PosePipeline] Angle error: {e}")

        # Step 5: Exercise classification (every 30 frames).
        if self.exercise_classifier and result['keypoints_sequence'] is not None:
            self._keypoints_buffer.append(result['keypoints_sequence'])
            if len(self._keypoints_buffer) == 30:
                window = np.array(self._keypoints_buffer)
                exercise, confidence = self.exercise_classifier.predict_exercise(window)
                result['exercise'] = exercise
                result['exercise_confidence'] = confidence
                self.session_metrics['exercise_predictions'].append({
                    'frame': self.frame_count,
                    'exercise': exercise,
                    'confidence': float(confidence)
                })

        # Step 6: Posture quality evaluation.
        if self.enable_posture_feedback and self.posture_classifier and result['angles_vector'] is not None:
            quality, feedback = self.posture_classifier.predict(result['angles_vector'])
            result['posture_quality'] = quality
            result['posture_feedback'] = feedback
            self.session_metrics['posture_feedback_counts'][quality] = \
                self.session_metrics['posture_feedback_counts'].get(quality, 0) + 1

        # Step 7: Repetition counting.
        if self.enable_rep_counter and self.rep_counter and result['angles']:
            rep_count = self.rep_counter.update(landmarks[24].x, landmarks[24].y)
            result['rep_count'] = rep_count
            self.session_metrics['rep_counts']['total'] = rep_count

        result['benchmark']['postprocess'] = (time.time() - t2) * 1000

        # Draw annotations on the frame.
        annotated = self._draw_annotations(result, frame, pose_landmarks, bbox)
        result['annotated_frame'] = annotated

        # Log data to CSV.
        self._log_to_csv(result)

        # Record video if enabled.
        if self.record_video:
            if self.video_writer is None:
                self._init_video_writer(w, h)
            self.video_writer.write(annotated)

        self._update_fps()
        return result

    def _draw_annotations(self, result, frame, pose_landmarks, bbox):
        # Draw bounding box, skeleton, and text overlays on the frame.
        annotated = frame.copy()
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        if self.use_yolo:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, "Person", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # MediaPipe connections I choose to draw (subset for clarity).
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (23, 25), (25, 27), (24, 26), (26, 28),
            (27, 29), (28, 30), (29, 31), (30, 32),
        ]
        try:
            for start_idx, end_idx in connections:
                start_lm = pose_landmarks[start_idx]
                end_lm = pose_landmarks[end_idx]
                sx = int(x1 + start_lm.x)
                sy = int(y1 + start_lm.y)
                ex = int(x1 + end_lm.x)
                ey = int(y1 + end_lm.y)
                cv2.line(annotated, (sx, sy), (ex, ey), (0, 255, 0), 2)
            for lm in pose_landmarks:
                px = int(x1 + lm.x)
                py = int(y1 + lm.y)
                cv2.circle(annotated, (px, py), 3, (0, 255, 0), -1)
        except Exception:
            pass

        y_offset = 25
        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        y_offset += 25

        if self.enable_benchmark:
            bench = result['benchmark']
            cv2.putText(annotated, f"YOLO: {bench.get('yolo',0):.1f}ms", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            y_offset += 18
            cv2.putText(annotated, f"MP: {bench.get('mediapipe',0):.1f}ms", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            y_offset += 18
            cv2.putText(annotated, f"Post: {bench.get('postprocess',0):.1f}ms", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            y_offset += 25

        if result['exercise']:
            ex_text = f"Exercise: {result['exercise']} ({result['exercise_confidence']:.2f})"
            cv2.putText(annotated, ex_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            y_offset += 25

        if result['angles']:
            for name in ['left_knee', 'right_knee', 'left_hip', 'right_hip']:
                if name in result['angles']:
                    val = result['angles'][name]
                    cv2.putText(annotated, f"{name}: {val:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                    y_offset += 18

        if self.enable_rep_counter:
            cv2.putText(annotated, f"Reps: {result['rep_count']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            y_offset += 25

        if result['posture_quality'] is not None:
            quality = result['posture_quality']
            color = (0,255,0) if quality == 'correct' else (0,0,255)
            cv2.putText(annotated, f"Posture: {quality}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 20
            if result['posture_feedback']:
                cv2.putText(annotated, result['posture_feedback'][:40], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        cv2.putText(annotated, f"Smooth: {self.smooth_method} Norm: {self.normalise_keypoints} Mode: {self.mode}",
                    (w-350, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        return annotated

    def _log_to_csv(self, result):
        if self.csv_writer is None:
            return
        row = [self.frame_count, datetime.now().isoformat(), int(result['person_detected'])]
        for ang in ['left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_elbow', 'right_elbow']:
            row.append(result['angles'].get(ang, ''))
        row.append(result['posture_quality'] or '')
        row.append(result['posture_feedback'] or '')
        row.append(result['rep_count'])
        row.append(result['exercise'] or '')
        row.append(f"{result['exercise_confidence']:.3f}" if result['exercise_confidence'] else '')
        self.csv_writer.writerow(row)

    def _update_fps(self):
        current_time = time.time()
        if self.prev_time:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / (current_time - self.prev_time))
        self.prev_time = current_time

    def release(self):
        # Clean up resources: close MediaPipe, CSV file, video writer, and save session report.
        if hasattr(self, 'pose_landmarker') and self.pose_landmarker:
            self.pose_landmarker.close()
        if self.csv_file:
            self.csv_file.close()
        if self.video_writer:
            self.video_writer.release()
        self.session_metrics['avg_fps'] = self.fps
        report_path = REPORTS_DIR / f"pose_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.session_metrics, f, indent=2)
        print(f"[PosePipeline] Session report saved to {report_path}")
        print("[PosePipeline] Resources released.")


if __name__ == "__main__":
    # Demo script: run the pipeline on the default webcam.
    # Press keys to interact: 'q' quit, 's' cycle smoothing, 'n' toggle normalisation, etc.
    print("FitCoach AI — Pose Detection Pipeline (Full Academic Version)")
    print("Controls: q=quit, s=cycle smoothing, n=toggle normalisation,")
    print("          b=toggle benchmark, f=toggle posture, r=toggle recording, m=toggle mode")

    pipeline = PoseDetectionPipeline(
        use_yolo=True,
        smooth_method='moving_avg',
        smooth_window=5,
        normalise_keypoints=True,
        enable_posture_feedback=True,
        enable_rep_counter=True,
        enable_benchmark=True,
        record_video=False,
        mode='fitness'
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    smooth_methods = ['moving_avg', 'kalman', 'none']
    smooth_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame)
        cv2.imshow("Pose Detection Pipeline", result['annotated_frame'])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            smooth_idx = (smooth_idx + 1) % len(smooth_methods)
            pipeline.smooth_method = smooth_methods[smooth_idx]
            if pipeline.smooth_method == 'moving_avg':
                pipeline.landmark_history = deque(maxlen=pipeline.smooth_window)
            else:
                pipeline.landmark_history = None
            if pipeline.smooth_method == 'kalman':
                pipeline.kalman_filter = KalmanFilterLandmarks()
            else:
                pipeline.kalman_filter = None
            print(f"Smoothing method: {pipeline.smooth_method}")
        elif key == ord('n'):
            pipeline.normalise_keypoints = not pipeline.normalise_keypoints
            print(f"Normalisation: {pipeline.normalise_keypoints}")
        elif key == ord('b'):
            pipeline.enable_benchmark = not pipeline.enable_benchmark
            print(f"Benchmark display: {pipeline.enable_benchmark}")
        elif key == ord('f'):
            if POSTURE_AVAILABLE:
                pipeline.enable_posture_feedback = not pipeline.enable_posture_feedback
                print(f"Posture feedback: {pipeline.enable_posture_feedback}")
        elif key == ord('r'):
            pipeline.record_video = not pipeline.record_video
            if pipeline.record_video and pipeline.video_writer is None:
                h, w = frame.shape[:2]
                pipeline._init_video_writer(w, h)
            print(f"Video recording: {pipeline.record_video}")
        elif key == ord('m'):
            new_mode = 'rehab' if pipeline.mode == 'fitness' else 'fitness'
            print(f"Switching mode to {new_mode}...")
            pipeline.mode = new_mode
            if new_mode == 'fitness' and FITNESS_CLASSIFIER_AVAILABLE:
                pipeline.exercise_classifier = ExerciseClassifierManager()
                print("  Fitness classifier loaded.")
            elif new_mode == 'rehab' and REHAB_CLASSIFIER_AVAILABLE:
                pipeline.exercise_classifier = RehabExerciseClassifier()
                print("  Rehabilitation classifier loaded.")
            else:
                pipeline.exercise_classifier = None
                print("  No classifier available for this mode.")

    cap.release()
    pipeline.release()
    cv2.destroyAllWindows()
    print("Demo finished.")