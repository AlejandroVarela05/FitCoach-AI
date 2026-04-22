# angle_calculator.py


# This utility module computes joint angles from MediaPipe pose landmarks.
# It provides functions to calculate angles between three 3D points, extract all
# relevant angles for exercise analysis, and convert the angle dictionary into
# a fixed‑order numpy vector for machine learning models.
#
# PURPOSE:
#   - Abstract the geometric calculations needed for posture evaluation.
#   - Offer a consistent interface that works with both MediaPipe's landmark
#     objects and plain lists of coordinates.
#   - Produce a vector of angles that can be fed into posture classifiers or
#     knowledge‑based systems.
#
# COURSE CONNECTION:
#   This module supports the "Computer Vision" and "Intelligent Systems" courses
#   by translating raw keypoints into biomechanically meaningful features.
#   The angle calculations are used in the Knowledge Base (rule‑based posture
#   feedback) and in the MLP posture classifier.
#
# DECISIONS:
#   - I define a landmark index mapping so I can refer to joints by name instead
#     of magic numbers.
#   - The `_get_coords` helper handles different landmark representations
#     (MediaPipe NormalizedLandmark, list, or custom object) for flexibility.
#   - I add a small epsilon (1e-8) to the denominator when computing the cosine
#     to avoid division‑by‑zero errors.
#   - I clip the cosine value to [-1, 1] before calling arccos to prevent
#     numerical issues from floating‑point inaccuracies.
#   - The default angle order in `angles_to_vector` matches the 12‑feature
#     vector expected by the posture classifier.



import numpy as np
from typing import Dict, List, Union, Tuple


# I map MediaPipe landmark indices to human‑readable names.
LANDMARK_INDICES = {
    "nose": 0,
    "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8,
    "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}


def _get_coords(landmarks, idx: int) -> Tuple[float, float, float]:
    # I extract the (x, y, z) coordinates of a landmark regardless of the input format.
    if hasattr(landmarks, 'landmark'):
        lm = landmarks.landmark[idx]
    elif isinstance(landmarks, list):
        lm = landmarks[idx]
    else:
        lm = landmarks[idx]  # assume it's already a list of landmarks
    return (lm.x, lm.y, lm.z)


def angle_between_points(a: Tuple[float, float, float],
                         b: Tuple[float, float, float],
                         c: Tuple[float, float, float]) -> float:
    # I compute the angle at point b formed by the segments ba and bc, in degrees.
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def calculate_all_angles(landmarks) -> Dict[str, float]:
    # I compute a dictionary of joint angles relevant for exercise posture analysis.
    angles = {}

    try:
        # Helper to get coordinates by landmark index.
        def p(idx): return _get_coords(landmarks, idx)

        # Knee angles (hip – knee – ankle).
        angles['left_knee'] = angle_between_points(p(23), p(25), p(27))
        angles['right_knee'] = angle_between_points(p(24), p(26), p(28))

        # Hip angles (shoulder – hip – knee).
        angles['left_hip'] = angle_between_points(p(11), p(23), p(25))
        angles['right_hip'] = angle_between_points(p(12), p(24), p(26))

        # Ankle angles (knee – ankle – foot index).
        angles['left_ankle'] = angle_between_points(p(25), p(27), p(31))
        angles['right_ankle'] = angle_between_points(p(26), p(28), p(32))

        # Elbow angles (shoulder – elbow – wrist).
        angles['left_elbow'] = angle_between_points(p(11), p(13), p(15))
        angles['right_elbow'] = angle_between_points(p(12), p(14), p(16))

        # Shoulder angles (elbow – shoulder – hip).
        angles['left_shoulder'] = angle_between_points(p(13), p(11), p(23))
        angles['right_shoulder'] = angle_between_points(p(14), p(12), p(24))

        # Spine angle (approximated by the angle at the hip between shoulder and knee).
        angles['spine'] = angle_between_points(p(11), p(23), p(25))

    except Exception as e:
        print(f"[AngleCalculator] Error computing angles: {e}")

    return angles


def angles_to_vector(angles: Dict[str, float],
                     order: List[str] = None) -> np.ndarray:
    # I convert the angle dictionary into a fixed‑order numpy vector.
    if order is None:
        # Default order matches the 12 features used by the posture classifier.
        order = ['left_knee', 'right_knee', 'left_hip', 'right_hip',
                 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
                 'left_ankle', 'right_ankle', 'spine']
    vec = []
    for key in order:
        vec.append(angles.get(key, 180.0))  # default to extended (180°) if missing
    return np.array(vec, dtype=np.float32)


if __name__ == "__main__":
    print("Angle Calculator module loaded.")
    print(f"Available landmark indices: {list(LANDMARK_INDICES.keys())}")