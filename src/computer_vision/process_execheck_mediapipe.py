# process_execheck_mediapipe.py


# This script preprocesses the ExeCheck dataset by extracting MediaPipe pose
# landmarks from video segments and saving them as numpy arrays.
#
# PURPOSE:
#   - Read the RepSeg.csv file that defines video paths and repetition boundaries.
#   - Use MediaPipe Pose Landmarker (Tasks API) to extract 33 landmarks (99 values)
#     for each frame of every exercise segment.
#   - Save the extracted sequences as .npy and .pkl files for later training.
#
# COURSE CONNECTION:
#   This script implements the data preprocessing step discussed in the
#   "Computer Vision" course (Unit I – feature extraction, Unit II – camera and
#   motion). Extracting keypoints with MediaPipe is the first step in building
#   the rehabilitation exercise classifier.
#
# DECISIONS:
#   - I use the lite version of MediaPipe Pose Landmarker because it is faster
#     and the ExeCheck videos are already clean (single person, good lighting).
#   - I pad or truncate each segment to exactly 160 frames to have a fixed‑size
#     input for the neural network.
#   - The output is stored in the `processed_dataset_mediapipe` folder to keep
#     raw and processed data separate.



import sys
import os
import pickle
import csv
import numpy as np
from pathlib import Path
import cv2
import mediapipe as mp
from tqdm import tqdm

# I add the project root to the system path so I can import the configuration.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import RANDOM_SEED  # RAW_DATA_DIR is not used here (path built manually to avoid spaces)

# This function downloads the MediaPipe Pose Landmarker model (lite) if not already present.
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

    # I configure the pose landmarker with moderate confidence thresholds.
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    # I return both the landmarker object and the mediapipe module (needed for Image creation).
    return vision.PoseLandmarker.create_from_options(options), mp

# Given a video path and start/end frame indices, it extracts a fixed‑length sequence of 99‑dimensional keypoints.
def extract_keypoints_from_video_segment(pose_landmarker, mp_module, video_path, start_frame, end_frame, max_frames=160):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    # I jump to the starting frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_kp = []
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
        detection_result = pose_landmarker.detect(mp_image)
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            kp = []
            for lm in landmarks:
                kp.extend([lm.x, lm.y, lm.z])
            frames_kp.append(kp)
        else:
            # If no pose is detected, I fill with zeros to keep the sequence length consistent.
            frames_kp.append([0.0] * 99)
        frame_idx += 1
    cap.release()
    if len(frames_kp) == 0:
        return None
    frames_kp = np.array(frames_kp, dtype=np.float32)
    # I pad with zeros if the segment is shorter than max_frames, or truncate if longer.
    if len(frames_kp) < max_frames:
        pad = np.zeros((max_frames - len(frames_kp), 99), dtype=np.float32)
        frames_kp = np.concatenate([frames_kp, pad], axis=0)
    else:
        frames_kp = frames_kp[:max_frames]
    return frames_kp

# I define the function read_repseg_csv to parse the RepSeg.csv file that comes with ExeCheck.
def read_repseg_csv(csv_path):
    data_list = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].startswith('file://'):
                continue
            path = row[0][7:]  # remove 'file://' prefix
            # I adjust the path to match my local data directory structure.
            path = path.replace('/research/yiwen/DATA/ExeCheck/', 'data/raw/ExeCheck_Dataset/raw_data/')
            path = os.path.normpath(path)
            frame_numbers = list(map(int, row[1:]))
            data_list.append((path, frame_numbers))
    return data_list

# I define the main function to orchestrate the whole preprocessing process.
def main():
    project_root = Path(__file__).parent.parent.parent
    csv_file = project_root / "data" / "raw" / "ExeCheck_Dataset" / "processing_scripts" / "RepSeg.csv"
    out_dir = project_root / "data" / "raw" / "ExeCheck_Dataset" / "processed_dataset_mediapipe"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_list = read_repseg_csv(csv_file)
    print(f"Total entries in CSV: {len(data_list)}")
    
    # I load the MediaPipe pose landmarker.
    pose_landmarker, mp_module = load_mediapipe_pose()
    print("[MediaPipe] Pose Landmarker loaded (Tasks API)")
    
    all_keypoints = []
    all_labels = []  # (class_id, correctness)
    skipped = 0
    
    # These are the 10 rehabilitation exercise classes in the ExeCheck dataset.
    exercise_names = [
        'arm_circle', 'forward_lunge', 'high_knee_raise', 'hip_abduction',
        'leg_extension', 'shoulder_abduction', 'shoulder_external_rotation',
        'shoulder_flexion', 'side_step_squat', 'squat'
    ]
    exe_to_id = {name: i for i, name in enumerate(exercise_names)}
    correct_to_bool = {'correct': 1, 'incorrect': 0}
    
    # I iterate over each entry in the CSV.
    for mkv_path, seg_list in tqdm(data_list, desc="Processing videos"):
        # The filename indicates the exercise and whether it is correct or incorrect.
        basename = os.path.basename(mkv_path)
        if '_correct' in basename:
            correctness_str = 'correct'
        elif '_incorrect' in basename:
            correctness_str = 'incorrect'
        else:
            print(f"  Skipping {basename}: unknown correctness")
            skipped += 1
            continue
        exe_name = basename.split(f'_{correctness_str}')[0]
        class_id = exe_to_id.get(exe_name)
        if class_id is None:
            print(f"  Skipping {basename}: unknown exercise {exe_name}")
            skipped += 1
            continue
        correctness = correct_to_bool[correctness_str]
        
        # The seg_list contains alternating start and end frame numbers for each repetition.
        for rep_idx in range(len(seg_list)-1):
            start = seg_list[rep_idx]
            end = seg_list[rep_idx+1]
            if start >= end:
                continue
            keypoints = extract_keypoints_from_video_segment(
                pose_landmarker, mp_module, mkv_path, start, end, max_frames=160)
            if keypoints is None:
                continue
            all_keypoints.append(keypoints)
            all_labels.append((class_id, correctness))
    
    pose_landmarker.close()
    
    print(f"Processed {len(all_keypoints)} segments. Skipped {skipped} videos.")
    
    # I stack all sequences into a single numpy array and shuffle them.
    X = np.array(all_keypoints, dtype=np.float32)  # shape (N, 160, 99)
    y = np.array(all_labels, dtype=object)
    perm = np.random.default_rng(RANDOM_SEED).permutation(len(X))
    X = X[perm]
    y = y[perm].tolist()
    
    # I save the keypoints as a .npy file and the labels as a .pkl file.
    np.save(out_dir / "seg_data_joint_mediapipe.npy", X)
    with open(out_dir / "seg_label_mediapipe.pkl", 'wb') as f:
        pickle.dump(y, f)
    
    print(f"Saved {X.shape[0]} samples to {out_dir}")
    print(f"Data shape: {X.shape}")

# I check this condition to make sure I only run this path when the state is valid.
if __name__ == "__main__":
    main()