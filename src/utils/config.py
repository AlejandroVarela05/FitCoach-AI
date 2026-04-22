# config.py


# This module is the central configuration hub for the entire FitCoach AI project.
# It defines project paths, random seed, exercise classes, model hyperparameters,
# dataset metadata, LLM provider settings, and optimisation algorithm parameters.
# All other modules import these constants, ensuring consistency across the codebase.
#
# PURPOSE:
#   - Provide a single source of truth for all configurable values.
#   - Automatically create required directory structures.
#   - Load sensitive API keys from a .env file.
#
# COURSE CONNECTION:
#   This file supports all four courses of the project:
#   - Computer Vision: dataset paths, BiLSTM/MLP hyperparameters.
#   - Intelligent Systems: SA_CONFIG, TABU_CONFIG, knowledge base rules.
#   - Advanced Machine Learning: DQN_CONFIG, training parameters.
#   - Speech & NLP: RAG_CONFIG, LLM provider settings.
#
# DECISIONS:
#   - I store paths using pathlib.Path for cross‑platform compatibility.
#   - I set a fixed random seed (42) for reproducibility.
#   - I load environment variables with dotenv to keep secrets out of version control.
#   - All directories are created at import time so modules can assume they exist.



from pathlib import Path
import os
from dotenv import load_dotenv

# I load environment variables from a .env file (API keys, LLM provider, etc.).
load_dotenv()

# I define the absolute path to the project root. This is where all data and models live.
PROJECT_ROOT = Path(r"C:\Users\Alejandro Varela\Documents\Alejandro_Varela_Garcia\UIE\Ingeniería_en_Sistemas_Inteligentes\Tercer_Curso\Segundo_Cuatri\Final_Project")
DATA_DIR           = PROJECT_ROOT / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
DATA_PROCESSED     = DATA_DIR / "processed"
DATA_INTERIM       = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_PROCESSED          # alias used by exercise_classifier
MODELS_DIR         = PROJECT_ROOT / "models"
CHECKPOINTS_DIR    = MODELS_DIR / "checkpoints"
REPORTS_DIR        = MODELS_DIR / "reports"
PLOTS_DIR          = MODELS_DIR / "plots"
PAPERS_DIR         = DATA_DIR / "papers"
OUTPUTS_DIR        = PROJECT_ROOT / "outputs"
SCRIPTS_DIR        = PROJECT_ROOT / "scripts"

# I ensure all necessary directories exist to avoid file‑not‑found errors later.
for dir_path in [DATA_PROCESSED, DATA_INTERIM, MODELS_DIR, CHECKPOINTS_DIR,
                 REPORTS_DIR, PLOTS_DIR, OUTPUTS_DIR, SCRIPTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# I fix the random seed so experiments are reproducible.
RANDOM_SEED = 42

# The 8 general fitness exercises that the main classifier recognises.
EXERCISE_CLASSES = [
    "push_up", "squat", "shoulder_press", "barbell_biceps_curl",
    "plank", "leg_raises", "lateral_raise", "deadlift"
]
NUM_EXERCISES = len(EXERCISE_CLASSES)

# A mapping from folder names (or label strings) to the canonical class names.
DATASET_FOLDER_TO_CLASS = {
    "push-up":              "push_up",
    "squat":                "squat",
    "shoulder press":       "shoulder_press",
    "barbell biceps curl":  "barbell_biceps_curl",
    "plank":                "plank",
    "leg raises":           "leg_raises",
    "lateral raise":        "lateral_raise",
    "deadlift":             "deadlift",
}

# Hyperparameters for the BiLSTM‑Attention exercise classifier.
BILSTM_CONFIG = {
    "input_size":      99,
    "hidden_size":     128,
    "num_layers":      2,
    "num_classes":     NUM_EXERCISES,
    "dropout":         0.3,
    "sequence_length": 30,    # 30 frames at 30 fps ≈ 1 second
    "batch_size":      32,
    "learning_rate":   0.0001,
    "epochs":          50,
}

# Hyperparameters for the MLP posture classifier (12 joint angles).
MLP_CONFIG = {
    "input_size":    12,
    "num_classes":   6,
    "hidden_sizes":  [64, 128, 64],
    "dropout":       0.2,
    "learning_rate": 0.001,
    "batch_size":    32,
    "epochs":        50,
}

# The six posture error categories used in the original multi‑class formulation.
POSTURE_CLASSES = [
    "correct",
    "knee_error",
    "back_error",
    "depth_error",
    "rhythm_error",
    "neck_error"
]

# Hyperparameters for the Deep Q‑Network (DQN) adaptive coach.
DQN_CONFIG = {
    "learning_rate":         1e-3,
    "buffer_size":           50000,   # size of experience replay buffer
    "learning_starts":       1000,    # steps before first gradient update
    "batch_size":            64,
    "gamma":                 0.99,    # discount factor
    "eps_start":             0.9,     # initial exploration rate
    "eps_end":               0.05,    # minimum exploration rate
    "eps_decay":             1000,    # decay speed (steps)
    "tau":                   0.005,   # soft target‑network update factor
    "total_timesteps":       100000,
    "train_freq":            4,
    "target_update_interval": 1000,
}

# Metadata about the datasets used in the project.
DATASETS = {
    "execheck": {
        "name":          "ExeCheck_Dataset",
        "path":          RAW_DATA_DIR / "ExeCheck_Dataset",
        "type":          "skeleton",
        "description":   "Rehabilitation exercises processed with skeleton data",
        "class_mapping": "execheck"
    },
    "gym_workout": {
        "name":          "Gym_WorkoutExercises_Video",
        "path":          RAW_DATA_DIR / "Gym_WorkoutExercises_Video",
        "type":          "video",
        "description":   "Gym exercise videos (22 classes)",
        "class_mapping": "gym_workout"
    },
    "qevd": {
        "name":          "QEVD",
        "path":          RAW_DATA_DIR / "QEVD",
        "type":          "video",
        "description":   "Qualcomm QEVD-FIT-300k dataset",
        "class_mapping": "qevd"
    },
    "realtime_exercise": {
        "name":          "Real-Time_Exercise_Recognition_Dataset",
        "path":          RAW_DATA_DIR / "Real-Time_Exercise_Recognition_Dataset",
        "type":          "video",
        "description":   "Real-time exercise recognition dataset (5 classes)",
        "class_mapping": "realtime"
    },
    "workout_exercises": {
        "name":          "WorkoutExercises_Video",
        "path":          RAW_DATA_DIR / "WorkoutExercises_Video",
        "type":          "video",
        "description":   "Additional workout video dataset (10 classes)",
        "class_mapping": "workout"
    }
}

# Preprocessing parameters for video and skeleton data.
PREPROCESS_CONFIG = {
    "video": {
        "target_fps":   30,
        "target_size":  (224, 224),
        "max_frames":   300,
        "normalize":    True,
        "augmentation": {
            "horizontal_flip":  True,
            "rotation_range":   15,
            "brightness_range": (0.8, 1.2)
        }
    },
    "skeleton": {
        "num_joints":       32,
        "normalize":        True,
        "sequence_length":  64,
        "features":         ["x", "y", "confidence"]
    }
}

# The class labels (or class lists) for each dataset. Useful for mapping raw labels.
EXERCISE_CLASSES_BY_DATASET = {
    "execheck": {
        0: "arm_circle", 1: "forward_lunge", 2: "high_knee_raise", 3: "hip_abduction",
        4: "leg_extension", 5: "shoulder_abduction", 6: "shoulder_external_rotation",
        7: "shoulder_flexion", 8: "side_step_squat", 9: "squat"
    },
    "gym_workout": [
        "barbell biceps curl", "bench press", "chest fly machine", "deadlift",
        "decline bench press", "hammer curl", "hip thrust", "incline bench press",
        "lat pulldown", "lateral raise", "leg extension", "leg raises", "plank",
        "pull Up", "push-up", "romanian deadlift", "russian twist", "shoulder press",
        "squat", "t bar row", "tricep dips", "tricep Pushdown"
    ],
    "qevd": [
        "air jump rope", "alternating forward lunges", "alternating lateral lunge",
        "alternating single leg glutes bridge", "alternating v ups",
        "arm circles (forward)", "arm crosses", "backwards windmills",
        "bending windmill stretch", "bobbing head (imagine there is music)",
        "boxing bounce steps (front to back)", "boxing bounce steps (side to side)",
        "boxing bounce-steps", "bunny hops", "burpee", "burpee (no pushup)",
        "buttkickers", "cat-cow pose", "catching your breath (crouching)",
        "catching your breath (hand on knees)", "catching your breath (hands behind head)",
        "catching your breath (leaning on something)", "catching your breath (walking around)",
        "changing the webcam view while lying down", "child pose",
        "clapping hands (long)", "clapping hands (short)", "cobra pose",
        "coming closer to the webcam", "criss-cross", "criss-cross (feet on the floor)",
        "cross (left leg front)", "cross (right leg front)",
        "cross + hook left (left leg front)", "cross + hook right (right leg front)",
        "cross + jab (left leg front)", "cross + jab (right leg front)",
        "cross + uppercut left (left leg front)", "cross + uppercut right (right leg front)",
        "cross-legged hamstring stretch", "crouching", "curtsy lunges",
        "dead bugs", "dead bugs (legs only)", "deltoid stretch (left arm)",
        "deltoid stretch (right arm)", "downward dog", "downward dog (frontal)",
        "drinking something from a bottle", "elbow plank", "falling over",
        "feet apart", "fire hydrant", "fire hydrant (standing)",
        "fist bump (hold)", "fist bump (preparation and hold)", "fist bump (quick)",
        "fixing hair (long, both hands)", "fixing hair (long, one hand)",
        "fixing hair (short, one hand)", "floor touches",
        "forward stance forward bend (left leg forward)",
        "forward stance forward bend (right leg forward)", "forward windmills",
        "front lunge kick (left leg)", "front lunge kick (right leg)", "garland pose",
        "give up gesture", "glute hamstring walkout", "glutes bridge",
        "going down on knees", "good morning", "grabbing a bottle (bottle visible from the start)",
        "grabbing a towel (towel visible from the start)", "grabbing an off-screen bottle",
        "grabbing an off-screen towel", "halfway lift", "heel lift",
        "high five (hold)", "high five (preparation and hold)", "high five (quick)",
        "high kicks", "high knees", "high knees march", "high plank",
        "hip abductions (left leg)", "hip abductions (right leg)", "hip circles",
        "hip circles (clockwise)", "hip circles (counterclockwise)",
        "hook left (feet next to each other)", "hook left (left leg front)",
        "hook left (right leg front)", "hook left + cross (left leg front)",
        "hook left + hook right (right leg front)", "hook left + jab (right leg front)",
        "hook left + uppercut right (right leg front)",
        "hook left and cross (left leg front)", "hook left and hook right (left leg front)",
        "hook left and uppercut right (left leg front)",
        "hook right (feet next to each other)", "hook right (left leg front)",
        "hook right (right leg front)", "hook right + cross (right leg front)",
        "hook right + hook left (left leg front)", "hook right + jab (left leg front)",
        "hook right + uppercut left (left leg front)",
        "hook right and cross (right leg front)", "hook right and hook left (right leg front)",
        "hook right and uppercut left (right leg front)", "inchworm",
        "jab (left leg front)", "jab (right leg front)",
        "jab + cross (left leg front)", "jab + cross (right leg front)",
        "jab + hook left (right leg front)", "jab + hook right (left leg front)",
        "jab + uppercut left (right leg front)", "jab + uppercut right (left leg front)",
        "jabs", "jump feet together", "jumping jacks", "jumping lunges",
        "keeping hands in pockets", "kickback", "kickback (left leg)", "kickback (right leg)",
        "knee circles", "leaving plank position", "leg lifts",
        "low lunge pose (left leg back)", "low lunge pose (right leg back)",
        "lunges", "lunges (left leg out in front)", "lunges (right leg out in front)",
        "lunges stance", "lying down after pushup", "lying down in random position",
        "mountain-climbers", "moving plank", "neck rolls",
        "neck warmup (with hands)", "neck warmup (without hands)",
        "nodding head to say yes (long)", "nodding head to say yes (short)",
        "oblique twists", "open and drink from a bottle",
        "opposite arm and leg lifts (on knees)", "picking up the camera",
        "plank preparation", "plank taps", "plank taps (on knees)", "plie squat",
        "pretending to towel off sweat (without using a towel)", "puddle jump",
        "punch left (feet next to each other)", "punch right (feet next to each other)",
        "pushups", "pushups (on knees)", "quad stretch (left)", "quad stretch (right)",
        "quadruped thoracic spine rotation (left)", "quadruped thoracic spine rotation (right)",
        "quick feet", "raised leg circles", "raised leg circles (clockwise)",
        "raised leg circles (counterclockwise)", "reverse crunches", "roll down",
        "running in place", "scratching arm", "scratching back of the head",
        "shaking head to say no (long)", "shaking head to say no (short)",
        "shoulder gators", "shoulder swipe", "shoulder warmup",
        "shrugging (long)", "shrugging (short)", "side plank",
        "sitting down", "sitting on a chair", "small kicks while waiting",
        "snowboarders", "spider man", "spider man pushup", "squat jabs",
        "squat jacks", "squat jump", "squat kick", "squat punch", "squats",
        "standing groin stretch", "standing hamstring stretch",
        "standing kick (alternate legs)", "standing kicks (left leg)",
        "standing kicks (right leg)", "standing knee-to-elbow",
        "standing knee-to-elbow (bouncing)", "standing knee-to-elbow (not bouncing)",
        "standing oblique crunches", "standing t", "standing twy", "standing tyw",
        "standing up", "standing wty", "standing wyt", "standing ytw", "standing ywt",
        "step feet together", "stretching arms", "tabletop position (frontal view)",
        "the hundred (extended legs)", "the hundred (feet on the floor)",
        "the hundred (table-top position)", "thumb down (hold)",
        "thumb down (preparation and hold)", "thumb down (quick)",
        "thumb up (hold)", "thumb up (preparation and hold)", "thumb up (quick)",
        "toe touch",
        "tree pose (left foot on the floor, right foot above knee joint)",
        "tree pose (left foot on the floor, right foot below knee joint)",
        "tree pose (left foot on the floor, right foot on knee joint)",
        "tree pose (right foot on the floor, left foot above knee joint)",
        "tree pose (right foot on the floor, left foot below knee joint)",
        "tree pose (right foot on the floor, left foot on knee joint)",
        "tricep stretch (left arm)", "tricep stretch (right arm)", "tuck jump",
        "uppercut left (feet next to each other)", "uppercut left (left leg front)",
        "uppercut left (right leg front)", "uppercut left + cross (left leg front)",
        "uppercut left + hook right (left leg front)",
        "uppercut left + hook right (right leg front)",
        "uppercut left + jab (right leg front)",
        "uppercut left + uppercut right (left leg front)",
        "uppercut left + uppercut right (right leg front)",
        "uppercut right (feet next to each other)", "uppercut right (left leg front)",
        "uppercut right (right leg front)", "uppercut right + cross (right leg front)",
        "uppercut right + hook left (left leg front)",
        "uppercut right + hook left (right leg front)",
        "uppercut right + jab (left leg front)",
        "uppercut right + uppercut left (left leg front)",
        "uppercut right + uppercut left (right leg front)",
        "upward salute", "using towel to remove sweat", "walking in place",
        "walking towards the webcam", "warrior 1 (left)", "warrior 1 (right)",
        "warrior 2 (left)", "warrior 2 (right)", "waving (hold)",
        "waving (preparation and hold)", "waving (quick)", "wide-legged forward bend",
        "wide-legged forward fold", "wiping face sweat on shirt", "wrist twists",
        "wrist twists (running in-place)", "yawning (covering mouth with hand)",
        "yawning (long)", "yawning (short)", "yoga pushup"
    ],
    "realtime": ["barbell biceps curl", "push-up", "shoulder press", "squat", "hammer curl"],
    "workout":  ["barbell biceps curl", "bench press", "push-up", "lat pulldown",
                 "tricep Pushdown", "incline bench press", "deadlift", "lateral raise",
                 "chest fly machine", "pull Up"]
}

# General training hyperparameters used across several models.
TRAINING_CONFIG = {
    "batch_size":               32,
    "learning_rate":            0.001,
    "epochs":                   50,
    "early_stopping_patience":  10,
    "validation_split":         0.2,
    "num_workers":              4
}

# The primary muscle groups targeted by each exercise. Used by the optimiser to avoid overlap.
EXERCISE_MUSCLE_GROUPS = {
    "push_up":             ["chest", "triceps", "shoulders"],
    "squat":               ["quadriceps", "glutes", "hamstrings"],
    "shoulder_press":      ["shoulders", "triceps", "upper_back"],
    "barbell_biceps_curl": ["biceps", "forearms"],
    "plank":               ["core", "shoulders", "back"],
    "leg_raises":          ["core", "hip_flexors"],
    "lateral_raise":       ["shoulders", "upper_back"],
    "deadlift":            ["hamstrings", "glutes", "lower_back", "core"],
}

# A subset of MediaPipe landmark indices used for angle calculations.
LANDMARK_INDICES = {
    "nose": 0,
    "left_shoulder": 11,   "right_shoulder": 12,
    "left_elbow": 13,      "right_elbow": 14,
    "left_wrist": 15,      "right_wrist": 16,
    "left_hip": 23,        "right_hip": 24,
    "left_knee": 25,       "right_knee": 26,
    "left_ankle": 27,      "right_ankle": 28,
    "left_foot": 31,       "right_foot": 32,
}

# The primary joint used for repetition counting for each exercise.
EXERCISE_PRIMARY_JOINT = {
    "push_up":             12,   # right shoulder
    "squat":               24,   # right hip
    "shoulder_press":      16,   # right wrist
    "barbell_biceps_curl": 16,   # right wrist
    "plank":               24,   # right hip
    "leg_raises":          28,   # right ankle
    "lateral_raise":       16,   # right wrist
    "deadlift":            24,   # right hip
}

# Simulated Annealing parameters for weekly routine optimisation.
SA_CONFIG = {
    "T0":       100.0,   # initial temperature
    "Tf":       0.01,    # final temperature (stopping condition)
    "alpha":    0.95,    # geometric cooling factor
    "max_iter": 2000     # iterations per temperature level
}

# Tabu Search parameters for comparison.
TABU_CONFIG = {
    "tabu_tenure":      20,
    "max_iter":         5000,
    "neighborhood_size": 10,
}

# API keys for external services. Loaded from environment variables.
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Large Language Model configuration for the RAG system.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

LMSTUDIO_URL   = os.getenv("LMSTUDIO_URL",   "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "local-model")

GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")

# Parameters for the Retrieval‑Augmented Generation pipeline.
RAG_CONFIG = {
    "chunk_size":    1000,
    "chunk_overlap": 200,
    "top_k":         5,
    "temperature":   0.3,
}

# Streamlit app configuration (if a web interface is built).
STREAMLIT_CONFIG = {
    "page_title": "FitCoach AI",
    "page_icon":  "fitness",
    "layout":     "wide",
}

# Path to the SQLite database for storing user data.
DATABASE_PATH = PROJECT_ROOT / "fitcoach.db"

# When this script is run directly, it prints a summary of the configuration.
if __name__ == "__main__":
    print("[Config] FitCoach AI — configuration loaded successfully.")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Data dir     : {DATA_DIR}")
    print(f"  Models dir   : {MODELS_DIR}")
    print(f"  Exercises    : {NUM_EXERCISES} classes -> {EXERCISE_CLASSES}")
    print(f"  LLM provider : {LLM_PROVIDER}")
    print(f"  Groq API key : {'SET' if GROQ_API_KEY else 'NOT SET'}")
    print(f"  Tavily key   : {'SET' if TAVILY_API_KEY else 'NOT SET'}")
    print(f"  Database     : {DATABASE_PATH}")