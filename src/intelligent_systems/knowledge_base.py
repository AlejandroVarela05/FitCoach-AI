# knowledge_base.py


# This module implements a rule‑based Knowledge Base System (KBS) for evaluating
# exercise posture. It contains IF‑THEN rules for general fitness exercises and
# rehabilitation exercises, all backed by scientific literature.
#
# PURPOSE:
#   - Provide explainable posture feedback using expert biomechanical rules.
#   - Support both strict (paper) and practical (relaxed) thresholds.
#   - Generate synthetic test data to compare the KBS with the MLP posture classifier.
#
# COURSE CONNECTION:
#   This module relates to the "Intelligent Systems" course (Unit IV – Knowledge
#   Representation and Reasoning). It demonstrates how logical agents can use
#   structured knowledge to make decisions. Each rule includes a citation, showing
#   the integration of evidence‑based practice.
#
# DECISIONS:
#   - I use a dictionary of lambda conditions so rules are evaluated quickly.
#   - Two threshold modes allow flexibility: strict values match the cited papers,
#     while practical values are more forgiving for real‑world use.
#   - The KnowledgeBase class can be instantiated with or without rehabilitation
#     rules, keeping it modular.



import os
import sys
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# I set a flag to switch between strict (paper) thresholds and more forgiving practical thresholds.
USE_PRACTICAL_THRESHOLDS = True  # Change to False to use strict paper values

# The eight general fitness exercises supported by the KBS.
GENERAL_EXERCISES = [
    "push_up", "squat", "shoulder_press", "barbell_biceps_curl",
    "plank", "leg_raises", "lateral_raise", "deadlift"
]

# The ten rehabilitation exercises supported by the KBS.
REHAB_EXERCISES = [
    "arm_circle", "forward_lunge", "high_knee_raise", "hip_abduction",
    "leg_extension", "shoulder_abduction", "shoulder_external_rotation",
    "shoulder_flexion", "side_step_squat", "squat"
]

ALL_EXERCISES = GENERAL_EXERCISES + REHAB_EXERCISES

# The names of the 12 joint angles used as input for the rules.
ANGLE_NAMES = [
    "left_knee", "right_knee", "left_hip", "right_hip", "spine",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "neck", "torso_lean", "head_tilt"
]

# A helper function that returns the practical threshold if the flag is True, otherwise the strict one.
def thresh(strict_val, practical_val):
    return practical_val if USE_PRACTICAL_THRESHOLDS else strict_val

# The rule set for general fitness exercises. Each rule has a name, a condition (lambda),
# a feedback message, a scientific reference, and a description of the threshold.
GENERAL_RULES = {
    "squat": [
        {
            "name": "knee_too_deep",
            "condition": lambda a: a["left_knee"] < thresh(65, 45) or a["right_knee"] < thresh(65, 45),
            "message": f"Knees bending too much (<{thresh(65,45)}°). Risk of ligament strain.",
            "paper": "Escamilla et al. (2001). Knee biomechanics of the dynamic squat. Medicine & Science in Sports & Exercise, 33(1), 108-115.",
            "threshold": f"<{thresh(65,45)}°"
        },
        {
            "name": "back_rounded",
            "condition": lambda a: a["spine"] < thresh(150, 132),
            "message": f"Spine rounded (<{thresh(150,132)}°). Keep your back straight.",
            "paper": "Swinton et al. (2012). A biomechanical comparison of the traditional squat, powerlifting squat, and box squat. Journal of Strength and Conditioning Research, 26(7), 1805-1816.",
            "threshold": f"<{thresh(150,132)}°"
        },
        {
            "name": "insufficient_depth",
            "condition": lambda a: a["left_knee"] > thresh(145, 150) and a["right_knee"] > thresh(145, 150),
            "message": f"Not deep enough (>{thresh(145,150)}°). Lower your hips.",
            "paper": "Schoenfeld, B. J. (2010). Squatting kinematics and kinetics and their application to exercise performance. Journal of Strength and Conditioning Research, 24(12), 3497-3506.",
            "threshold": f">{thresh(145,150)}°"
        },
        {
            "name": "knee_valgus",
            "condition": lambda a: abs(a["left_knee"] - a["right_knee"]) > thresh(20, 30),
            "message": f"Knees collapsing inward (difference >{thresh(20,30)}°). Keep aligned with feet.",
            "paper": "Powers, C. M. (2003). The influence of altered lower-extremity kinematics on patellofemoral joint dysfunction: a theoretical perspective. Journal of Orthopaedic & Sports Physical Therapy, 33(11), 639-646.",
            "threshold": f"difference >{thresh(20,30)}°"
        }
    ],
    "push_up": [
        {
            "name": "elbow_not_low_enough",
            "condition": lambda a: a["left_elbow"] > thresh(120, 150) or a["right_elbow"] > thresh(120, 150),
            "message": f"Elbow angle >{thresh(120,150)}° at bottom. Go deeper.",
            "paper": "Calatayud et al. (2015). Progression of core stability exercises based on electromyographic analysis. Journal of Strength and Conditioning Research, 29(8), 2302-2310.",
            "threshold": f">{thresh(120,150)}°"
        },
        {
            "name": "back_sagging",
            "condition": lambda a: a["spine"] < thresh(170, 150),
            "message": f"Back sagging (spine <{thresh(170,150)}°). Keep straight line head to heels.",
            "paper": "Kontor, J. (2016). The push-up: A comprehensive review of biomechanics and training implications. Strength and Conditioning Journal, 38(4), 10-20.",
            "threshold": f"<{thresh(170,150)}°"
        }
    ],
    "shoulder_press": [
        {
            "name": "elbow_not_full_extension",
            "condition": lambda a: a["left_elbow"] < thresh(170, 140) or a["right_elbow"] < thresh(170, 140),
            "message": f"Elbows not fully extended at top (<{thresh(170,140)}°). Lock out your arms.",
            "paper": "NSCA (2016). Essentials of Strength Training and Conditioning (4th ed.). Human Kinetics.",
            "threshold": f"<{thresh(170,140)}°"
        },
        {
            "name": "back_hyperextension",
            "condition": lambda a: a["spine"] > thresh(180, 205),
            "message": f"Back overarched (>{thresh(180,205)}°). Keep core tight.",
            "paper": "ACSM (2009). American College of Sports Medicine position stand. Progression models in resistance training for healthy adults. Medicine & Science in Sports & Exercise, 41(3), 687-708.",
            "threshold": f">{thresh(180,205)}°"
        }
    ],
    "barbell_biceps_curl": [
        {
            "name": "elbow_moving_forward",
            "condition": lambda a: a["left_shoulder"] < thresh(30, 12) or a["right_shoulder"] < thresh(30, 12),
            "message": f"Elbow moving forward (shoulder angle <{thresh(30,12)}°). Keep elbows tucked.",
            "paper": "NSCA (2016). Essentials of Strength Training and Conditioning (4th ed.). Human Kinetics.",
            "threshold": f"shoulder angle <{thresh(30,12)}°"
        },
        {
            "name": "incomplete_range",
            "condition": lambda a: a["left_elbow"] > thresh(50, 85) or a["right_elbow"] > thresh(50, 85),
            "message": f"Incomplete curl (elbow >{thresh(50,85)}° at top). Bring bar all the way up.",
            "paper": "American Council on Exercise (2014). ACE Personal Trainer Manual (5th ed.).",
            "threshold": f"elbow >{thresh(50,85)}° at top"
        }
    ],
    "plank": [
        {
            "name": "hip_dropping",
            "condition": lambda a: a["spine"] < thresh(160, 145),
            "message": f"Hips dropping (spine <{thresh(160,145)}°). Keep body straight.",
            "paper": "Calatayud et al. (2015). Progression of core stability exercises. Journal of Strength and Conditioning Research, 29(8), 2302-2310.",
            "threshold": f"<{thresh(160,145)}°"
        },
        {
            "name": "hip_raising",
            "condition": lambda a: a["spine"] > thresh(190, 215),
            "message": f"Hips too high (spine >{thresh(190,215)}°). Lower them.",
            "paper": "NSCA (2016). Essentials of Strength Training and Conditioning.",
            "threshold": f">{thresh(190,215)}°"
        }
    ],
    "leg_raises": [
        {
            "name": "lower_back_lifting",
            "condition": lambda a: a["spine"] < thresh(170, 150),
            "message": f"Lower back lifting off floor (spine <{thresh(170,150)}°). Press down.",
            "paper": "ACSM (2009). Progression models in resistance training.",
            "threshold": f"<{thresh(170,150)}°"
        },
        {
            "name": "knee_bending",
            "condition": lambda a: a["left_knee"] < thresh(160, 140) or a["right_knee"] < thresh(160, 140),
            "message": f"Knees bending (<{thresh(160,140)}°). Keep legs straight.",
            "paper": "NSCA (2016). Essentials of Strength Training and Conditioning.",
            "threshold": f"<{thresh(160,140)}°"
        }
    ],
    "lateral_raise": [
        {
            "name": "elbow_above_shoulder",
            "condition": lambda a: a["left_elbow"] > a["left_shoulder"] + thresh(30, 50),
            "message": f"Elbows too high (elbow > shoulder+{thresh(30,50)}°). Stop at shoulder height.",
            "paper": "NSCA (2016). Essentials of Strength Training and Conditioning.",
            "threshold": f"elbow > shoulder+{thresh(30,50)}°"
        }
    ],
    "deadlift": [
        {
            "name": "rounded_back",
            "condition": lambda a: a["spine"] < thresh(155, 135),
            "message": f"Back rounded (spine <{thresh(155,135)}°). Keep neutral spine.",
            "paper": "Swinton et al. (2012). A biomechanical comparison of the traditional squat, powerlifting squat, and box squat. JSCR, 26(7), 1805-1816.",
            "threshold": f"<{thresh(155,135)}°"
        },
        {
            "name": "knee_collapse",
            "condition": lambda a: a["left_knee"] < thresh(90, 70) or a["right_knee"] < thresh(90, 70),
            "message": f"Knees too bent (<{thresh(90,70)}°). Straighten earlier.",
            "paper": "Escamilla et al. (2000). Biomechanics of the deadlift. Strength and Conditioning Journal, 22(4), 28-33.",
            "threshold": f"<{thresh(90,70)}°"
        }
    ]
}

# The rule set for rehabilitation exercises. It follows the same structure as GENERAL_RULES.
REHAB_RULES = {
    "arm_circle": [
        {
            "name": "shoulder_abduction_limited",
            "condition": lambda a: a["left_shoulder"] < thresh(60, 45) or a["right_shoulder"] < thresh(60, 45),
            "message": f"Arm circle too small (<{thresh(60,45)}°). Increase range of motion.",
            "paper": "Wilke et al. (2019). Shoulder rehabilitation exercises: a systematic review. Sports Medicine, 49(4), 589-605.",
            "threshold": f"<{thresh(60,45)}°"
        }
    ],
    "forward_lunge": [
        {
            "name": "knee_past_toes",
            "condition": lambda a: a["left_knee"] < thresh(80, 65) or a["right_knee"] < thresh(80, 65),
            "message": f"Knee past toes (<{thresh(80,65)}°). Step further forward.",
            "paper": "Escamilla et al. (2001). Knee biomechanics of the dynamic squat (adapted for lunge).",
            "threshold": f"<{thresh(80,65)}°"
        },
        {
            "name": "back_lean",
            "condition": lambda a: a["spine"] < thresh(160, 145),
            "message": f"Torso leaning forward (spine <{thresh(160,145)}°). Keep upright.",
            "paper": "Swinton et al. (2012). Biomechanics of squat (adapted).",
            "threshold": f"<{thresh(160,145)}°"
        }
    ],
    "high_knee_raise": [
        {
            "name": "hip_flexion_insufficient",
            "condition": lambda a: a["left_hip"] < thresh(80, 62) or a["right_hip"] < thresh(80, 62),
            "message": f"Raise knee higher (hip angle <{thresh(80,62)}°). Aim for 90°.",
            "paper": "NSCA (2016). Essentials of Strength Training and Conditioning.",
            "threshold": f"<{thresh(80,62)}°"
        }
    ],
    "hip_abduction": [
        {
            "name": "hip_abduction_limited",
            "condition": lambda a: a["left_hip"] - a["right_hip"] < thresh(20, 10),
            "message": f"Lift leg higher to the side (difference <{thresh(20,10)}°).",
            "paper": "Neumann, D. A. (2010). Kinesiology of the hip: a focus on muscular actions. Journal of Orthopaedic & Sports Physical Therapy, 40(2), 82-94.",
            "threshold": f"difference <{thresh(20,10)}°"
        }
    ],
    "leg_extension": [
        {
            "name": "knee_extension_incomplete",
            "condition": lambda a: a["left_knee"] > thresh(150, 168) or a["right_knee"] > thresh(150, 168),
            "message": f"Straighten your knee fully (angle >{thresh(150,168)}°).",
            "paper": "Escamilla et al. (2001). Knee biomechanics (adapted).",
            "threshold": f">{thresh(150,168)}°"
        }
    ],
    "shoulder_abduction": [
        {
            "name": "shoulder_abduction_limited",
            "condition": lambda a: a["left_shoulder"] < thresh(80, 62) or a["right_shoulder"] < thresh(80, 62),
            "message": f"Raise arms higher to shoulder level (<{thresh(80,62)}°).",
            "paper": "Wilke et al. (2019). Shoulder rehabilitation exercises.",
            "threshold": f"<{thresh(80,62)}°"
        }
    ],
    "shoulder_external_rotation": [
        {
            "name": "rotation_limited",
            "condition": lambda a: a["left_shoulder"] - a["right_shoulder"] < thresh(30, 15),
            "message": f"Rotate further outward (difference <{thresh(30,15)}°).",
            "paper": "Ellenbecker, T. S., & Cools, A. (2010). Shoulder rehabilitation. Sports Medicine, 40(8), 663-685.",
            "threshold": f"<{thresh(30,15)}° difference"
        }
    ],
    "shoulder_flexion": [
        {
            "name": "flexion_limited",
            "condition": lambda a: a["left_shoulder"] < thresh(140, 110) or a["right_shoulder"] < thresh(140, 110),
            "message": f"Raise arms higher overhead (<{thresh(140,110)}°).",
            "paper": "NSCA (2016). Essentials of Strength Training and Conditioning.",
            "threshold": f"<{thresh(140,110)}°"
        }
    ],
    "side_step_squat": [
        {
            "name": "knee_valgus",
            "condition": lambda a: abs(a["left_knee"] - a["right_knee"]) > thresh(20, 30),
            "message": f"Knees collapsing inward (difference >{thresh(20,30)}°). Keep aligned.",
            "paper": "Powers (2003). Knee valgus in squat (adapted).",
            "threshold": f"difference >{thresh(20,30)}°"
        }
    ],
    "squat": [
        {
            "name": "knee_too_deep",
            "condition": lambda a: a["left_knee"] < thresh(65, 45) or a["right_knee"] < thresh(65, 45),
            "message": f"Knees bending too much (<{thresh(65,45)}°).",
            "paper": "Escamilla et al. (2001). Knee biomechanics of the dynamic squat.",
            "threshold": f"<{thresh(65,45)}°"
        }
    ]
}

# A simple forward‑chaining engine that evaluates all rules for a given exercise.
class SimpleRuleEngine:
    def __init__(self, rules_dict):
        self.rules_dict = rules_dict
    
    def evaluate(self, exercise: str, angles_dict: Dict) -> List[Dict]:
        if exercise not in self.rules_dict:
            return []
        triggered = []
        for rule in self.rules_dict[exercise]:
            try:
                if rule["condition"](angles_dict):
                    triggered.append({
                        "name": rule["name"],
                        "message": rule["message"],
                        "paper": rule["paper"],
                        "threshold": rule["threshold"]
                    })
            except:
                pass
        return triggered

# The main Knowledge Base class. It can be configured to include only general exercises
# or both general and rehabilitation exercises.
class KnowledgeBase:
    def __init__(self, include_rehab=False):
        self.include_rehab = include_rehab
        self.rules_dict = GENERAL_RULES.copy()
        if include_rehab:
            self.rules_dict.update(REHAB_RULES)
        self.engine = SimpleRuleEngine(self.rules_dict)
    
    def explain_rules(self, exercise: str = None):
        # Print all rules (or rules for a specific exercise) with their scientific justification.
        print("KNOWLEDGE BASE - RULES WITH SCIENTIFIC JUSTIFICATION")
        print(f"Current threshold mode: {'PRACTICAL (relaxed)' if USE_PRACTICAL_THRESHOLDS else 'STRICT (paper values)'}")
        exercises = [exercise] if exercise else self.rules_dict.keys()
        for ex in exercises:
            if ex not in self.rules_dict:
                continue
            print(f"\n[{ex.upper()}]")
            for rule in self.rules_dict[ex]:
                print(f"  Rule: {rule['name']}")
                print(f"    Threshold: {rule['threshold']}")
                print(f"    Message: {rule['message']}")
                print(f"    Source: {rule['paper']}")
                print()
    
    def analyze(self, exercise: str, angles_array: np.ndarray) -> Dict:
        # Given an exercise name and a vector of 12 joint angles, return a dictionary
        # with the analysis result (correct/incorrect, triggered rules, feedback, citations).
        if len(angles_array) != 12:
            raise ValueError("angles_array must have 12 elements")
        angles_dict = {ANGLE_NAMES[i]: angles_array[i] for i in range(12)}
        triggered = self.engine.evaluate(exercise, angles_dict)
        is_correct = len(triggered) == 0
        return {
            "exercise": exercise,
            "is_correct": is_correct,
            "errors": [t["name"] for t in triggered],
            "feedback": [t["message"] for t in triggered],
            "papers": [t["paper"] for t in triggered],
            "triggered_rules": triggered
        }

# I generate synthetic angle vectors for rehabilitation exercises to test the KBS.
# Each sample is either correct or contains a common error for that exercise.
def generate_synthetic_rehab_data(exercise, n_samples=200, seed=42):
    np.random.seed(seed)
    X = []
    y_true = []  # 1 = correct, 0 = incorrect
    normal = np.array([100, 100, 160, 160, 175, 45, 45, 160, 160, 90, 175, 90], dtype=np.float32)
    
    if exercise == "arm_circle":
        for _ in range(n_samples):
            angles = normal.copy()
            if np.random.rand() < 0.5:
                angles[5] = np.random.uniform(30, 59)
                angles[6] = np.random.uniform(30, 59)
                y_true.append(0)
            else:
                y_true.append(1)
            X.append(angles)
    elif exercise == "forward_lunge":
        for _ in range(n_samples):
            angles = normal.copy()
            error_type = np.random.choice(["none", "knee_past_toes", "back_lean"], p=[0.4, 0.3, 0.3])
            if error_type == "knee_past_toes":
                angles[0] = np.random.uniform(50, 79)
                angles[1] = np.random.uniform(50, 79)
                y_true.append(0)
            elif error_type == "back_lean":
                angles[4] = np.random.uniform(130, 159)
                y_true.append(0)
            else:
                y_true.append(1)
            X.append(angles)
    elif exercise == "high_knee_raise":
        for _ in range(n_samples):
            angles = normal.copy()
            if np.random.rand() < 0.5:
                angles[2] = np.random.uniform(50, 79)
                angles[3] = np.random.uniform(50, 79)
                y_true.append(0)
            else:
                y_true.append(1)
            X.append(angles)
    elif exercise == "hip_abduction":
        for _ in range(n_samples):
            angles = normal.copy()
            if np.random.rand() < 0.5:
                angles[2] = np.random.uniform(30, 69)
                y_true.append(0)
            else:
                y_true.append(1)
            X.append(angles)
    elif exercise == "leg_extension":
        for _ in range(n_samples):
            angles = normal.copy()
            if np.random.rand() < 0.5:
                angles[0] = np.random.uniform(151, 170)
                angles[1] = np.random.uniform(151, 170)
                y_true.append(0)
            else:
                y_true.append(1)
            X.append(angles)
    elif exercise == "shoulder_abduction":
        for _ in range(n_samples):
            angles = normal.copy()
            if np.random.rand() < 0.5:
                angles[5] = np.random.uniform(30, 79)
                angles[6] = np.random.uniform(30, 79)
                y_true.append(0)
            else:
                y_true.append(1)
            X.append(angles)
    elif exercise == "shoulder_external_rotation":
        for _ in range(n_samples):
            angles = normal.copy()
            if np.random.rand() < 0.5:
                angles[5] = np.random.uniform(10, 29)
                y_true.append(0)
            else:
                y_true.append(1)
            X.append(angles)
    elif exercise == "shoulder_flexion":
        for _ in range(n_samples):
            angles = normal.copy()
            if np.random.rand() < 0.5:
                angles[5] = np.random.uniform(100, 139)
                angles[6] = np.random.uniform(100, 139)
                y_true.append(0)
            else:
                y_true.append(1)
            X.append(angles)
    elif exercise == "side_step_squat":
        for _ in range(n_samples):
            angles = normal.copy()
            if np.random.rand() < 0.5:
                angles[0] = np.random.uniform(50, 69)
                angles[1] = np.random.uniform(50, 69)
                y_true.append(0)
            else:
                y_true.append(1)
            X.append(angles)
    elif exercise == "squat":
        for _ in range(n_samples):
            angles = normal.copy()
            if np.random.rand() < 0.5:
                angles[0] = np.random.uniform(40, 69)
                angles[1] = np.random.uniform(40, 69)
                y_true.append(0)
            else:
                y_true.append(1)
            X.append(angles)
    else:
        for _ in range(n_samples):
            X.append(normal.copy())
            y_true.append(1)
    return np.array(X), np.array(y_true)

def evaluate_rule_system(kb, X_test, y_test, exercise):
    from sklearn.metrics import accuracy_score
    y_pred = []
    for angles in X_test:
        result = kb.analyze(exercise, angles)
        y_pred.append(1 if result["is_correct"] else 0)
    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": acc}

def compare_with_mlp_for_all_rehab():
    # I compare the KBS with the trained MLP posture classifier on synthetic test data.
    try:
        import tensorflow as tf
        from tensorflow import keras
        model_path = os.path.join("models", "checkpoints", "posture_mediapipe", "mlp_posture_mediapipe_best.keras")
        if not os.path.exists(model_path):
            print("MLP model not found. Skipping comparison.")
            return
        mlp = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Could not load MLP model: {e}")
        return
    
    kb = KnowledgeBase(include_rehab=True)
    
    print("COMPARISON: Rule-based KBS vs MLP Posture Classifier")
    print("On ALL Rehabilitation Exercises (synthetic test sets)")
    print(f"{'Exercise':<25} {'KBS Acc':<10} {'MLP Acc':<10} {'Agreement':<10}")
    print("-"*60)
    
    for ex in REHAB_EXERCISES:
        X_test, y_test = generate_synthetic_rehab_data(ex, n_samples=200)
        metrics_kbs = evaluate_rule_system(kb, X_test, y_test, ex)
        y_pred_mlp = []
        for angles in X_test:
            prob = mlp.predict(angles.reshape(1, -1), verbose=0)[0]
            y_pred_mlp.append(1 if prob[1] > 0.5 else 0)
        from sklearn.metrics import accuracy_score
        acc_mlp = accuracy_score(y_test, y_pred_mlp)
        y_pred_kbs = [1 if kb.analyze(ex, a)["is_correct"] else 0 for a in X_test]
        agreement = np.mean(np.array(y_pred_mlp) == np.array(y_pred_kbs))
        print(f"{ex:<25} {metrics_kbs['accuracy']:<10.3f} {acc_mlp:<10.3f} {agreement:<10.1%}")
    return

if __name__ == "__main__":
    # A quick demonstration of the KBS: show rules for squat, compare with MLP,
    # and test a few individual angle vectors.
    print("[KBS] Knowledge Base System - demo and comparison")
    print(f"[KBS] Threshold mode: {'PRACTICAL (relaxed)' if USE_PRACTICAL_THRESHOLDS else 'STRICT (paper values)'}")
    print("[KBS] To change mode, set USE_PRACTICAL_THRESHOLDS at the top of this file.")
    print()
    
    kb = KnowledgeBase(include_rehab=False)
    kb.explain_rules("squat")
    
    compare_with_mlp_for_all_rehab()
    
    X_squat, _ = generate_synthetic_rehab_data("squat", n_samples=50)
    from collections import defaultdict
    rule_counts = defaultdict(int)
    for angles in X_squat:
        result = kb.analyze("squat", angles)
        for err in result["errors"]:
            rule_counts[err] += 1
    print(f"\n[KBS] Coverage for squat (general): {len(rule_counts)}/4 rules triggered")
    for r, c in rule_counts.items():
        print(f"  {r}: {c} times")

    print("\n[KBS] Individual analyses:")
    
    correct_angles = np.array([100, 100, 160, 160, 175, 45, 45, 160, 160, 90, 175, 90])
    result = kb.analyze("squat", correct_angles)
    print(f"\nSquat (correct): is_correct={result['is_correct']}, feedback={result['feedback']}")
    
    knee_deep = np.array([55, 58, 160, 160, 175, 45, 45, 160, 160, 90, 175, 90])
    result = kb.analyze("squat", knee_deep)
    print(f"Squat (knee deep): is_correct={result['is_correct']}, errors={result['errors']}")
    
    pushup_bad = np.array([100, 100, 160, 160, 140, 45, 45, 90, 90, 90, 140, 90])
    result = kb.analyze("push_up", pushup_bad)
    print(f"Push-up (back sag): is_correct={result['is_correct']}, feedback={result['feedback']}")
    
    kb_rehab = KnowledgeBase(include_rehab=True)
    lunge_correct = np.array([100, 100, 160, 160, 175, 45, 45, 160, 160, 90, 175, 90])
    result = kb_rehab.analyze("forward_lunge", lunge_correct)
    print(f"\nForward lunge (rehab, correct): is_correct={result['is_correct']}, feedback={result['feedback']}")
    
    lunge_bad = np.array([75, 75, 160, 160, 175, 45, 45, 160, 160, 90, 175, 90])
    result = kb_rehab.analyze("forward_lunge", lunge_bad)
    print(f"Forward lunge (rehab, knee past toes): is_correct={result['is_correct']}, errors={result['errors']}")
    
    print("\n[Demo complete]")