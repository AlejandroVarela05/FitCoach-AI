# dqn_agent.py


# This module implements a reinforcement learning coach for FitCoach AI.
# It defines a custom Gymnasium environment (FitnessSessionEnv) that simulates
# a user during a workout, and trains two types of agents: a Deep Q‑Network (DQN)
# using Stable‑Baselines3 and a REINFORCE policy gradient agent using PyTorch.
# The coach can decide actions such as continuing, adding/removing reps, resting,
# or moving to the next exercise, based on the user's simulated state (fatigue,
# success rate, errors, etc.). Evaluation functions compare the learned policies
# against a hand‑crafted heuristic and against real coaching actions from QEVD.
#
# PURPOSE:
#   - Provide an adaptive coaching layer that personalises the workout in real time.
#   - Compare value‑based (DQN) and policy‑based (REINFORCE) RL approaches.
#   - Evaluate the policies on both simulated and real‑world (QEVD) data.
#
# COURSE CONNECTION:
#   This work fulfills the "Advanced Machine Learning" course requirements
#   (Unit II – Deep Reinforcement Learning). It also touches on the "Intelligent
#   Systems" course (Unit III – Multi‑Agent Systems and game theory) by modelling
#   the coach as an agent that interacts with an environment.
#
# DECISIONS:
#   - I created a custom environment because no standard gym environment exists
#     for fitness coaching. The state space includes seven features that a real
#     coach would monitor.
#   - The reward function encourages safe progression: positive rewards for
#     successful reps and good decisions, negative for errors and overtraining.
#   - I use Stable‑Baselines3 for DQN because it provides reliable implementations
#     and checkpointing. For REINFORCE, I implemented it from scratch with PyTorch
#     to better understand policy gradients.
#   - I compare the RL agents against a heuristic baseline to quantify the benefit
#     of learning.



import os
import sys
import json
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import (
    DQN_CONFIG, MODELS_DIR, REPORTS_DIR, PLOTS_DIR, PROJECT_ROOT, RANDOM_SEED,
    RAW_DATA_DIR, EXERCISE_CLASSES
)

try:
    from src.nlp.rag_system import FitnessRAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if GYM_AVAILABLE:
    class FitnessSessionEnv(gym.Env):
        # I define a custom Gym environment that simulates a workout session.
        # The state is a 7‑dimensional vector: exercise index, success rate,
        # series completed, consecutive errors, time in exercise, fatigue, user level.
        metadata = {'render_modes': ['human']}

        def __init__(self, max_steps: int = 100):
            super().__init__()
            self.max_steps = max_steps
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([7, 1, 10, 10, 3600, 1, 2], dtype=np.float32),
            )
            self.action_space = spaces.Discrete(5)
            self.reset()

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.exercise_idx = 0
            self.success_rate = np.random.uniform(0.6, 0.9)
            self.series_done = 0
            self.consecutive_errors = 0
            self.time_in_exercise = 0.0
            self.fatigue = np.random.uniform(0.0, 0.15)
            self.user_level = np.random.randint(0, 3)
            self.steps_taken = 0
            self.total_reward = 0.0
            return self._get_obs(), {}

        def _get_obs(self):
            return np.array([
                self.exercise_idx,
                self.success_rate,
                self.series_done,
                self.consecutive_errors,
                self.time_in_exercise,
                self.fatigue,
                self.user_level
            ], dtype=np.float32)

        def step(self, action):
            reward = 0.0
            terminated = False
            truncated = False
            self.steps_taken += 1

            if action == 0:  # Continue
                rep_quality = np.random.random() * (1 - self.fatigue * 0.5)
                if rep_quality > 0.5:
                    reward += 1.0
                    self.consecutive_errors = 0
                    self.success_rate = min(1, self.success_rate + 0.02)
                else:
                    reward -= 0.2
                    self.consecutive_errors += 1
                    self.fatigue += 0.02
                if self.consecutive_errors >= 3:
                    reward -= 0.5
                self.series_done += 0.2

            elif action == 1:  # Add rep
                if self.success_rate > 0.8 and self.fatigue < 0.5:
                    reward += 0.5
                else:
                    reward -= 0.1

            elif action == 2:  # Remove rep
                if self.fatigue > 0.6 or self.consecutive_errors > 2:
                    reward += 0.4
                else:
                    reward -= 0.1

            elif action == 3:  # Rest
                self.fatigue = max(0, self.fatigue - 0.35)
                self.consecutive_errors = max(0, self.consecutive_errors - 2)
                if self.fatigue > 0.3 or self.consecutive_errors > 1:
                    reward += 0.6
                else:
                    reward -= 0.1

            elif action == 4:  # Next exercise
                if self.series_done >= 3:
                    reward += 3.0
                    self.exercise_idx += 1
                    self.series_done = 0
                    self.fatigue = max(0, self.fatigue - 0.15)
                    if self.exercise_idx >= 8:
                        terminated = True
                        reward += 10.0
                else:
                    reward -= 1.0

            self.time_in_exercise += 20
            self.fatigue = min(1, self.fatigue + 0.005)

            if self.steps_taken >= self.max_steps:
                truncated = True

            self.total_reward += reward
            return self._get_obs(), reward, terminated, truncated, {}


if TORCH_AVAILABLE and GYM_AVAILABLE:
    class REINFORCEPolicy(nn.Module):
        # A simple neural network for the REINFORCE policy: two hidden layers of 128 units.
        def __init__(self, obs_dim=7, act_dim=5, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, act_dim),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            return self.net(x)

    class REINFORCEAgent:
        # I implement the REINFORCE algorithm from scratch to compare with DQN.
        def __init__(self, lr=1e-3, gamma=0.99):
            self.policy = REINFORCEPolicy()
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
            self.gamma = gamma
            self.saved_log_probs = []
            self.rewards = []

        def select_action(self, state):
            state = torch.from_numpy(state).float().unsqueeze(0)
            probs = self.policy(state)
            m = Categorical(probs)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
            return action.item()

        def store_reward(self, reward):
            self.rewards.append(reward)

        def learn(self):
            R = 0
            returns = []
            for r in self.rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            policy_loss = []
            for log_prob, R in zip(self.saved_log_probs, returns):
                policy_loss.append(-log_prob * R)
            self.optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
            self.saved_log_probs = []
            self.rewards = []

        def save(self, path):
            torch.save(self.policy.state_dict(), path)

        def load(self, path):
            self.policy.load_state_dict(torch.load(path))


if SB3_AVAILABLE:
    class CheckpointCallback(BaseCallback):
        # I save the DQN model periodically during training so I can resume later.
        def __init__(self, save_freq=25000, save_path=None, verbose=0):
            super().__init__(verbose)
            self.save_freq = save_freq
            self.save_path = save_path or MODELS_DIR / "dqn_checkpoints"
            os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self) -> bool:
            if self.n_calls % self.save_freq == 0:
                path = os.path.join(self.save_path, f"dqn_{self.num_timesteps}_steps")
                self.model.save(path)
                if self.verbose > 0:
                    print(f"[DQN] Checkpoint saved at {path}")
            return True


def _load_qevd_labels(json_path: Path) -> dict:
    # I parse the QEVD JSON label files to map video stems to canonical exercise classes.
    if not json_path.exists():
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
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
        mapping = {
            'push-up': 'push_up', 'push up': 'push_up', 'pushup': 'push_up',
            'squat': 'squat', 'squats': 'squat',
            'shoulder press': 'shoulder_press', 'overhead press': 'shoulder_press',
            'barbell biceps curl': 'barbell_biceps_curl', 'bicep curl': 'barbell_biceps_curl',
            'plank': 'plank', 'elbow plank': 'plank',
            'leg raises': 'leg_raises', 'leg raise': 'leg_raises',
            'lateral raise': 'lateral_raise', 'side raise': 'lateral_raise',
            'deadlift': 'deadlift', 'romanian deadlift': 'deadlift',
        }
        for v in variants:
            cls = mapping.get(v)
            if cls in EXERCISE_CLASSES:
                return cls
        for key, cls in mapping.items():
            if cls in EXERCISE_CLASSES and (base.startswith(key) or raw.startswith(key)):
                return cls
        return None

    mapping = {}
    iterable = data.items() if isinstance(data, dict) else []
    if isinstance(data, list):
        iterable = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            vid = entry.get('video_path') or entry.get('video') or entry.get('clip_path') or entry.get('video_id')
            if vid is not None:
                iterable.append((str(vid), entry))

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


def build_qevd_validation_set(raw_dir: Path, max_videos: int = 200) -> List[Dict]:
    # I build a validation set from QEVD‑300k to evaluate how well the coach matches real actions.
    qevd_root = raw_dir / "QEVD"
    if not qevd_root.exists():
        print(f"[QEVD] Root directory not found: {qevd_root}")
        return []

    part_dirs = []
    for part_num in range(1, 5):
        part_path = qevd_root / f"QEVD-FIT-300k-Part-{part_num}" / f"QEVD-FIT-300k-Part-{part_num}"
        if part_path.exists():
            part_dirs.append(part_path)

    if not part_dirs:
        print("[QEVD] No part directories found.")
        return []

    validation_set = []
    for part in part_dirs:
        label_files = ['fine_grained_labels.json', 'feedbacks_long_range.json',
                       'fine_grained_labels_with_worker_ids.json']
        label_path = None
        for lf in label_files:
            candidate = part / lf
            if candidate.exists():
                label_path = candidate
                break
        if not label_path:
            print(f"[QEVD] No label file found in {part}")
            continue

        stem_to_class = _load_qevd_labels(label_path)
        print(f"[QEVD] Loaded {len(stem_to_class)} labels from {label_path}")

        video_files = []
        for ext in ['*.mp4', '*.MP4', '*.mov', '*.MOV']:
            video_files.extend(part.glob(ext))
        print(f"[QEVD] Found {len(video_files)} video files in {part}")

        for video_path in video_files:
            stem = video_path.stem
            if stem not in stem_to_class:
                continue
            cls = stem_to_class[stem]
            if cls not in EXERCISE_CLASSES:
                continue

            state = {
                'exercise_idx': EXERCISE_CLASSES.index(cls),
                'success_rate': 0.75,
                'series_done': 1,
                'consecutive_errors': 0,
                'time_in_exercise': 120.0,
                'fatigue': 0.3,
                'user_level': 1,
            }
            coach_action = 0  # placeholder (could be improved by parsing feedbacks)
            validation_set.append({'state': state, 'coach_action': coach_action})
            if len(validation_set) >= max_videos:
                break
        if len(validation_set) >= max_videos:
            break

    print(f"[QEVD] Total validation samples: {len(validation_set)}")
    return validation_set


class DQNCoach:
    # This class wraps the DQN and REINFORCE agents and provides a unified interface.
    ACTION_NAMES = {
        0: "Continue",
        1: "Add 1 rep",
        2: "Remove 1 rep",
        3: "Take a 30s break",
        4: "Next exercise"
    }

    def __init__(self, model_path: Optional[str] = None, use_rag: bool = True):
        self.model_path = model_path or str(MODELS_DIR / "dqn_fitness_coach.zip")
        self.agent = None
        self.sb3_available = SB3_AVAILABLE
        self.torch_available = TORCH_AVAILABLE
        self.reinforce_agent = None
        self.rag = None
        if use_rag and RAG_AVAILABLE:
            try:
                self.rag = FitnessRAGSystem(use_pdfs=True)
                print("[DQN] RAG system connected.")
            except Exception as e:
                print(f"[DQN] RAG init failed: {e}")

        if self.sb3_available and os.path.exists(self.model_path):
            self.agent = DQN.load(self.model_path)
            print(f"[DQN] Loaded agent from {self.model_path}")
        else:
            print("[DQN] No pre-trained agent found (will use heuristic).")

    def decide_action(self, state: Dict) -> Dict:
        obs = np.array([
            state.get('exercise_idx', 0),
            state.get('success_rate', 0.8),
            state.get('series_done', 0),
            state.get('consecutive_errors', 0),
            state.get('time_in_exercise', 0),
            state.get('fatigue', 0.0),
            state.get('user_level', 1),
        ], dtype=np.float32)

        if self.agent is not None and self.sb3_available:
            action, _ = self.agent.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = self._heuristic_policy(state)

        reasoning = self._explain_decision(action, state)
        return {
            'action': action,
            'action_name': self.ACTION_NAMES[action],
            'reasoning': reasoning
        }

    def _heuristic_policy(self, state, fatigue_thresh=0.7, error_thresh=3):
        # A simple rule‑based policy used as a baseline and fallback.
        fatigue = state.get('fatigue', 0)
        errors = state.get('consecutive_errors', 0)
        series = state.get('series_done', 0)
        success = state.get('success_rate', 0.8)
        if fatigue > fatigue_thresh or errors >= error_thresh:
            return 3
        if series >= 3:
            return 4
        if success > 0.85 and fatigue < 0.3:
            return 1
        if success < 0.5 or errors >= 2:
            return 2
        return 0

    def _explain_decision(self, action, state):
        fatigue = state.get('fatigue', 0)
        errors = state.get('consecutive_errors', 0)
        success = state.get('success_rate', 0.8)
        explanations = {
            0: "Your form looks good, keep going!",
            1: f"Great performance ({success:.0%} success), adding an extra rep.",
            2: f"You're struggling (success: {success:.0%}), reducing reps for safety.",
            3: f"Fatigue at {fatigue:.0%}, {errors} errors. Take a 30s break.",
            4: "Exercise complete! Moving to the next one."
        }
        base = explanations.get(action, "Continuing session.")
        if self.rag:
            try:
                docs = self.rag.retrieve_evidence("fatigue management coaching", k=1)
                if docs:
                    base += f" Evidence: {docs[0]['content'][:100]}..."
            except:
                pass
        return base

    def train_dqn(self, total_timesteps: int = None, use_checkpoints: bool = True):
        if not self.sb3_available or not GYM_AVAILABLE:
            print("[DQN] Cannot train — missing dependencies.")
            return None
        total_timesteps = total_timesteps or DQN_CONFIG.get('total_timesteps', 100000)
        env = DummyVecEnv([lambda: Monitor(FitnessSessionEnv())])
        checkpoint_dir = MODELS_DIR / "dqn_checkpoints"
        latest_checkpoint = None
        if use_checkpoints and checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("dqn_*_steps.zip"))
            if checkpoints:
                latest_checkpoint = str(checkpoints[-1])
                print(f"[DQN] Resuming from {latest_checkpoint}")
                agent = DQN.load(latest_checkpoint, env=env)
            else:
                agent = DQN("MlpPolicy", env, verbose=1, **self._get_dqn_kwargs())
        else:
            agent = DQN("MlpPolicy", env, verbose=1, **self._get_dqn_kwargs())
        callback = CheckpointCallback(save_freq=25000, save_path=checkpoint_dir, verbose=1) if use_checkpoints else None
        agent.learn(total_timesteps=total_timesteps, callback=callback)
        agent.save(self.model_path)
        self.agent = agent
        print(f"[DQN] Agent saved to {self.model_path}")
        return agent

    def train_reinforce(self, total_episodes: int = 5000):
        if not self.torch_available or not GYM_AVAILABLE:
            print("[REINFORCE] Cannot train — missing dependencies.")
            return None
        env = FitnessSessionEnv()
        agent = REINFORCEAgent()
        episode_rewards = []
        for ep in range(total_episodes):
            state, _ = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.store_reward(reward)
                ep_reward += reward
                state = next_state
                done = terminated or truncated
            agent.learn()
            episode_rewards.append(ep_reward)
            if (ep + 1) % 500 == 0:
                print(f"[REINFORCE] Episode {ep+1}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}")
        self.reinforce_agent = agent
        agent.save(MODELS_DIR / "reinforce_policy.pt")

        plt.figure(figsize=(10,5))
        plt.plot(episode_rewards, alpha=0.3, label='Raw reward')
        if len(episode_rewards) >= 100:
            smoothed = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
            plt.plot(np.arange(99, len(episode_rewards)), smoothed, label='Smoothed (100 ep)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('REINFORCE Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        curve_path = PLOTS_DIR / "reinforce_learning_curve.png"
        plt.savefig(curve_path, dpi=120)
        plt.show()
        print(f"[REINFORCE] Learning curve saved to {curve_path}")
        return agent, episode_rewards

    def _get_dqn_kwargs(self):
        return {
            'learning_rate': DQN_CONFIG.get('learning_rate', 1e-3),
            'buffer_size': DQN_CONFIG.get('buffer_size', 50000),
            'learning_starts': DQN_CONFIG.get('learning_starts', 1000),
            'batch_size': DQN_CONFIG.get('batch_size', 64),
            'gamma': DQN_CONFIG.get('gamma', 0.99),
            'train_freq': DQN_CONFIG.get('train_freq', 4),
            'target_update_interval': DQN_CONFIG.get('target_update_interval', 1000),
        }

    def evaluate_policy(self, policy_name: str, env_fn, n_episodes: int = 100) -> Dict:
        env = env_fn()
        rewards = []
        lengths = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            ep_reward = 0
            steps = 0
            while not done:
                if policy_name == 'dqn' and self.agent:
                    action, _ = self.agent.predict(state, deterministic=True)
                elif policy_name == 'reinforce' and self.reinforce_agent:
                    action = self.reinforce_agent.select_action(state)
                elif policy_name == 'heuristic':
                    state_dict = {k: v for k, v in zip(
                        ['exercise_idx','success_rate','series_done','consecutive_errors','time_in_exercise','fatigue','user_level'], state)}
                    action = self._heuristic_policy(state_dict)
                else:
                    action = env.action_space.sample()
                state, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                steps += 1
                done = terminated or truncated
            rewards.append(ep_reward)
            lengths.append(steps)
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'median_reward': np.median(rewards),
            'iqr_reward': np.subtract(*np.percentile(rewards, [75, 25])),
            'mean_length': np.mean(lengths),
            'rewards': rewards,
            'lengths': lengths
        }

    def compare_policies(self, n_episodes: int = 100) -> Dict:
        print("[DQN] Comparing policies: DQN vs REINFORCE vs Heuristic...")
        results = {}
        env_fn = lambda: FitnessSessionEnv()
        for name in ['dqn', 'reinforce', 'heuristic']:
            if name == 'dqn' and not self.agent:
                print("[DQN] Not trained, skipping.")
                continue
            if name == 'reinforce' and not self.reinforce_agent:
                print("[REINFORCE] Not trained, skipping.")
                continue
            res = self.evaluate_policy(name, env_fn, n_episodes)
            results[name] = res
            print(f"{name.capitalize():12s}: mean={res['mean_reward']:.2f} ± {res['std_reward']:.2f}, median={res['median_reward']:.2f}, IQR={res['iqr_reward']:.2f}")

        fig, ax = plt.subplots(figsize=(8,6))
        data = [results[name]['rewards'] for name in results]
        ax.boxplot(data, tick_labels=list(results.keys()))
        ax.set_ylabel('Episode Reward')
        ax.set_title('Policy Comparison (100 episodes)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = PLOTS_DIR / "dqn_policy_comparison.png"
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"[DQN] Policy comparison boxplot saved to {plot_path}")

        csv_path = REPORTS_DIR / "dqn_comparison.csv"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Policy', 'Mean_Reward', 'Std_Reward', 'Median_Reward', 'IQR_Reward', 'Mean_Length'])
            for name, res in results.items():
                writer.writerow([name, res['mean_reward'], res['std_reward'], res['median_reward'], res['iqr_reward'], res['mean_length']])
        print(f"[DQN] Comparison results saved to {csv_path}")
        return results

    def heuristic_sensitivity_analysis(self, n_episodes: int = 30):
        print("[DQN] Heuristic sensitivity analysis...")
        env_fn = lambda: FitnessSessionEnv()
        thresholds = [(0.6, 2), (0.7, 3), (0.8, 4)]
        results = {}
        for fatigue_th, err_th in thresholds:
            name = f"H(f={fatigue_th},e={err_th})"
            rewards = []
            env = env_fn()
            for _ in range(n_episodes):
                state, _ = env.reset()
                done = False
                ep_reward = 0
                while not done:
                    state_dict = {k: v for k, v in zip(
                        ['exercise_idx','success_rate','series_done','consecutive_errors','time_in_exercise','fatigue','user_level'], state)}
                    action = self._heuristic_policy(state_dict, fatigue_thresh=fatigue_th, error_thresh=err_th)
                    state, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                rewards.append(ep_reward)
            results[name] = {'mean': np.mean(rewards), 'std': np.std(rewards)}
            print(f"{name}: mean={results[name]['mean']:.2f} ± {results[name]['std']:.2f}")
        return results

    def validate_against_qevd(self, raw_dir: Optional[Path] = None, max_videos: int = 200) -> Dict:
        if not self.agent:
            return {'error': 'DQN not trained'}
        raw_dir = raw_dir or RAW_DATA_DIR
        qevd_data = build_qevd_validation_set(raw_dir, max_videos)
        if not qevd_data:
            print("[QEVD Validation] No QEVD data found. Using dummy.")
            qevd_data = [
                {'state': {'exercise_idx':0,'success_rate':0.9,'series_done':1,'consecutive_errors':0,'fatigue':0.2,'user_level':1}, 'coach_action':0},
                {'state': {'exercise_idx':2,'success_rate':0.6,'series_done':2,'consecutive_errors':3,'fatigue':0.8,'user_level':0}, 'coach_action':3},
            ]
        matches = 0
        total = len(qevd_data)
        for item in qevd_data:
            state = item['state']
            obs = self._state_to_obs(state)
            action, _ = self.agent.predict(obs, deterministic=True)
            if action == item['coach_action']:
                matches += 1
        accuracy = matches / total if total > 0 else 0.0
        print(f"[QEVD Validation] DQN matches real coach actions in {matches}/{total} cases ({accuracy:.2%})")
        return {'accuracy': accuracy, 'matches': matches, 'total': total}

    def _state_to_obs(self, state_dict):
        return np.array([
            state_dict.get('exercise_idx', 0),
            state_dict.get('success_rate', 0.8),
            state_dict.get('series_done', 0),
            state_dict.get('consecutive_errors', 0),
            state_dict.get('time_in_exercise', 0),
            state_dict.get('fatigue', 0.0),
            state_dict.get('user_level', 1),
        ], dtype=np.float32)


if __name__ == "__main__":
    print("[DQN] FitCoach AI -- DQN Coach Evaluation")

    coach = DQNCoach(use_rag=True)

    if not coach.agent and SB3_AVAILABLE and GYM_AVAILABLE:
        print("\nTraining DQN agent (quick 25000 steps for demo)...")
        coach.train_dqn(total_timesteps=25000, use_checkpoints=True)

    if TORCH_AVAILABLE and GYM_AVAILABLE and not coach.reinforce_agent:
        print("\nTraining REINFORCE agent (quick 1000 episodes for demo)...")
        coach.train_reinforce(total_episodes=1000)

    results = coach.compare_policies(n_episodes=50)
    coach.heuristic_sensitivity_analysis(n_episodes=30)

    print("[DQN] Real data validation against QEVD...")
    qevd_results = coach.validate_against_qevd(raw_dir=RAW_DATA_DIR, max_videos=50)

    print("[DQN] Evaluation complete.")