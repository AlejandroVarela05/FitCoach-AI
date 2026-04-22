# simulated_annealing.py


# This module implements optimisation algorithms for generating a balanced weekly
# exercise routine. It includes Simulated Annealing (SA), Tabu Search (TS), and
# Hill Climbing (HC). The cost function penalises consecutive muscle overlap,
# muscle imbalance, and insufficient rest days.
#
# PURPOSE:
#   - Find a weekly schedule of exercises that minimises the defined cost.
#   - Compare SA against TS and HC with statistical significance tests.
#   - Perform hyperparameter sensitivity analysis for SA.
#   - Visualise convergence curves and boxplots of final costs.
#
# COURSE CONNECTION:
#   This module directly applies concepts from "Intelligent Systems" (Unit II –
#   search in complex environments) and "Advanced Machine Learning" (Unit III –
#   metaheuristic methods). The Simulated Annealing algorithm was covered in
#   detail as a probabilistic technique for escaping local optima.
#
# DECISIONS:
#   - I chose SA as the main optimiser because it balances exploration and
#     exploitation via the temperature schedule.
#   - The cost function includes three weighted components based on exercise
#     science literature (recovery, balance, rest).
#   - I run 30 independent trials and use the Wilcoxon signed‑rank test to
#     confirm that SA significantly outperforms TS and HC.



import numpy as np
import random
import copy
import time
import json
import csv
from collections import deque
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import (
    EXERCISE_CLASSES, EXERCISE_MUSCLE_GROUPS, SA_CONFIG,
    PROJECT_ROOT, PLOTS_DIR, REPORTS_DIR
)

# I try to import the RAG system. If it is not available, I set a flag to skip it.
try:
    from src.nlp.rag_system import FitnessRAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class RoutineOptimizer:
    # This class contains the problem definition and the optimisation algorithms.
    # It generates an initial random schedule, evaluates its cost, and applies
    # neighbourhood operators to improve it.

    DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday', 'Saturday', 'Sunday']

    def __init__(self, user_profile=None):
        # I store the user's training preferences. If none are given, I use a default profile.
        self.profile = user_profile or {
            'days_per_week': 4,
            'level': 'intermediate',
            'goal': 'general_fitness',
            'exercises_per_day': 3,
        }
        self.all_exercises = EXERCISE_CLASSES
        self.muscle_groups = EXERCISE_MUSCLE_GROUPS
        # I count how many times the cost function is evaluated. This helps measure efficiency.
        self.energy_evaluations = 0

    def generate_initial_solution(self):
        # I create a random weekly schedule that respects the user's available training days.
        days_per_week = self.profile['days_per_week']
        exercises_per_day = self.profile.get('exercises_per_day', 3)
        routine = {}
        rest_days = 7 - days_per_week
        # I spread the rest days as evenly as possible across the week.
        if rest_days >= 1:
            rest_indices = set(np.linspace(0, 6, rest_days + 2, dtype=int)[1:-1])
            if len(rest_indices) < rest_days:
                all_days = set(range(7))
                while len(rest_indices) < rest_days:
                    rest_indices.add(random.choice(list(all_days - rest_indices)))
        else:
            rest_indices = set()

        for i, day in enumerate(self.DAYS):
            if i in rest_indices:
                # I mark this day as a rest day (empty exercise list).
                routine[day] = []
            else:
                # I randomly select exercises for this training day.
                exercises = random.sample(self.all_exercises, min(exercises_per_day, len(self.all_exercises)))
                routine[day] = exercises
        return routine

    def calculate_energy(self, routine):
        # The energy (cost) function measures how bad a schedule is. Lower is better.
        self.energy_evaluations += 1
        energy = 0.0
        training_days = [(day, exs) for day, exs in routine.items() if exs]

        # Penalty 1: Consecutive days working the same muscle groups (bad for recovery).
        for i in range(len(training_days) - 1):
            _, exs_today = training_days[i]
            _, exs_tomorrow = training_days[i + 1]
            muscles_today = set()
            for ex in exs_today:
                muscles_today.update(self.muscle_groups.get(ex, []))
            muscles_tomorrow = set()
            for ex in exs_tomorrow:
                muscles_tomorrow.update(self.muscle_groups.get(ex, []))
            overlap = len(muscles_today & muscles_tomorrow)
            energy += overlap * 3.0

        # Penalty 2: Muscle group imbalance (standard deviation of weekly frequency).
        muscle_count = {}
        for day, exs in routine.items():
            for ex in exs:
                for muscle in self.muscle_groups.get(ex, []):
                    muscle_count[muscle] = muscle_count.get(muscle, 0) + 1
        if muscle_count:
            counts = list(muscle_count.values())
            energy += np.std(counts) * 2.0

        # Penalty 3: Excessive repetition of the same exercise (more than 3 times per week).
        exercise_freq = {}
        for day, exs in routine.items():
            for ex in exs:
                exercise_freq[ex] = exercise_freq.get(ex, 0) + 1
        for ex, freq in exercise_freq.items():
            if freq > 3:
                energy += (freq - 3) * 2.0

        # Penalty 4: Not enough rest days based on fitness level.
        level = self.profile.get('level', 'intermediate')
        rest_count = sum(1 for exs in routine.values() if not exs)
        if level == 'beginner' and rest_count < 3:
            energy += (3 - rest_count) * 4.0
        elif level == 'intermediate' and rest_count < 2:
            energy += (2 - rest_count) * 3.0

        return energy

    def swap_operator(self, routine):
        # This operator swaps two exercises between two different training days.
        new_routine = copy.deepcopy(routine)
        training_days = [d for d, exs in new_routine.items() if exs]
        if len(training_days) < 2:
            return new_routine
        day1, day2 = random.sample(training_days, 2)
        if new_routine[day1] and new_routine[day2]:
            idx1 = random.randint(0, len(new_routine[day1]) - 1)
            idx2 = random.randint(0, len(new_routine[day2]) - 1)
            new_routine[day1][idx1], new_routine[day2][idx2] = new_routine[day2][idx2], new_routine[day1][idx1]
        return new_routine

    def insertion_operator(self, routine):
        # This operator replaces one exercise on a training day with a different exercise.
        new_routine = copy.deepcopy(routine)
        training_days = [d for d, exs in new_routine.items() if exs]
        if not training_days:
            return new_routine
        day = random.choice(training_days)
        if new_routine[day]:
            idx = random.randint(0, len(new_routine[day]) - 1)
            current = set(new_routine[day])
            available = [ex for ex in self.all_exercises if ex not in current]
            if available:
                new_routine[day][idx] = random.choice(available)
        return new_routine

    # Simulated Annealing uses a temperature schedule to sometimes accept worse solutions.
    def simulated_annealing(self, T0=None, Tf=None, alpha=None, max_iter=None, verbose=False):
        T0 = T0 or SA_CONFIG['T0']
        Tf = Tf or SA_CONFIG['Tf']
        alpha = alpha or SA_CONFIG['alpha']
        max_iter = max_iter or SA_CONFIG['max_iter']

        self.energy_evaluations = 0
        current = self.generate_initial_solution()
        current_energy = self.calculate_energy(current)
        best = copy.deepcopy(current)
        best_energy = current_energy
        T = T0
        total_iterations = 0
        history = [current_energy]
        start_time = time.time()

        while T > Tf:
            for _ in range(max_iter):
                total_iterations += 1
                # I choose randomly between swap and insertion operators.
                if random.random() < 0.5:
                    neighbour = self.swap_operator(current)
                else:
                    neighbour = self.insertion_operator(current)

                neighbour_energy = self.calculate_energy(neighbour)
                delta_E = neighbour_energy - current_energy

                # I accept better solutions, or worse ones with probability exp(-delta_E / T).
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    current = neighbour
                    current_energy = neighbour_energy
                    if current_energy < best_energy:
                        best = copy.deepcopy(current)
                        best_energy = current_energy

            # I cool down the temperature.
            T *= alpha
            history.append(current_energy)
            if verbose and len(history) % 10 == 0:
                print(f"  [SA] temp={T:.3f}, current={current_energy:.2f}, best={best_energy:.2f}")

        elapsed = time.time() - start_time
        return best, best_energy, history, total_iterations, elapsed, self.energy_evaluations

    # Tabu Search uses a short‑term memory (tabu list) to avoid cycling.
    def tabu_search(self, tabu_tenure=10, max_iter=5000, verbose=False):
        self.energy_evaluations = 0
        current = self.generate_initial_solution()
        current_energy = self.calculate_energy(current)
        best = copy.deepcopy(current)
        best_energy = current_energy
        tabu_list = deque(maxlen=tabu_tenure)
        history = [current_energy]
        start_time = time.time()
        total_iterations = 0

        for it in range(max_iter):
            total_iterations += 1
            neighbors = []
            training_days = [d for d, exs in current.items() if exs]

            # I generate all possible swap neighbours.
            if len(training_days) >= 2:
                for i in range(len(training_days)):
                    for j in range(i + 1, len(training_days)):
                        day1, day2 = training_days[i], training_days[j]
                        if current[day1] and current[day2]:
                            for idx1 in range(len(current[day1])):
                                for idx2 in range(len(current[day2])):
                                    neighbor = copy.deepcopy(current)
                                    neighbor[day1][idx1], neighbor[day2][idx2] = \
                                        neighbor[day2][idx2], neighbor[day1][idx1]
                                    cost = self.calculate_energy(neighbor)
                                    move = ('swap', day1, day2, idx1, idx2)
                                    neighbors.append((cost, neighbor, move))

            # I generate insertion neighbours (limit to 5 per day for efficiency).
            for day in training_days:
                for idx in range(len(current[day])):
                    current_set = set(current[day])
                    available = [ex for ex in self.all_exercises if ex not in current_set]
                    for new_ex in available[:5]:
                        neighbor = copy.deepcopy(current)
                        neighbor[day][idx] = new_ex
                        cost = self.calculate_energy(neighbor)
                        move = ('insert', day, idx, new_ex)
                        neighbors.append((cost, neighbor, move))

            if not neighbors:
                continue

            # I sort neighbours by cost and pick the best one that is not tabu.
            neighbors.sort(key=lambda x: x[0])
            selected = None
            for cost, neighbor, move in neighbors:
                if move not in tabu_list:
                    selected = (cost, neighbor, move)
                    break

            # If all moves are tabu, I pick the best one anyway (aspiration criterion).
            if selected is None:
                selected = neighbors[0]

            cost, neighbor, move = selected
            current = neighbor
            current_energy = cost
            tabu_list.append(move)
            history.append(current_energy)

            if current_energy < best_energy:
                best = copy.deepcopy(current)
                best_energy = current_energy

            if verbose and it % 500 == 0:
                print(f"  [TS] iter={it}, current={current_energy:.2f}, best={best_energy:.2f}")

        elapsed = time.time() - start_time
        return best, best_energy, history, total_iterations, elapsed, self.energy_evaluations

    # Hill Climbing only accepts better solutions. I use random restarts to escape local optima.
    def hill_climbing(self, max_restarts=10, max_iter_per_restart=500):
        self.energy_evaluations = 0
        best_overall = None
        best_overall_energy = float('inf')
        history = []
        start_time = time.time()
        total_iterations = 0

        for restart in range(max_restarts):
            current = self.generate_initial_solution()
            current_energy = self.calculate_energy(current)
            for _ in range(max_iter_per_restart):
                total_iterations += 1
                if random.random() < 0.5:
                    neighbour = self.swap_operator(current)
                else:
                    neighbour = self.insertion_operator(current)
                neighbour_energy = self.calculate_energy(neighbour)
                # I only accept moves that strictly improve the cost.
                if neighbour_energy < current_energy:
                    current = neighbour
                    current_energy = neighbour_energy
                history.append(current_energy)
            if current_energy < best_overall_energy:
                best_overall = copy.deepcopy(current)
                best_overall_energy = current_energy

        elapsed = time.time() - start_time
        return best_overall, best_overall_energy, history, total_iterations, elapsed, self.energy_evaluations

    def format_routine(self, routine):
        # I produce a human‑readable string of the weekly schedule.
        lines = ["\n  Weekly Routine:"]
        for day in self.DAYS:
            exercises = routine.get(day, [])
            if not exercises:
                lines.append(f"  {day:12s}:  REST")
            else:
                ex_list = ", ".join([ex.replace("_", " ").title() for ex in exercises])
                lines.append(f"  {day:12s}:  {ex_list}")
        return "\n".join(lines)


# I compare SA, TS, and HC over multiple independent runs and test statistical significance.
def compare_algorithms_statistical(optimizer, n_runs=30):
    print(f"[Comparison] Statistical comparison: SA vs Tabu Search vs Hill Climbing ({n_runs} runs)")

    algorithms = {
        'SA': lambda: optimizer.simulated_annealing(verbose=False),
        'TS': lambda: optimizer.tabu_search(max_iter=5000, verbose=False),
        'HC': lambda: optimizer.hill_climbing(max_restarts=10, max_iter_per_restart=500)
    }
    results = {name: {'costs': [], 'times': [], 'evals': []} for name in algorithms}

    for i in range(n_runs):
        random.seed(i)
        np.random.seed(i)
        for name, algo in algorithms.items():
            _, cost, _, _, exec_time, evals = algo()
            results[name]['costs'].append(cost)
            results[name]['times'].append(exec_time)
            results[name]['evals'].append(evals)

    for name in algorithms:
        costs = results[name]['costs']
        times = results[name]['times']
        print(f"\n  {name}:")
        print(f"    Mean cost  = {np.mean(costs):.2f} +/- {np.std(costs):.2f}")
        print(f"    Mean time  = {np.mean(times):.2f} s")
        print(f"    Mean evals = {np.mean(results[name]['evals']):.0f}")

    print("\n  Wilcoxon signed-rank tests (paired, SA vs other):")
    for other in ['TS', 'HC']:
        stat, p = wilcoxon(results['SA']['costs'], results[other]['costs'])
        sig = "(significant, p<0.05)" if p < 0.05 else ""
        print(f"    SA vs {other}: p = {p:.4f} {sig}")

    # I generate a boxplot comparing the final cost distributions.
    fig, ax = plt.subplots(figsize=(8, 6))
    data_to_plot = [results[name]['costs'] for name in algorithms]
    ax.boxplot(data_to_plot, tick_labels=list(algorithms.keys()))
    ax.set_ylabel('Final cost')
    ax.set_title('Cost distributions: SA vs Tabu Search vs HC (30 runs)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    boxplot_path = PLOTS_DIR / "sa_ts_hc_boxplot.png"
    plt.savefig(boxplot_path, dpi=120)
    plt.close()
    print(f"\n  [Comparison] Boxplot saved to {boxplot_path}")

    # I save the detailed results to a CSV file.
    csv_path = REPORTS_DIR / "optimization_comparison.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'Algorithm', 'Cost', 'Time_s', 'Evaluations'])
        for run in range(n_runs):
            for name in algorithms:
                writer.writerow([run, name, results[name]['costs'][run],
                                 results[name]['times'][run], results[name]['evals'][run]])
    print(f"  [Comparison] Detailed results saved to {csv_path}")

    return results


# I test different combinations of initial temperature (T0) and cooling rate (alpha).
def sensitivity_analysis_sa(optimizer, n_runs=5, max_iter=2000):
    print(f"\n  [SA Sensitivity] Testing T0 in {{50,100,200}} x alpha in {{0.90,0.95,0.99}}, {n_runs} runs each")
    T0_values = [50, 100, 200]
    alpha_values = [0.90, 0.95, 0.99]
    results = np.zeros((len(T0_values), len(alpha_values)))

    for i, T0 in enumerate(T0_values):
        for j, alpha in enumerate(alpha_values):
            costs = []
            for run in range(n_runs):
                random.seed(run)
                np.random.seed(run)
                _, cost, _, _, _, _ = optimizer.simulated_annealing(
                    T0=T0, alpha=alpha, max_iter=max_iter, verbose=False)
                costs.append(cost)
            results[i, j] = np.mean(costs)
            print(f"    T0={T0}, alpha={alpha}: mean cost = {results[i, j]:.2f}")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(results, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(im, label='Mean cost (lower is better)')
    plt.xticks(range(len(alpha_values)), alpha_values)
    plt.yticks(range(len(T0_values)), T0_values)
    plt.xlabel('alpha (cooling rate)')
    plt.ylabel('T0 (initial temperature)')
    plt.title('SA Hyperparameter Sensitivity (lower = better)')
    for i in range(len(T0_values)):
        for j in range(len(alpha_values)):
            plt.text(j, i, f'{results[i, j]:.1f}', ha='center', va='center', color='white')
    plt.tight_layout()
    heatmap_path = PLOTS_DIR / "sa_hyperparameter_heatmap.png"
    plt.savefig(heatmap_path, dpi=120)
    plt.close()
    print(f"  [SA Sensitivity] Heatmap saved to {heatmap_path}")
    return results


# I plot the mean and standard deviation of the cost over iterations for all three algorithms.
def plot_convergence_comparison(optimizer, n_runs=10, max_iter=3000):
    print(f"\n  [Convergence] Generating convergence curves (mean over {n_runs} runs)...")
    histories = {'SA': [], 'TS': [], 'HC': []}
    for i in range(n_runs):
        random.seed(i)
        np.random.seed(i)
        _, _, hist_sa, _, _, _ = optimizer.simulated_annealing(max_iter=max_iter)
        histories['SA'].append(hist_sa)
        _, _, hist_ts, _, _, _ = optimizer.tabu_search(max_iter=max_iter)
        histories['TS'].append(hist_ts)
        _, _, hist_hc, _, _, _ = optimizer.hill_climbing(max_restarts=1, max_iter_per_restart=max_iter)
        histories['HC'].append(hist_hc)

    # I truncate all histories to the same length for a fair comparison.
    min_len = min(min(len(h) for h in histories['SA']),
                  min(len(h) for h in histories['TS']),
                  min(len(h) for h in histories['HC']))
    for algo in histories:
        histories[algo] = [h[:min_len] for h in histories[algo]]

    plt.figure(figsize=(12, 5))
    colors = {'SA': 'blue', 'TS': 'red', 'HC': 'green'}
    for algo in ['SA', 'TS', 'HC']:
        data = np.array(histories[algo])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        iterations = np.arange(len(mean))
        plt.plot(iterations, mean, label=algo, color=colors[algo])
        plt.fill_between(iterations, mean - std, mean + std, alpha=0.2, color=colors[algo])

    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('SA vs TS vs HC — Convergence (mean +/- 1 std, 10 runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    conv_path = PLOTS_DIR / "sa_ts_hc_convergence.png"
    plt.savefig(conv_path, dpi=120)
    plt.close()
    print(f"  [Convergence] Plot saved to {conv_path}")


if __name__ == "__main__":
    print("[SA Demo] Starting optimisation demo: SA vs Tabu Search vs Hill Climbing")

    # If the RAG system is available, I show evidence for the penalties used in the cost function.
    if RAG_AVAILABLE:
        print("\n[RAG Demo] Retrieving scientific evidence for cost function penalties...")
        rag = FitnessRAGSystem(use_pdfs=True)
        queries = ["muscle recovery 48 hours", "rest days beginner intermediate",
                   "weekly volume per muscle group"]
        for q in queries:
            evidence = rag.retrieve_evidence(q, k=1)
            if evidence:
                print(f"  Query: '{q}'")
                print(f"  Evidence: {evidence[0]['content'][:100]}...")
            else:
                print(f"  Query: '{q}' — no evidence retrieved.")
    else:
        print("\n[RAG] RAG system not available, skipping evidence demo.")

    profile = {
        'days_per_week': 4,
        'level': 'intermediate',
        'goal': 'general_fitness',
        'exercises_per_day': 3,
    }
    optimizer = RoutineOptimizer(profile)

    print("\n[Demo] Generating a random initial routine as the starting point...")
    initial_routine = optimizer.generate_initial_solution()
    print(optimizer.format_routine(initial_routine))
    print(f"  Initial cost: {optimizer.calculate_energy(initial_routine):.2f}")

    results = compare_algorithms_statistical(optimizer, n_runs=30)
    sensitivity_analysis_sa(optimizer, n_runs=5, max_iter=2000)
    plot_convergence_comparison(optimizer, n_runs=10, max_iter=3000)

    best_routine, best_energy, _, _, _, _ = optimizer.simulated_annealing(verbose=True)
    print("\n[SA] Best routine found by Simulated Annealing:")
    print(optimizer.format_routine(best_routine))
    print(f"  Final cost: {best_energy:.2f}")

    print("\n[Demo] Complete. Plots and reports saved to models/plots/ and models/reports/")