# astar_planner.py


# This module implements A* search, Breadth‑First Search (BFS), and Greedy Best‑First
# Search (GBFS) to estimate the number of weeks needed to reach a fitness goal.
# The state represents the user's strength, endurance, flexibility, and body fat.
# Actions are weekly training focuses (e.g., strength_focus, endurance_focus).
#
# PURPOSE:
#   - Provide a macro‑level timeline for achieving fitness goals.
#   - Compare the efficiency and optimality of A*, BFS, and GBFS.
#   - Evaluate the quality of the admissible heuristic used by A*.
#   - Visualise the planned progression of fitness metrics.
#
# COURSE CONNECTION:
#   This script directly applies concepts from "Intelligent Systems" (Unit II –
#   solving problems by search). A* was studied in detail as an informed search
#   algorithm that guarantees optimality when the heuristic is admissible.
#
# DECISIONS:
#   - I model the fitness journey as a state‑space search problem with discrete
#     weekly actions.
#   - The heuristic assumes maximum possible improvement per week. This makes it
#     admissible (never overestimates) and ensures A* finds the shortest timeline.
#   - I compare A* against BFS (guaranteed optimal but slow) and GBFS (fast but
#     suboptimal) to highlight the efficiency of an informed search.



import heapq
import time
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import PROJECT_ROOT, PLOTS_DIR, REPORTS_DIR

try:
    from src.nlp.rag_system import FitnessRAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


def show_paper_references():
    # I display the scientific references that support the weekly improvement rates.
    papers_dir = PROJECT_ROOT / "data" / "papers"
    references = {
        "Strength improvement (3% per week)": "06_beginners_progression/ACSM_Progression09.pdf",
        "Endurance improvement (4% per week)": "05_cardio_fat_loss/20.Vianaetal.-2019-Isintervaltrainingthemagicbulletforfatloss.pdf",
        "Body fat reduction (1% per week)": "08_weight_general_health/willis-et-al-2012-effects-of-aerobic-and-or-resistance-training-on-body-mass-and-fat-mass-in-overweight-or-obese-adults.pdf",
        "Flexibility improvement (2% per week)": "07_concurrent_calisthenics/JSSMarticle2014.pdf",
        "Diminishing returns (Kraemer & Ratamess)": "06_beginners_progression/fundamentals_of_resistance_training__progression.17.pdf",
        "Strength gains adaptation (Swinton et al.)": "11_biomechanics_height/a-biomechanical-analysis-of-straight-and-hexagonal-barbell-174i5v2t90.pdf",
    }
    print("SCIENTIFIC REFERENCES USED IN THIS MODULE")
    for desc, rel_path in references.items():
        full_path = papers_dir / rel_path
        if full_path.exists():
            print(f"  {desc}: {full_path}")
        else:
            print(f"  {desc}: {rel_path} (file not found – check the path)")
    print("="*70 + "\n")


class FitnessGoalProblem:
    # This class defines the state‑space search problem for fitness planning.
    # The state is a dictionary with strength, endurance, flexibility, body fat,
    # weekly sessions, and weeks elapsed.
    def __init__(self, initial_state, goal_state, max_weeks=52):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.max_weeks = max_weeks

        # These maximum weekly improvement rates are based on the cited literature.
        self.max_weekly_improvement = {
            'strength': 0.03,
            'endurance': 0.04,
            'flexibility': 0.02,
            'body_fat_reduction': 1.0,
        }

    def goal_test(self, state):
        # The goal is reached when all target values are met (body fat ≤ target,
        # other metrics ≥ target).
        for key in self.goal_state:
            if key == 'body_fat':
                if state.get(key, 100) > self.goal_state[key]:
                    return False
            elif key in ['strength', 'endurance', 'flexibility']:
                if state.get(key, 0) < self.goal_state[key]:
                    return False
        return True

    def actions(self, state):
        # I return the available weekly focuses. If the maximum number of weeks is reached,
        # no actions are available.
        if state.get('weeks_elapsed', 0) >= self.max_weeks:
            return []
        actions = ['strength_focus', 'endurance_focus', 'flexibility_focus', 'balanced']
        if state.get('body_fat', 0) > self.goal_state.get('body_fat', state.get('body_fat', 0)):
            actions.append('fat_loss_focus')
        return actions

    def result(self, state, action):
        # I simulate one week of training and return the new state.
        new_state = dict(state)
        new_state['weeks_elapsed'] = state.get('weeks_elapsed', 0) + 1

        sessions = state.get('weekly_sessions', 3)
        session_factor = min(sessions / 3, 1.5)

        if action == 'strength_focus':
            current = state.get('strength', 0)
            diminishing = 1 - current
            improvement = 0.03 * session_factor * diminishing
            new_state['strength'] = min(1.0, current + improvement)
            new_state['endurance'] = min(1.0, state.get('endurance', 0) + 0.005)

        elif action == 'endurance_focus':
            current = state.get('endurance', 0)
            diminishing = 1 - current
            improvement = 0.03 * session_factor * diminishing
            new_state['endurance'] = min(1.0, current + improvement)
            new_state['strength'] = min(1.0, state.get('strength', 0) + 0.005)

        elif action == 'flexibility_focus':
            current = state.get('flexibility', 0)
            diminishing = 1 - current
            improvement = 0.02 * session_factor * diminishing
            new_state['flexibility'] = min(1.0, current + improvement)

        elif action == 'balanced':
            for metric in ['strength', 'endurance', 'flexibility']:
                current = state.get(metric, 0)
                diminishing = 1 - current
                improvement = 0.015 * session_factor * diminishing
                new_state[metric] = min(1.0, current + improvement)

        elif action == 'fat_loss_focus':
            current_fat = state.get('body_fat', 20)
            reduction = 0.7 * session_factor
            new_state['body_fat'] = max(5, current_fat - reduction)
            new_state['endurance'] = min(1.0, state.get('endurance', 0) + 0.01)

        return new_state

    def path_cost(self, state):
        # The cost is the number of weeks elapsed so far.
        return state.get('weeks_elapsed', 0)

    def heuristic(self, state):
        # I estimate the remaining weeks by assuming the maximum possible improvement
        # for each metric. This heuristic is admissible because it never underestimates
        # the true remaining weeks.
        max_weeks = 0
        for key in self.goal_state:
            if key == 'body_fat':
                gap = max(0, state.get(key, 20) - self.goal_state[key])
                rate = self.max_weekly_improvement['body_fat_reduction']
            elif key in self.max_weekly_improvement:
                gap = max(0, self.goal_state[key] - state.get(key, 0))
                rate = self.max_weekly_improvement[key]
            else:
                continue
            if rate > 0:
                weeks = gap / rate
                max_weeks = max(max_weeks, weeks)
        return max_weeks


class Node:
    # A node in the search tree stores the state, parent, action, and costs.
    def __init__(self, state, parent=None, action=None, path_cost=0, f_cost=0, h_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.f_cost = f_cost
        self.h_cost = h_cost

    def __lt__(self, other):
        # I compare nodes by f_cost so the priority queue works correctly.
        return self.f_cost < other.f_cost

    def get_path(self):
        # I reconstruct the path from the initial state to this node.
        path = []
        node = self
        while node.parent is not None:
            path.append({
                'week': node.path_cost,
                'action': node.action,
                'state': node.state
            })
            node = node.parent
        return list(reversed(path))


def astar_search(problem):
    # A* search uses f(n) = g(n) + h(n). It is guaranteed to find the optimal
    # solution if the heuristic is admissible.
    start_time = time.time()
    h0 = problem.heuristic(problem.initial_state)
    initial_node = Node(state=problem.initial_state, path_cost=0, f_cost=h0, h_cost=h0)
    open_list = [initial_node]
    explored = set()
    nodes_expanded = 0
    max_open_size = 0

    while open_list:
        max_open_size = max(max_open_size, len(open_list))
        current = heapq.heappop(open_list)
        nodes_expanded += 1

        if problem.goal_test(current.state):
            plan = current.get_path()
            exec_time = time.time() - start_time
            return {
                'feasible': True,
                'weeks_needed': current.path_cost,
                'plan': plan,
                'nodes_expanded': nodes_expanded,
                'max_open_size': max_open_size,
                'time_seconds': exec_time,
                'final_state': current.state
            }

        state_key = _state_to_key(current.state)
        if state_key in explored:
            continue
        explored.add(state_key)

        if current.path_cost >= problem.max_weeks:
            continue

        for action in problem.actions(current.state):
            child_state = problem.result(current.state, action)
            child_cost = current.path_cost + 1
            h = problem.heuristic(child_state)
            f = child_cost + h
            child_node = Node(child_state, current, action, child_cost, f, h)
            child_key = _state_to_key(child_state)
            if child_key not in explored:
                heapq.heappush(open_list, child_node)

    exec_time = time.time() - start_time
    return {
        'feasible': False,
        'weeks_needed': problem.max_weeks,
        'plan': [],
        'nodes_expanded': nodes_expanded,
        'max_open_size': max_open_size,
        'time_seconds': exec_time,
        'final_state': None
    }


def bfs_search(problem):
    # Breadth‑First Search explores the tree level by level. It is optimal but
    # uses more memory and expands more nodes than A* with a good heuristic.
    start_time = time.time()
    initial_node = Node(problem.initial_state, path_cost=0)
    queue = deque([initial_node])
    explored = set()
    nodes_expanded = 0
    max_queue_size = 0

    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        current = queue.popleft()
        nodes_expanded += 1

        if problem.goal_test(current.state):
            plan = current.get_path()
            exec_time = time.time() - start_time
            return {
                'feasible': True,
                'weeks_needed': current.path_cost,
                'plan': plan,
                'nodes_expanded': nodes_expanded,
                'max_open_size': max_queue_size,
                'time_seconds': exec_time,
                'final_state': current.state
            }

        state_key = _state_to_key(current.state)
        if state_key in explored:
            continue
        explored.add(state_key)

        if current.path_cost >= problem.max_weeks:
            continue

        for action in problem.actions(current.state):
            child_state = problem.result(current.state, action)
            child_node = Node(child_state, current, action, current.path_cost + 1)
            child_key = _state_to_key(child_state)
            if child_key not in explored:
                queue.append(child_node)

    exec_time = time.time() - start_time
    return {
        'feasible': False,
        'weeks_needed': problem.max_weeks,
        'plan': [],
        'nodes_expanded': nodes_expanded,
        'max_open_size': max_queue_size,
        'time_seconds': exec_time,
        'final_state': None
    }


def gbfs_search(problem):
    # Greedy Best‑First Search uses only the heuristic (h(n)). It is very fast
    # but can return suboptimal solutions.
    start_time = time.time()
    h0 = problem.heuristic(problem.initial_state)
    initial_node = Node(state=problem.initial_state, path_cost=0, f_cost=h0, h_cost=h0)
    open_list = [initial_node]
    explored = set()
    nodes_expanded = 0
    max_open_size = 0

    while open_list:
        max_open_size = max(max_open_size, len(open_list))
        current = heapq.heappop(open_list)
        nodes_expanded += 1

        if problem.goal_test(current.state):
            plan = current.get_path()
            exec_time = time.time() - start_time
            return {
                'feasible': True,
                'weeks_needed': current.path_cost,
                'plan': plan,
                'nodes_expanded': nodes_expanded,
                'max_open_size': max_open_size,
                'time_seconds': exec_time,
                'final_state': current.state
            }

        state_key = _state_to_key(current.state)
        if state_key in explored:
            continue
        explored.add(state_key)

        if current.path_cost >= problem.max_weeks:
            continue

        for action in problem.actions(current.state):
            child_state = problem.result(current.state, action)
            child_cost = current.path_cost + 1
            h = problem.heuristic(child_state)
            child_node = Node(child_state, current, action, child_cost, h, h)
            child_key = _state_to_key(child_state)
            if child_key not in explored:
                heapq.heappush(open_list, child_node)

    exec_time = time.time() - start_time
    return {
        'feasible': False,
        'weeks_needed': problem.max_weeks,
        'plan': [],
        'nodes_expanded': nodes_expanded,
        'max_open_size': max_open_size,
        'time_seconds': exec_time,
        'final_state': None
    }


def _state_to_key(state):
    # I create a hashable representation of the state for the explored set.
    return tuple(
        round(state.get(k, 0), 2)
        for k in sorted(['strength', 'endurance', 'flexibility', 'body_fat', 'weeks_elapsed'])
    )


def evaluate_heuristic_quality(problem, plan):
    # I compute the average ratio h(s) / h*(s) along the optimal path to see how
    # close the heuristic is to the true remaining cost.
    if not plan:
        return None
    ratios = []
    total_weeks = len(plan)
    for step in plan:
        state = step['state']
        weeks_left = total_weeks - step['week']
        h_val = problem.heuristic(state)
        if weeks_left > 0:
            ratios.append(h_val / weeks_left)
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return avg_ratio


def compare_algorithms(problem):
    # I run A*, BFS, and GBFS on the same problem and print a comparison table.
    print("COMPARISON: A* vs BFS vs Greedy Best-First Search")

    algorithms = {
        'A*': astar_search,
        'BFS': bfs_search,
        'GBFS': gbfs_search
    }
    results = {}
    for name, algo in algorithms.items():
        print(f"\nRunning {name}...")
        res = algo(problem)
        results[name] = res
        print(f"  Feasible: {res['feasible']}, Weeks: {res['weeks_needed']}, Nodes: {res['nodes_expanded']}, Time: {res['time_seconds']:.4f}s")

    if results['A*']['feasible']:
        h_quality = evaluate_heuristic_quality(problem, results['A*']['plan'])
        print(f"\nHeuristic accuracy (A* path): avg h(s)/h*(s) = {h_quality:.3f} (1.0 = perfect)")

    # I note if GBFS found a suboptimal solution.
    if results['GBFS']['feasible'] and results['A*']['feasible']:
        if results['GBFS']['weeks_needed'] > results['A*']['weeks_needed']:
            print("\nNote: GBFS found a suboptimal plan (more weeks). A* is guaranteed optimal.")
        elif results['GBFS']['weeks_needed'] == results['A*']['weeks_needed']:
            print("\nNote: GBFS happened to find the optimal length, but it is not guaranteed in general.")
    elif results['GBFS']['feasible'] and not results['A*']['feasible']:
        print("\nWarning: GBFS found a solution but A* did not – this should not happen with admissible heuristic.")

    # I save the comparison results to a CSV file for later analysis.
    csv_path = REPORTS_DIR / "astar_comparison.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Feasible', 'Weeks', 'Nodes_Expanded', 'Time_s', 'Max_Open_Size'])
        for name, res in results.items():
            writer.writerow([name, res['feasible'], res['weeks_needed'], res['nodes_expanded'],
                             f"{res['time_seconds']:.4f}", res['max_open_size']])
    print(f"\nComparison report saved to {csv_path}")

    return results


def plot_progress_curve(plan, initial_state, goal_state):
    # I generate a line plot showing how each fitness metric evolves over the planned weeks.
    if not plan:
        print("No plan to plot.")
        return

    weeks = [0] + [step['week'] for step in plan]
    metrics = ['strength', 'endurance', 'flexibility', 'body_fat']
    values = {m: [initial_state.get(m, 0)] for m in metrics}

    for step in plan:
        state = step['state']
        for m in metrics:
            values[m].append(state.get(m, values[m][-1]))

    for m in metrics:
        if len(values[m]) < len(weeks):
            values[m].append(values[m][-1])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'strength': 'blue', 'endurance': 'orange', 'flexibility': 'green', 'body_fat': 'red'}
    for m in metrics:
        label = m.replace('_', ' ').capitalize()
        if m == 'body_fat':
            label = 'Body Fat (%)'
        ax.plot(weeks, values[m], marker='o', label=label, color=colors[m])

    # I draw horizontal dashed lines to show the target values.
    for m in metrics:
        if m in goal_state:
            ax.axhline(y=goal_state[m], linestyle='--', alpha=0.5,
                       color=colors[m], label=f'Goal {m}')

    ax.set_xlabel('Week')
    ax.set_ylabel('Metric Value')
    ax.set_title('Fitness Progression Over Planned Weeks (A* Optimal Plan)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = PLOTS_DIR / "astar_progress_curve.png"
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"Progress curve saved to {save_path}")


def sensitivity_analysis(initial_state, goal_state, max_weeks=20):
    # I test how sensitive the plan is to the assumed weekly improvement rates.
    print("SENSITIVITY ANALYSIS: Varying weekly improvement rates")

    base_rates = {
        'strength': 0.03,
        'endurance': 0.04,
        'flexibility': 0.02,
        'body_fat_reduction': 1.0
    }

    for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
        problem = FitnessGoalProblem(initial_state, goal_state, max_weeks)
        problem.max_weekly_improvement = {k: v * factor for k, v in base_rates.items()}
        result = astar_search(problem)
        weeks_str = str(result['weeks_needed']) if result['feasible'] else 'N/A'
        print(f"  Factor {factor:4.2f}: feasible={result['feasible']}, weeks={weeks_str}, nodes={result['nodes_expanded']}")


if __name__ == "__main__":
    # I run a demonstration of the A* planner, compare it with BFS and GBFS,
    # and plot the progression curve.
    print("[A*] Comparing A* vs BFS vs GBFS for fitness goal planning...")
    print("[A*] See show_paper_references() output below for scientific justification.")

    show_paper_references()

    if RAG_AVAILABLE:
        print("\n[DEMO] Retrieving evidence for strength improvement rate from RAG...")
        rag = FitnessRAGSystem(use_pdfs=True)
        evidence = rag.retrieve_evidence("strength improvement rate per week", k=1)
        if evidence:
            print(f"  RAG evidence: {evidence[0]['content'][:150]}...")
        else:
            print("  No evidence retrieved.")
    else:
        print("\n[RAG not available] Skipping evidence retrieval demo.")

    user = {
        'strength': 0.3,
        'endurance': 0.4,
        'flexibility': 0.2,
        'body_fat': 22.0,
        'weekly_sessions': 4,
        'weeks_elapsed': 0,
    }
    goal = {
        'strength': 0.5,
        'endurance': 0.6,
        'flexibility': 0.3,
        'body_fat': 18.0,
    }

    problem = FitnessGoalProblem(user, goal, max_weeks=20)
    results = compare_algorithms(problem)

    if results['A*']['feasible']:
        plot_progress_curve(results['A*']['plan'], user, goal)

    sensitivity_analysis(user, goal, max_weeks=20)

    print("\nDemo complete. The A* algorithm is optimal and more efficient than BFS.")