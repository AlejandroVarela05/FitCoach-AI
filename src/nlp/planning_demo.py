# planning_demo.py


# This script provides an interactive command‑line interface to demonstrate the
# integration of the three planning and reasoning modules of FitCoach AI:
# the A* macro‑planner, the Simulated Annealing routine optimiser, and the
# RAG‑based evidence‑aware workout generator. Users can run each module
# individually or combine them to see how the system builds a complete,
# scientifically grounded training plan.
#
# PURPOSE:
#   - Allow the user to input their fitness profile and goals interactively.
#   - Showcase the A* search for estimating weeks to goal.
#   - Showcase Simulated Annealing for generating a balanced weekly schedule.
#   - Showcase the RAG system for creating an evidence‑based routine with
#     explicit citations.
#   - Demonstrate full integration where A* and SA outputs are fed into the RAG
#     prompt to produce a coherent, multi‑level plan.
#
# COURSE CONNECTION:
#   This demo ties together "Intelligent Systems" (A*, Simulated Annealing,
#   Knowledge Base) and "Speech & Natural Language Processing" (RAG, LLM
#   prompting). It illustrates how different AI techniques can be orchestrated
#   to solve a complex, real‑world problem.
#
# DECISIONS:
#   - I provide a menu‑driven interface so the user can explore each component
#     at their own pace.
#   - The helper functions `input_float` and `input_int` gracefully handle
#     default values, making the demo user‑friendly.
#   - The `parse_rag_response` function attempts to extract structured JSON from
#     the LLM output, falling back to raw text if parsing fails.
#   - Integration options inject the outputs of earlier steps as additional
#     context into the RAG prompt, ensuring consistency across planning levels.



import sys
import json
import re
from pathlib import Path

# I add the project root to the path so I can import the core modules.
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.intelligent_systems.astar_planner import FitnessGoalProblem, astar_search
from src.intelligent_systems.simulated_annealing import RoutineOptimizer
from src.nlp.rag_system import FitnessRAGSystem


def input_float(prompt, default):
    # I read a floating‑point value from the user, returning the default if the input is empty.
    val = input(prompt).strip()
    return float(val) if val else default


def input_int(prompt, default):
    # I read an integer value from the user, returning the default if the input is empty.
    val = input(prompt).strip()
    return int(val) if val else default


def get_basic_profile():
    # I collect the user's fitness level, primary goal, and available training days.
    level_map = {'1': 'beginner', '2': 'intermediate', '3': 'advanced'}
    goal_map = {
        '1': 'muscle_gain',
        '2': 'weight_loss',
        '3': 'endurance',
        '4': 'strength',
        '5': 'recovery'
    }

    print("\n--- Basic profile ---")
    print("\nFitness level:")
    print("  1. Beginner")
    print("  2. Intermediate")
    print("  3. Advanced")
    level_choice = input("Choose (1-3) [2]: ").strip() or '2'
    level = level_map.get(level_choice, 'intermediate')

    print("\nPrimary goal:")
    print("  1. Muscle gain")
    print("  2. Weight loss")
    print("  3. Endurance")
    print("  4. Strength")
    print("  5. Recovery / Rehabilitation")
    goal_choice = input("Choose (1-5) [1]: ").strip() or '1'
    goal = goal_map.get(goal_choice, 'muscle_gain')

    days = input_int("Available training days per week (2-6) [4]: ", 4)
    return level, goal, days


def run_astar():
    # I run the A* planner to estimate how many weeks are needed to reach the user's goals.
    print("\n[A* Planner] Estimating weeks needed to reach your fitness goals.")
    print("This module gives a macro-level timeline, not a specific exercise list.\n")

    print("--- Current fitness values (0.0 - 1.0, except body fat %) ---")
    init_strength    = input_float("Current strength [0.3]: ", 0.3)
    init_endurance   = input_float("Current endurance [0.4]: ", 0.4)
    init_flexibility = input_float("Current flexibility [0.2]: ", 0.2)
    init_body_fat    = input_float("Current body fat (%) [22.0]: ", 22.0)

    print("\n--- Target values ---")
    goal_strength    = input_float("Target strength [0.7]: ", 0.7)
    goal_endurance   = input_float("Target endurance [0.7]: ", 0.7)
    goal_flexibility = input_float("Target flexibility [0.5]: ", 0.5)
    goal_body_fat    = input_float("Target body fat (%) [15.0]: ", 15.0)

    max_weeks = input_int("Maximum weeks to plan [25]: ", 25)
    sessions  = input_int("Training sessions per week [4]: ", 4)

    user_state = {
        'strength':        init_strength,
        'endurance':       init_endurance,
        'flexibility':     init_flexibility,
        'body_fat':        init_body_fat,
        'weekly_sessions': sessions,
        'weeks_elapsed':   0,
    }
    goal_state = {
        'strength':    goal_strength,
        'endurance':   goal_endurance,
        'flexibility': goal_flexibility,
        'body_fat':    goal_body_fat,
    }

    # I loop until a feasible plan is found or the user quits.
    while True:
        problem = FitnessGoalProblem(user_state, goal_state, max_weeks)
        result  = astar_search(problem)

        if result['feasible']:
            print(f"\n[A*] Feasible plan found: {result['weeks_needed']} weeks needed.")
            print("Recommended focus per week:")
            for step in result['plan']:
                print(f"  Week {step['week']:2d}: {step['action']}")
            return result
        else:
            print(f"\n[A*] No feasible plan found within {max_weeks} weeks.")
            new_max = input_int("   Enter a higher week limit (0 to exit): ", 0)
            if new_max > max_weeks:
                max_weeks = new_max
            else:
                print("   You can adjust your targets or week limit and try again.")
                return None


def run_sa():
    # I run Simulated Annealing to produce an optimised weekly exercise schedule.
    print("\n[Simulated Annealing] Generating an optimised weekly exercise schedule.")
    print("The cost function penalises muscle-group overlap and insufficient rest.\n")

    level, goal, days = get_basic_profile()
    exercises_per_day = input_int("Exercises per training day [3]: ", 3)

    print("\n[SA] Optimising... (this may take a few seconds)")
    opt = RoutineOptimizer({
        'days_per_week':    days,
        'level':            level,
        'goal':             goal,
        'exercises_per_day': exercises_per_day,
    })
    routine, cost, _, _, _, _ = opt.simulated_annealing(verbose=False)

    print(f"\n[SA] Optimised routine (cost = {cost:.2f}, lower is better):")
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days_of_week:
        exs = routine.get(day, [])
        if exs:
            ex_list = ", ".join([ex.replace('_', ' ').title() for ex in exs])
            print(f"  {day}: {ex_list}")
        else:
            print(f"  {day}: Rest")
    return routine, cost


def parse_rag_response(raw_text):
    # I attempt to extract a valid JSON object from the LLM's raw response.
    try:
        data = json.loads(raw_text)
        if 'choices' in data and len(data['choices']) > 0:
            msg     = data['choices'][0].get('message', {})
            content = msg.get('content', '')
            if content:
                raw_text = content
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # I clean up common formatting issues (e.g., double braces).
    cleaned = raw_text.replace('{{', '{').replace('}}', '}')

    start = cleaned.find('{')
    end   = cleaned.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None

    json_candidate = cleaned[start:end + 1]

    # I add quotes around numeric ranges so they become valid JSON strings.
    json_candidate = re.sub(
        r':\s*(\d+-\d+[%]?(?:\s*min)?)(?=[,\n\r}\]])',
        r': "\1"',
        json_candidate
    )

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        return None


def run_rag(extra_context=""):
    # I run the RAG system to generate an evidence‑based training routine.
    print("\n[RAG] Generating an evidence-based training routine using an LLM.")
    print("The RAG system retrieves relevant papers from FAISS and passes them")
    print("to the language model as grounding context.\n")

    level, goal, days = get_basic_profile()

    print("\n--- Language model configuration ---")
    print("Available providers:")
    print("  1. Groq (cloud API, requires GROQ_API_KEY in .env)")
    print("  2. Ollama (local server, e.g. http://localhost:11434)")
    print("  3. LM Studio (local server, e.g. http://localhost:1234/v1)")
    prov_choice = input("Choose (1-3) [3]: ").strip() or '3'

    if prov_choice == '1':
        provider = 'groq'
        print("\nGroq models:")
        print("  1. llama-3.1-8b-instant  (fast, lower cost)")
        print("  2. llama-3.3-70b-versatile (more capable)")
        model_choice = input("Choose (1-2) [1]: ").strip() or '1'
        model    = "llama-3.1-8b-instant" if model_choice == '1' else "llama-3.3-70b-versatile"
        endpoint = None
    elif prov_choice == '2':
        provider = 'ollama'
        print("\nTypical Ollama models:")
        print("  1. tinyllama:latest  (very lightweight)")
        print("  2. llama3:8b")
        model_choice = input("Choose (1-2) [1]: ").strip() or '1'
        model    = "tinyllama:latest" if model_choice == '1' else "llama3:8b"
        endpoint = input("Endpoint [http://localhost:11434]: ").strip() or "http://localhost:11434"
    else:
        provider = 'lmstudio'
        model    = "local-model"
        endpoint = input("Endpoint [http://localhost:1234/v1]: ").strip() or "http://localhost:1234/v1"

    print("\nPrompting strategies (covered in NLP lectures):")
    print("  1. evidence_first  -- retrieval context shown before the question")
    print("  2. chain_of_thought -- forces step-by-step reasoning before answer")
    print("  3. standard         -- direct generation without reasoning steps")
    strat_choice = input("Choose (1-3) [1]: ").strip() or '1'
    strategy = {'1': 'evidence_first', '2': 'chain_of_thought', '3': 'standard'}.get(strat_choice, 'evidence_first')

    query = input("\nDescribe your request (e.g. 'Weekly plan for muscle gain in 4 days'):\n> ").strip()
    if not query:
        query = "Weekly plan for muscle gain in 4 days"

    if extra_context:
        query = f"{query}\n\n[ADDITIONAL CONTEXT]\n{extra_context}"

    # I instruct the LLM to respond only with a JSON object containing a weekly plan.
    json_instruction = (
        "\n\nRespond ONLY with a valid JSON object containing the key 'weekly_plan'. "
        "The 'weekly_plan' object must map days of the week (Monday, Tuesday, ...) "
        "to lists of exercises (strings or dicts with 'exercise', 'sets', 'reps'). "
        "Ensure all 'reps' values are quoted strings (e.g. \"reps\": \"8-12\"). "
        "Do not include any text outside the JSON."
    )
    query += json_instruction

    print("\n[RAG] Initialising system (indexing papers on first run, may take a few seconds)...")
    rag = FitnessRAGSystem(use_pdfs=True)
    rag.llm_provider = provider
    rag.model_name   = model
    if endpoint:
        if provider == 'ollama':
            rag.ollama_url   = endpoint
        elif provider == 'lmstudio':
            rag.lmstudio_url = endpoint
    rag._init_llm()

    print("[RAG] Generating routine...")
    result       = rag.generate_routine(query, goal, level, days, strategy)
    raw_response = result['raw_response']

    response = parse_rag_response(raw_response)
    if response and 'weekly_plan' in response:
        print("\n[RAG] Routine generated (JSON format):")
        for day, exercises in response['weekly_plan'].items():
            if exercises:
                if isinstance(exercises[0], dict):
                    ex_list = ", ".join([ex.get('exercise', '') for ex in exercises])
                else:
                    ex_list = ", ".join(exercises)
                print(f"  {day}: {ex_list}")
            else:
                print(f"  {day}: Rest")

        params = response.get('derived_parameters') or response.get('parameters')
        if params:
            print("\n[RAG] Recommended training parameters:")
            for k, v in params.items():
                val = v.get('value', v) if isinstance(v, dict) else v
                print(f"  {k}: {val}")

        cited = result.get('citation_analysis', {}).get('cited_sources', [])
        if cited:
            print("\n[RAG] Cited sources:")
            for src in cited[:3]:
                print(f"  - {src}")
    else:
        print("\n[RAG] The model responded in free text (not structured JSON).")
        try:
            data = json.loads(raw_response)
            if 'choices' in data:
                text = data['choices'][0]['message']['content']
            else:
                text = raw_response
        except Exception:
            text = raw_response
        print(text)


def run_astar_plus_rag():
    # I combine A* and RAG: the A* macro plan is injected as context into the RAG prompt.
    print("\n[Integration] A* + RAG")
    print("Step 1: A* Planner will estimate your macro timeline.\n")
    astar_result = run_astar()

    if astar_result is None:
        print("\n[Warning] No A* plan obtained. RAG will run without that context.")
        extra = ""
    else:
        plan_text = (
            f"The A* planner estimates {astar_result['weeks_needed']} weeks to reach your goals, "
            f"with the following weekly focus:\n"
        )
        for step in astar_result['plan']:
            plan_text += f"  Week {step['week']:2d}: {step['action']}\n"
        extra  = plan_text
        extra += (
            "\nINSTRUCTION: Generate a weekly routine that reflects these focus periods. "
            "For example, if the focus is 'strength_focus', use low rep ranges (1-5) "
            "and long rest periods; if 'endurance_focus', use high reps (15-20) and "
            "short rest periods. Explain in the JSON how you applied the A* plan."
        )
        print("\n[Integration] A* plan will be injected as context into the RAG prompt.")

    input("\nPress Enter to continue to RAG...")
    run_rag(extra_context=extra)


def run_full_integration():
    # I run the complete pipeline: A* → SA → RAG.
    print("\n[Integration] A* + Simulated Annealing + RAG")
    print("1. A* estimates the timeline and weekly focus.")
    print("2. SA generates a balanced exercise schedule.")
    print("3. RAG refines the schedule with scientific evidence.\n")

    astar_result = run_astar()
    if astar_result is None:
        print("\n[Warning] No A* plan obtained. Continuing without that context.")
        extra_astar = ""
    else:
        extra_astar = (
            f"A* plan ({astar_result['weeks_needed']} weeks): "
            + ", ".join([step['action'] for step in astar_result['plan']])
        )

    input("\nPress Enter to continue to Simulated Annealing...")
    sa_routine, sa_cost = run_sa()

    extra = ""
    if extra_astar:
        extra += f"A* CONTEXT: {extra_astar}\n\n"
    extra += f"SA CONTEXT: Optimised weekly schedule (cost {sa_cost:.2f}):\n"
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days_of_week:
        exs = sa_routine.get(day, [])
        if exs:
            extra += f"  {day}: {', '.join(exs)}\n"
        else:
            extra += f"  {day}: Rest\n"

    extra += (
        "\nSTRICT INSTRUCTION: The routine MUST respect the rest/training day "
        "distribution produced by SA (do not change the days). Apply the A* "
        "periodisation focus for each week. For this first week, assign balanced "
        "sets, reps, and rest. Explain in the JSON how you integrated both contexts."
    )

    print("\n[Integration] A* and SA results will be injected into the RAG prompt.")
    input("\nPress Enter to continue to RAG...")
    run_rag(extra_context=extra)


def main():
    # Main menu loop.
    while True:
        print("\n[FitCoach AI] Planning Assistant")
        print("\nWhat would you like to do?\n")
        print("1. Estimate how many weeks to reach my fitness goals")
        print("   (A* Planner -- macro timeline without specific exercises)")
        print("\n2. Get an optimised weekly exercise schedule")
        print("   (Simulated Annealing -- balanced day distribution)")
        print("\n3. Generate a personalised routine backed by scientific papers")
        print("   (RAG System -- LLM + paper retrieval)")
        print("\n4. A* + RAG integration")
        print("   (Use A* macro plan as guidance for RAG)")
        print("\n5. Full integration: A* + SA + RAG")
        print("   (A* timeline + SA schedule + RAG evidence refinement)")
        print("\n6. Help me choose")
        print("\n7. Exit")

        choice = input("\nChoose an option (1-7): ").strip()

        if choice == '1':
            run_astar()
        elif choice == '2':
            run_sa()
        elif choice == '3':
            run_rag()
        elif choice == '4':
            run_astar_plus_rag()
        elif choice == '5':
            run_full_integration()
        elif choice == '6':
            print("\n--- Quick guide ---")
            print("- Option 1 (A*):  best if you want to know HOW LONG it takes to")
            print("  reach numeric targets (e.g. go from 22% to 15% body fat).")
            print("- Option 2 (SA):  best if you already know HOW MANY DAYS you train")
            print("  and want a well-distributed weekly schedule.")
            print("- Option 3 (RAG): best if you want a routine EXPLAINED with paper")
            print("  citations and can accept exercise variety beyond the fixed list.")
            print("- Options 4 and 5 combine modules for the best of each approach.")
            input("\nPress Enter to return to the menu...")
        elif choice == '7':
            print("\n[FitCoach AI] Goodbye!")
            break
        else:
            print("\n[Error] Invalid option, please choose 1-7.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()