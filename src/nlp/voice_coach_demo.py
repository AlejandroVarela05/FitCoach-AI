# voice_coach_demo.py


# This script provides an interactive demonstration of the FitCoach AI voice coach.
# It guides the user through a predefined workout routine (squat, push‑up, deadlift,
# shoulder press) and responds to spoken or typed feedback. The coach proactively
# asks how the user is feeling every 5 repetitions and adjusts the target reps
# based on detected fatigue or ease.
#
# PURPOSE:
#   - Showcase the full voice interaction pipeline in a controlled environment.
#   - Allow the user to choose between text input (for testing) and real voice
#     input (using the microphone).
#   - Demonstrate the coach's ability to understand intents, analyse sentiment,
#     and adapt the workout in real time.
#
# COURSE CONNECTION:
#   This demo integrates the "Speech and Natural Language Processing" module
#   (Whisper ASR, sentiment analysis, intent classification) with the proactive
#   coaching logic. It also illustrates the practical application of the RAG
#   system when citations are provided in the responses.
#
# DECISIONS:
#   - I use a fixed sample routine so the user can focus on the interaction
#     without needing to configure exercises.
#   - The user can press Enter to simulate one repetition, making the demo easy
#     to run without actual exercise detection.
#   - I disable the TTS engine in the demo and use a Windows PowerShell fallback
#     for speech output to avoid dependency issues.
#   - The proactive check every 5 reps mimics a human trainer's behaviour.



import os
import sys
import time
import tempfile
import subprocess
import re
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.nlp.voice_coach import VoiceCoach


# A sample workout routine used for the demonstration.
SAMPLE_ROUTINE = [
    {"exercise": "squat",          "target_reps": 12},
    {"exercise": "push_up",        "target_reps": 10},
    {"exercise": "deadlift",       "target_reps": 8},
    {"exercise": "shoulder_press", "target_reps": 10},
]


def speak_text(text):
    # I use Windows PowerShell's System.Speech to speak the given text.
    # This is a fallback when pyttsx3 is disabled or not available.
    if not text:
        return
    # Clean the text: remove newlines, non‑ASCII characters, and truncate if too long.
    cleaned = text.replace('\n', ' ').replace('\r', ' ')
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)
    if len(cleaned) > 100:
        cutoff = cleaned.rfind(' ', 0, 100)
        cutoff = cutoff if cutoff != -1 else 100
        cleaned = cleaned[:cutoff] + "..."
    if not cleaned:
        return
    try:
        safe_text  = cleaned.replace('"', '\\"')
        ps_command = (
            'Add-Type -AssemblyName System.Speech; '
            f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{safe_text}")'
        )
        subprocess.run(["powershell", "-Command", ps_command],
                       capture_output=True, timeout=5)
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass


def record_audio(duration=5, sample_rate=16000):
    # I record audio from the default microphone for a fixed duration.
    # This requires the sounddevice and soundfile libraries.
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("[Voice] sounddevice or soundfile not installed. Falling back to text mode.")
        return None

    print(f"[Voice] Recording for {duration} seconds... (speak now)")
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("[Voice] Recording finished.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name


def get_user_response(mode):
    # I obtain the user's response either as typed text or as a recorded audio file.
    if mode == 'text':
        return input("Your response (text): ").strip()
    else:
        audio_path = record_audio(duration=5)
        if audio_path is None:
            return input("Voice mode failed. Type your response: ").strip()
        return audio_path


def process_response(coach, user_input, response_mode, ctx):
    # I pass the user's input to the VoiceCoach and display the recognised intent,
    # sentiment, and the coach's spoken response. I also provide voice feedback
    # about the remaining repetitions.
    if response_mode == 'text':
        res = coach.process_command(text_command=user_input, session_context=ctx)
    else:
        res = coach.process_command(audio_path=user_input, session_context=ctx)
        try:
            os.unlink(user_input)
        except Exception:
            pass

    sentiment = res['sentiment']
    print(f"   [NLP] Intent: {res['intent']} | "
          f"Sentiment: {sentiment['sentiment']} (fatigue: {sentiment['fatigue_level']})")
    print(f"   [Coach] {res['response']}")

    reps_done = ctx['reps']
    target    = ctx['target_reps']
    remaining = target - reps_done
    if res.get('adjustment_made'):
        voice_msg = f"Target adjusted to {target} reps."
    else:
        voice_msg = f"You have done {reps_done} out of {target} reps. {remaining} left."
    speak_text(voice_msg)

    if res.get('adjustment_made'):
        print(f"   [Coach] Target adjusted to {ctx['target_reps']} reps.")

    return res


def main():
    print("[VoiceCoach] FitCoach AI - Voice Coach Demo")
    print("This demo simulates a training session with a fixed routine.")
    print("Press Enter each time you complete one repetition.")
    print("Every 5 reps the coach will ask how you feel (text or voice).")
    print("The coach adjusts the target based on your fatigue or ease.\n")

    mode_choice   = input("Response mode: (1) Text  (2) Voice  [1]: ").strip()
    response_mode = 'voice' if mode_choice == '2' else 'text'
    print(f"[Config] Mode: {'Voice' if response_mode == 'voice' else 'Text'}\n")

    print("[VoiceCoach] Initialising Voice Coach (loading models)...")
    coach = VoiceCoach(whisper_model_size="base", use_rag=True)
    # I disable the TTS engine inside VoiceCoach to avoid conflicts and use the PowerShell fallback.
    coach.tts_available = False
    print("[VoiceCoach] Coach ready.\n")

    ctx = {
        'exercise':    '',
        'target_reps': 0,
        'reps':        0,
        'form_status': 'correct',
    }

    input("Press Enter to start the session...")
    speak_text("Let's start the workout!")
    print("\n--- SESSION STARTED ---\n")

    for idx, ex in enumerate(SAMPLE_ROUTINE):
        exercise          = ex['exercise']
        target            = ex['target_reps']
        ctx['exercise']   = exercise
        ctx['target_reps'] = target
        ctx['reps']       = 0

        print(f"\n[Exercise {idx+1}/{len(SAMPLE_ROUTINE)}] {exercise.upper()} -- target: {target} reps")
        speak_text(f"Next exercise: {exercise}. Target: {target} repetitions.")

        while ctx['reps'] < ctx['target_reps']:
            input("   -> Press Enter to count 1 repetition...")
            ctx['reps'] += 1
            print(f"   Reps: {ctx['reps']}/{ctx['target_reps']}")

            # Every 5 repetitions (and not at the very end), I proactively ask for feedback.
            if (ctx['reps'] > 0
                    and ctx['reps'] % 5 == 0
                    and ctx['reps'] < ctx['target_reps']):
                question = (
                    f"You have done {ctx['reps']} reps of {exercise}. "
                    f"How are you feeling? Easy, hard, or okay?"
                )
                print(f"\n[Coach] \"{question}\"")
                speak_text(question)

                user_input = get_user_response(response_mode)
                if not user_input:
                    print("[Warning] No response received. Continuing without changes.")
                    continue

                process_response(coach, user_input, response_mode, ctx)
                target = ctx['target_reps']

        print(f"\n[Done] {exercise} completed ({ctx['reps']} reps)\n")
        speak_text(f"Good job! You finished {exercise}.")

        if idx < len(SAMPLE_ROUTINE) - 1:
            input("Press Enter to continue to the next exercise...")
        else:
            print("\n[Done] Full workout completed!")
            speak_text("Congratulations! You completed the entire workout.")

    print("\n[VoiceCoach] Demo finished. Thank you for trying FitCoach AI!")


if __name__ == "__main__":
    main()