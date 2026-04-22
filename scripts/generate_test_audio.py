# scripts/generate_test_audio.py


# This script generates a diverse set of test audio files using text‑to‑speech.
# I use these audio files to evaluate the Word Error Rate (WER) of the VoiceCoach
# in a realistic way, with different phrases, speeds, and voices.



import os
import json
import random
from pathlib import Path

def generate_test_audio():
    # I find the project root directory and create the test_audio folder inside data.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    TEST_AUDIO_DIR = PROJECT_ROOT / "data" / "test_audio"
    TEST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # I define a large set of fitness‑related phrases and voice commands.
    # I include variations, numbers, and longer phrases to challenge the ASR system.
    test_phrases = [
        # Basic commands
        "start workout",
        "begin exercise",
        "let's go",
        "start now",
        "pause workout",
        "stop session",
        "wait a moment",
        "next exercise",
        "skip this one",
        "previous exercise",
        "how many reps left",
        "how many sets remain",
        "what's my progress",
        "show me my stats",
        "how am I doing",
        "am I doing it right",
        "check my form",
        "correct my posture",
        "is my back straight",
        "are my knees too far forward",
        "take a break",
        "I need rest",
        "give me thirty seconds",
        "I'm done",
        "finish workout",
        "end session",
        "help me",
        "what can I say",
        "list commands",
        # Phrases with fatigue or sentiment
        "I'm feeling tired",
        "this is hard",
        "I'm exhausted",
        "my muscles hurt",
        "I'm sore from yesterday",
        "can't keep up",
        "this is too heavy",
        "I need a longer rest",
        "I'm struggling",
        "I feel great",
        "that was easy",
        "I can do more",
        # Phrases with numbers (these often cause errors)
        "I did ten reps",
        "I completed twelve repetitions",
        "set the target to eight",
        "increase weight by five kilograms",
        "decrease reps to six",
        "I want to do four sets",
        "my max is one hundred kilos",
    ]

    # I will create some variations with different speed or intonation.
    # In total I will have around 35‑40 files.

    transcriptions = {}

    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        # I try to select two different English voices for more variety.
        english_voices = [v for v in voices if 'english' in v.name.lower() or 'en' in v.id.lower()]
        if len(english_voices) >= 2:
            voice1, voice2 = english_voices[0], english_voices[1]
        else:
            voice1 = voice2 = voices[0] if voices else None

        print("[TTS] Generating diverse test set...")
        
        file_counter = 1
        for i, text in enumerate(test_phrases):
            # I alternate voices to simulate different speakers.
            current_voice = voice1 if i % 2 == 0 else voice2
            engine.setProperty('voice', current_voice.id)
            
            # I slightly vary the speech rate for some phrases.
            rate = 150
            if i % 3 == 0:
                rate = 170  # a bit faster
            elif i % 5 == 0:
                rate = 130  # a bit slower
            engine.setProperty('rate', rate)
            
            # I generate the main audio file.
            filename = f"test_{file_counter:03d}.wav"
            filepath = TEST_AUDIO_DIR / filename
            engine.save_to_file(text, str(filepath))
            transcriptions[filename] = text
            print(f"  -> {filename}: '{text}' (voice={current_voice.name[:20]}, rate={rate})")
            file_counter += 1
            
            # For a few key phrases, I create a second version with a different intonation.
            # This increases the number of samples and the variability.
            if text in ["start workout", "how many reps left", "I'm feeling tired", "check my form", "I'm done"]:
                filename2 = f"test_{file_counter:03d}.wav"
                filepath2 = TEST_AUDIO_DIR / filename2
                engine.setProperty('rate', 160)
                engine.save_to_file(text, str(filepath2))
                transcriptions[filename2] = text
                print(f"  -> {filename2}: '{text}' (variant)")
                file_counter += 1

        engine.runAndWait()
        print(f"[TTS] Generated {len(transcriptions)} audio files.")
    except ImportError:
        print("[ERROR] pyttsx3 is not installed. Run: pip install pyttsx3")
        return
    except Exception as e:
        print(f"[ERROR] Could not generate audio: {e}")
        return

    # I save the transcription mapping as a JSON file.
    json_path = TEST_AUDIO_DIR / "transcriptions.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcriptions, f, indent=2)
    print(f"[JSON] Transcriptions saved to: {json_path}")

    print("\n[OK] Test data ready. Now VoiceCoach can evaluate WER with a realistic set.")
    print(f"Total samples: {len(transcriptions)}")

if __name__ == "__main__":
    generate_test_audio()