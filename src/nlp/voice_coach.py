# voice_coach.py


# This module implements a voice‑activated coach for the FitCoach AI system.
# It uses OpenAI Whisper for speech‑to‑text, DistilBERT for sentiment analysis,
# pyttsx3 for text‑to‑speech, and a keyword‑based intent classifier. It also
# integrates with the RAG system to provide evidence‑based citations when giving
# feedback. The coach can detect user fatigue from spoken words and adjust the
# workout target (repetitions) in real time.
#
# PURPOSE:
#   - Transcribe user speech to text.
#   - Classify the user's intent (start, stop, status, etc.).
#   - Analyse sentiment and fatigue level from the transcribed text.
#   - Generate spoken responses and proactively check on the user every 5 reps.
#   - Evaluate the performance of the voice pipeline (Word Error Rate, latency,
#     sentiment accuracy) and compare different Whisper model sizes.
#
# COURSE CONNECTION:
#   This module fulfills the "Speech and Natural Language Processing" course
#   requirements (Unit I – NLP basics, Unit IV – speech processing). It also
#   connects to the RAG system studied in the "Advanced Machine Learning" course
#   (Unit V – Transformers and LLMs). The proactive coaching loop demonstrates
#   an intelligent agent that perceives user state and adapts its behaviour.
#
# DECISIONS:
#   - I chose Whisper "base" as the default model because it offers a good
#     balance between accuracy and speed for real‑time interaction.
#   - I use a simple keyword‑matching approach for intent classification because
#     the vocabulary of a workout session is limited and predictable.
#   - Sentiment analysis helps me understand the user's emotional state; when
#     combined with fatigue keywords, it triggers workload adjustments.
#   - I keep a log of adjustments and fatigue history to evaluate the coach's
#     responsiveness.
#   - The proactive check‑in every 5 repetitions mimics a human trainer who
#     periodically asks how the client is feeling.



import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import soundfile as sf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import MODELS_DIR, REPORTS_DIR, PLOTS_DIR, PROJECT_ROOT

# I try to import the RAG system. If it is not available, I disable that feature.
try:
    from src.nlp.rag_system import FitnessRAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# I define a dataclass to store all evaluation metrics in one place.
@dataclass
class EvaluationMetrics:
    wer: float = 0.0
    sentiment_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    fatigue_trigger_rate: float = 0.0
    model_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)

class VoiceCoach:
    # These keyword dictionaries map user phrases to intents and fatigue categories.
    # I include both English and Spanish keywords to support bilingual users.
    INTENT_KEYWORDS = {
        'start':  ['start', 'begin', 'go', 'empezar', 'vamos', 'comenzar'],
        'stop':   ['stop', 'pause', 'wait', 'para', 'parar', 'detener'],
        'next':   ['next', 'skip', 'siguiente', 'next exercise', 'proximo'],
        'status': ['status', 'how am i', 'how many', 'cuantas', 'reps', 'repeticiones'],
        'form':   ['form', 'posture', 'correct', 'forma', 'postura', 'tecnica'],
        'rest':   ['rest', 'break', 'descanso', 'descansar', 'pausa'],
        'done':   ['done', 'finish', 'end', 'terminar', 'listo', 'finalizar'],
        'help':   ['help', 'what can i say', 'ayuda', 'commands', 'comandos'],
        'feedback': ['tired', 'hard', 'pain', 'sore', 'heavy', 'struggling',
                     'cansado', 'duro', 'duele', 'pesado', 'me cuesta'],
        # I added these intents to allow the user to express perceived difficulty.
        'easy':   ['easy', 'light', 'facil', 'ligero'],
        'hard':   ['hard', 'difficult', 'tough', 'duro', 'dificil'],
        'okay':   ['okay', 'fine', 'good', 'bien', 'normal'],
    }
    # These keywords are used specifically to estimate the user's fatigue level.
    FATIGUE_KEYWORDS = {
        'tired':    ['tired', 'exhausted', 'fatigued', 'drained', 'cansado', 'agotado'],
        'hard':     ['hard', 'difficult', 'tough', 'struggling', 'duro', 'dificil', 'cuesta'],
        'pain':     ['pain', 'hurt', 'sore', 'ache', 'duele', 'dolor', 'molestia'],
        'slow':     ['slow', 'heavy', 'can\'t keep up', 'lento', 'pesado'],
    }

    def __init__(self, whisper_model_size: str = "base", use_rag: bool = True,
                 sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.whisper_model_size = whisper_model_size
        self.whisper_model = None
        self.whisper_available = False
        self.tts_engine = None
        self.tts_available = False
        self.sentiment_analyzer = None
        self.sentiment_available = False
        self.rag = None
        self.use_rag = use_rag and RAG_AVAILABLE
        self.session_state = "idle"
        self.fatigue_history = []       # record of fatigue levels per interaction
        self.adjustment_log = []        # record of when the coach changed the target
        self.metrics = EvaluationMetrics()
        # I initialise all components.
        self._init_whisper()
        self._init_tts()
        self._init_sentiment(sentiment_model)
        if self.use_rag:
            try:
                self.rag = FitnessRAGSystem(use_pdfs=True)
                print("[VoiceCoach] RAG system connected.")
            except Exception as e:
                print(f"[VoiceCoach] RAG init failed: {e}")

    def _init_whisper(self):
        # I load the OpenAI Whisper model for speech recognition.
        try:
            import whisper
            self.whisper_model = whisper.load_model(self.whisper_model_size)
            self.whisper_available = True
            print(f"[VoiceCoach] Whisper ASR loaded ({self.whisper_model_size})")
        except Exception as e:
            print(f"[VoiceCoach] Whisper load error: {e}")

    def _init_tts(self):
        # I initialise the pyttsx3 text‑to‑speech engine and try to select an English voice.
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 160)
            self.tts_engine.setProperty('volume', 0.9)
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            self.tts_available = True
            print("[VoiceCoach] pyttsx3 TTS initialised")
        except Exception as e:
            print(f"[VoiceCoach] TTS init error: {e}")

    def _init_sentiment(self, model_name: str):
        # I load a DistilBERT model fine‑tuned for sentiment analysis.
        try:
            from transformers import pipeline
            self.sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, device=-1)
            self.sentiment_available = True
            print(f"[VoiceCoach] Sentiment analyzer loaded ({model_name})")
        except Exception as e:
            print(f"[VoiceCoach] Sentiment model error: {e}")

    def transcribe(self, audio_file_path: str) -> Dict[str, Any]:
        # I convert an audio file to text using Whisper.
        if not self.whisper_available:
            return {'text': '', 'language': 'en', 'segments': []}
        try:
            src_path = Path(audio_file_path).resolve()
            if not src_path.exists():
                print(f"[VoiceCoach] Audio file does not exist: {src_path}")
                return {'text': '', 'language': 'en', 'segments': []}

            # I read the audio file with soundfile and convert it to mono float32.
            audio_data, sample_rate = sf.read(str(src_path))

            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max()

            # I run Whisper transcription. I set fp16=False for compatibility.
            result = self.whisper_model.transcribe(audio_data, language=None, fp16=False, task="transcribe")
            return {'text': result['text'].strip(), 'language': result.get('language', 'en'),
                    'segments': result.get('segments', [])}
        except Exception as e:
            print(f"[VoiceCoach] Transcription error: {e}")
            return {'text': '', 'language': 'en', 'segments': []}

    def analyze_sentiment_and_fatigue(self, text: str) -> Dict[str, Any]:
        # I use the sentiment pipeline and keyword matching to estimate user state.
        sentiment_label = 'NEUTRAL'
        sentiment_score = 0.5
        if self.sentiment_available and text.strip():
            try:
                res = self.sentiment_analyzer(text[:512])[0]
                sentiment_label = res['label']
                sentiment_score = res['score']
            except Exception:
                pass
        fatigue_level = 0
        detected_keywords = []
        text_lower = text.lower()
        for category, keywords in self.FATIGUE_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    fatigue_level += 1
                    detected_keywords.append(kw)
        return {'sentiment': sentiment_label, 'sentiment_score': sentiment_score,
                'fatigue_level': fatigue_level, 'fatigue_keywords': detected_keywords}

    def classify_intent(self, text: str) -> str:
        # I match the user's text against the intent keyword dictionaries.
        text = text.lower().strip()
        best_intent = 'unknown'
        best_score = 0
        for intent, keywords in self.INTENT_KEYWORDS.items():
            # I sum the lengths of matched keywords to give longer phrases more weight.
            score = sum(len(kw) for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_intent = intent
        if best_intent == 'unknown' and any(kw in text for kw in self.INTENT_KEYWORDS['feedback']):
            best_intent = 'feedback'
        return best_intent

    def _get_paper_citation(self, topic: str) -> str:
        # I try to retrieve a scientific citation from the RAG system.
        if self.rag:
            try:
                docs = self.rag.retrieve_evidence(topic, k=1)
                if docs:
                    return docs[0]['content'][:200] + "..."
            except Exception:
                pass
        # Fallback citations for common topics.
        citations = {
            'fatigue': "Damas et al. (2016) recommend 48h recovery for optimal muscle protein synthesis.",
            'form': "Escamilla et al. (2001) found knee angle <70° increases PCL stress in squats.",
            'volume': "Schoenfeld et al. (2019): ≥10 weekly sets per muscle maximises hypertrophy.",
        }
        return citations.get(topic, "ACSM (2009) suggests adjusting load when RPE exceeds 7/10.")

    def generate_response(self, intent: str, session_context: Optional[Dict] = None,
                          sentiment_info: Optional[Dict] = None) -> str:
        # I generate a spoken response based on the intent and the user's state.
        ctx = session_context or {}
        exercise = ctx.get('exercise', 'exercise').replace('_', ' ')
        reps = ctx.get('reps', 0)
        target_reps = ctx.get('target_reps', 12)
        form_status = ctx.get('form_status', 'correct')
        form_feedback = ctx.get('form_feedback', '')
        base_responses = {
            'start': f"Let's go! Starting {exercise}. Your target is {target_reps} reps.",
            'stop': "Pausing the session. Take your time.",
            'next': f"Moving to next exercise. You completed {reps} reps of {exercise}. Good work!",
            'status': f"You've done {reps} out of {target_reps} reps of {exercise}.",
            'form': f"Your form is {form_status}. {form_feedback}" if form_feedback else f"Form looks {'good' if form_status=='correct' else 'needs correction'}.",
            'rest': "Taking a 30‑second rest. Breathe deeply.",
            'done': f"Session complete! You finished {reps} reps of {exercise}. Great effort!",
            'help': "You can say: start, stop, next, status, form, rest, done.",
            'feedback': "Thanks for letting me know.",
            'unknown': "I didn't catch that. Try 'help' for commands.",
            'easy': "That's great! I'll keep the intensity as planned.",
            'hard': "I hear you. I'm adjusting the target to make it more manageable.",
            'okay': "Good, keep going!",
        }
        response = base_responses.get(intent, base_responses['unknown'])
        # If sentiment or fatigue indicates the user is struggling, I reduce the target.
        if sentiment_info:
            fatigue_level = sentiment_info.get('fatigue_level', 0)
            sentiment = sentiment_info.get('sentiment', 'NEUTRAL')
            if fatigue_level >= 2 or sentiment == 'NEGATIVE' or intent == 'hard':
                if 'target_reps' in ctx and target_reps > 6:
                    new_target = max(6, int(target_reps * 0.8))
                    ctx['target_reps'] = new_target
                    self.adjustment_log.append({'timestamp': time.time(), 'original_target': target_reps,
                                                'new_target': new_target, 'reason': f"Fatigue {fatigue_level}, {sentiment}"})
                    response += f" I've reduced your target to {new_target} reps. " + self._get_paper_citation('fatigue')
                self.fatigue_history.append(fatigue_level)
            elif intent == 'easy' and 'target_reps' in ctx:
                # If the user says it is too easy, I increase the target slightly.
                new_target = min(20, int(target_reps * 1.1))
                ctx['target_reps'] = new_target
                response += f" Since it feels easy, I've increased your target to {new_target} reps."
        # I update the session state based on the intent.
        if intent == 'start':
            self.session_state = "exercising"
        elif intent == 'stop':
            self.session_state = "idle"
        elif intent == 'rest':
            self.session_state = "resting"
        elif intent == 'done':
            self.session_state = "finished"
        return response

    def speak(self, text: str):
        # I use the TTS engine to speak the response aloud.
        if not self.tts_available:
            print(f"[VoiceCoach TTS] {text}")
            return
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[VoiceCoach] TTS error: {e}. Text: {text}")

    def process_command(self, audio_path: Optional[str] = None, text_command: Optional[str] = None,
                        session_context: Optional[Dict] = None) -> Dict[str, Any]:
        # This is the main entry point for processing a user command (voice or text).
        start_time = time.time()
        if audio_path and self.whisper_available:
            trans = self.transcribe(audio_path)
            text = trans['text']
        elif text_command:
            text = text_command.strip()
            trans = {'text': text, 'language': 'en', 'segments': []}
        else:
            return {'error': 'No input provided'}
        sentiment_info = self.analyze_sentiment_and_fatigue(text)
        intent = self.classify_intent(text)
        response = self.generate_response(intent, session_context, sentiment_info)
        self.speak(response)
        latency_ms = (time.time() - start_time) * 1000
        return {'transcription': trans, 'intent': intent, 'sentiment': sentiment_info,
                'response': response, 'session_state': self.session_state,
                'latency_ms': latency_ms, 'adjustment_made': len(self.adjustment_log) > 0}

    def proactive_check(self, session_context: Optional[Dict] = None) -> Optional[Dict]:
        # I proactively ask the user how they are feeling every 5 repetitions.
        ctx = session_context or {}
        reps   = ctx.get('reps', 0)
        target = ctx.get('target_reps', 12)
        exercise = ctx.get('exercise', 'exercise')

        should_ask = False
        question   = ""
        if self.session_state == "exercising" and reps > 0 and reps % 5 == 0 and reps < target:
            should_ask = True
            question   = f"You've done {reps} reps of {exercise}. How are you feeling? Easy, hard, or okay?"
        elif self.session_state == "resting":
            should_ask = True
            question   = "Are you ready to continue? Say yes or no."

        if not should_ask:
            return None

        print(f"\n[Proactive] Coach asks: \"{question}\"")
        self.speak(question)

        # In a real deployment, this would be replaced by voice input.
        user_input = input("Your response (text): ").strip()
        if not user_input:
            return None

        result = self.process_command(text_command=user_input, session_context=ctx)
        return result

    def evaluate_wer(self, test_set: List[Tuple[str, str]]) -> float:
        # I compute the Word Error Rate on a test set of audio files and reference transcriptions.
        try:
            from jiwer import wer
        except ImportError:
            print("jiwer not installed.")
            return -1.0
        hypotheses, references = [], []
        for audio, ref in test_set:
            if not os.path.exists(audio):
                print(f"[WARN] Audio file not found: {audio}")
                continue
            trans = self.transcribe(audio)
            hypotheses.append(trans['text'])
            references.append(ref)
        if not hypotheses:
            return 0.0
        self.metrics.wer = wer(references, hypotheses)
        print(f"[Eval] WER = {self.metrics.wer:.3f}")
        return self.metrics.wer

    def compare_whisper_models(self, test_audio_files: List[str], reference_texts: List[str]) -> Dict[str, Dict[str, float]]:
        # I compare the three main Whisper model sizes (tiny, base, small) in terms of WER and latency.
        results = {}
        original_model = self.whisper_model
        original_size = self.whisper_model_size
        for size in ['tiny', 'base', 'small']:
            print(f"\n[Eval] Testing Whisper {size}...")
            self.whisper_model_size = size
            self._init_whisper()
            if not self.whisper_available:
                results[size] = {'wer': -1.0, 'avg_latency_ms': -1.0}
                continue
            latencies, transcripts = [], []
            for audio in test_audio_files:
                if not os.path.exists(audio):
                    print(f"[WARN] Audio file not found: {audio}")
                    continue
                start = time.time()
                trans = self.transcribe(audio)
                latencies.append((time.time() - start) * 1000)
                transcripts.append(trans['text'])
            try:
                from jiwer import wer
                w = wer(reference_texts, transcripts)
            except:
                w = -1.0
            results[size] = {'wer': w, 'avg_latency_ms': np.mean(latencies) if latencies else -1.0}
        self.whisper_model_size = original_size
        self.whisper_model = original_model
        self.metrics.model_comparison = results
        return results

    def evaluate_sentiment_accuracy(self, validation_set: List[Tuple[str, str]]) -> float:
        # I evaluate how well the sentiment analyzer performs on a labeled validation set.
        if not self.sentiment_available:
            return -1.0
        correct = 0
        for text, true_label in validation_set:
            info = self.analyze_sentiment_and_fatigue(text)
            if info['sentiment'] == true_label:
                correct += 1
        acc = correct / len(validation_set) if validation_set else 0.0
        self.metrics.sentiment_accuracy = acc
        print(f"[Eval] Sentiment accuracy = {acc:.3f}")
        return acc

    def measure_end_to_end_latency(self, test_commands: List[str], n_runs: int = 5) -> float:
        # I measure the average end‑to‑end latency from command to spoken response.
        latencies = []
        for cmd in test_commands:
            for _ in range(n_runs):
                res = self.process_command(text_command=cmd)
                latencies.append(res.get('latency_ms', 0))
        avg = np.mean(latencies) if latencies else 0.0
        self.metrics.avg_latency_ms = avg
        print(f"[Eval] Avg end‑to‑end latency = {avg:.2f} ms")
        return avg

    def save_evaluation_report(self):
        # I save all collected metrics and generate a comparison plot for Whisper models.
        report_dir = REPORTS_DIR / "voice_coach"
        plots_dir = PLOTS_DIR / "voice_coach"
        report_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        total_interactions = len(self.fatigue_history)
        fatigue_triggers = sum(1 for f in self.fatigue_history if f >= 2)
        self.metrics.fatigue_trigger_rate = fatigue_triggers / total_interactions if total_interactions > 0 else 0.0
        report = {
            'wer': self.metrics.wer,
            'sentiment_accuracy': self.metrics.sentiment_accuracy,
            'avg_latency_ms': self.metrics.avg_latency_ms,
            'fatigue_trigger_rate': self.metrics.fatigue_trigger_rate,
            'model_comparison': self.metrics.model_comparison,
            'adjustment_log': self.adjustment_log[-10:],
        }
        with open(report_dir / "evaluation_metrics.json", 'w') as f:
            json.dump(report, f, indent=2)
        if self.metrics.model_comparison:
            models = list(self.metrics.model_comparison.keys())
            wer_vals = [self.metrics.model_comparison[m]['wer'] for m in models]
            lat_vals = [self.metrics.model_comparison[m]['avg_latency_ms'] for m in models]
            fig, ax1 = plt.subplots(figsize=(8,5))
            ax1.bar(models, wer_vals, color='steelblue', alpha=0.7, label='WER')
            ax1.set_ylabel('Word Error Rate', color='steelblue')
            ax2 = ax1.twinx()
            ax2.plot(models, lat_vals, 'ro-', label='Latency (ms)')
            ax2.set_ylabel('Latency (ms)', color='red')
            plt.title('Whisper Model Comparison')
            fig.tight_layout()
            plt.savefig(plots_dir / "whisper_comparison.png", dpi=120)
            plt.close()
        print(f"[VoiceCoach] Evaluation report saved to {report_dir}")


if __name__ == "__main__":
    print("[VoiceCoach] FitCoach AI — Sentiment-Aware Voice Coach (proactive demo)")

    coach = VoiceCoach(whisper_model_size="base", use_rag=True)

    print("\n[Demo] Starting a simulated workout session with proactive check-ins...")
    ctx = {'exercise': 'squat', 'target_reps': 12, 'reps': 0, 'form_status': 'correct'}
    
    # I start the session.
    res = coach.process_command(text_command="start", session_context=ctx)
    print(f"\n  You: 'start'")
    print(f"  Intent: {res['intent']} | Response: {res['response']}")

    # I simulate performing 12 repetitions, with a proactive check every 5 reps.
    for rep in range(1, 13):
        ctx['reps'] = rep
        print(f"\n[Rep {rep}/{ctx['target_reps']}]")

        proactive_result = coach.proactive_check(ctx)
        if proactive_result:
            print(f"  User responded: '{proactive_result['transcription']['text']}'")
            print(f"  Intent: {proactive_result['intent']} | Sentiment: {proactive_result['sentiment']['sentiment']}")
            print(f"  Coach: {proactive_result['response']}")
            if proactive_result.get('adjustment_made'):
                print(f"  Target adjusted to {ctx['target_reps']} reps.")
        time.sleep(0.5)

    # I finish the session.
    res = coach.process_command(text_command="done", session_context=ctx)
    print(f"\n  You: 'done'")
    print(f"  Response: {res['response']}")

    print("\n[Eval] Running quantitative evaluations...")

    # I look for test audio files to evaluate WER and compare Whisper models.
    test_audio_dir = PROJECT_ROOT / "data" / "test_audio"
    print(f"[VoiceCoach] Looking for test audio in: {test_audio_dir}")
    if test_audio_dir.exists():
        audio_files = list(test_audio_dir.glob("*.wav"))
        print(f"[VoiceCoach] Found {len(audio_files)} .wav files")
        if audio_files:
            transcriptions_file = test_audio_dir / "transcriptions.json"
            if transcriptions_file.exists():
                with open(transcriptions_file, 'r', encoding='utf-8') as f:
                    ref_map = json.load(f)
                test_set = [(str(f.resolve()), ref_map.get(f.name, "")) for f in audio_files if f.name in ref_map]
                if test_set:
                    coach.evaluate_wer(test_set)
                    sample_audio = [str(f.resolve()) for f in audio_files[:10]]
                    sample_refs  = [ref_map.get(f.name, "") for f in audio_files[:10]]
                    coach.compare_whisper_models(sample_audio, sample_refs)
                else:
                    print("[VoiceCoach] Warning: no references found for the audio files.")
            else:
                print("[VoiceCoach] Warning: transcriptions.json not found — skipping WER evaluation.")
    else:
        print(f"[VoiceCoach] Warning: test audio folder not found at {test_audio_dir}")

    # I evaluate sentiment accuracy on a small validation set.
    val_set = [
        ("I feel great", "POSITIVE"), ("This is too hard", "NEGATIVE"),
        ("I'm exhausted", "NEGATIVE"), ("Not bad", "POSITIVE"),
        ("I could do this all day", "POSITIVE"), ("That was easy", "POSITIVE"),
        ("I'm struggling", "NEGATIVE"), ("My legs are killing me", "NEGATIVE"),
        ("So tired", "NEGATIVE"), ("Let's go", "POSITIVE"),
    ]
    coach.evaluate_sentiment_accuracy(val_set)
    coach.measure_end_to_end_latency(["start", "status", "I'm tired"], n_runs=3)
    coach.save_evaluation_report()

    print("[VoiceCoach] Evaluation complete. All metrics saved.")