# FitCoach AI

Hi, I'm Alejandro. This is FitCoach AI, my final project for the second semester of my third year in Intelligent Systems Engineering. I wanted to build something that really used everything I had learned, not just one AI technique but several of them working together. The result is a virtual personal coach that watches you exercise, counts your repetitions, checks your form, plans your workouts, and talks to you. It runs on a normal webcam and microphone, no special sensors needed.

## Why I built this

When I started this project, I looked at what already existed. Most fitness apps do one thing: they classify the exercise, or they count reps, or they give you a static workout plan. A real coach does all of these things at the same time, and more importantly, they change the plan when you are tired or struggling. I wanted to see if I could combine four different areas of AI—Computer Vision, Intelligent Systems, Advanced Machine Learning, and Speech & NLP—into one system that feels like a human trainer.

The hard part was not making each module work on its own. The hard part was making them talk to each other. For example, if the voice coach hears that I am tired, it should tell the workout planner to lower the target repetitions. If the computer vision module sees that my squat form is bad, the explainable AI module should tell me *why* it is bad, not just that it is bad. That integration was the real challenge, and it is what makes this project different from just a collection of separate scripts.

## How it works (the big picture)

I designed FitCoach AI as four modules that share a central state. Think of the central state as a whiteboard where every module can read and write. One module writes "user is doing squats, 8 reps done, fatigue level medium" and another module reads that and decides to suggest a rest.

- **Perception (Computer Vision):** I use YOLOv8 to find the person in the frame and MediaPipe to get 33 body landmarks. I compared five neural network architectures (MLP, CNN1D, SimpleRNN, LSTM, and BiLSTM with attention) to classify 8 general exercises and 10 rehabilitation exercises. For counting reps, I built seven different methods and ended up creating my own called HybridCounter. It uses a Butterworth filter to clean the noisy keypoint signal before detecting peaks. For posture, I use both a rule‑based knowledge base (with real citations from biomechanics papers) and a neural network that I trained on joint angles. I chose joint angles instead of raw coordinates because angles do not change if the person stands closer or farther from the camera.

- **Reasoning (Intelligent Systems):** I modeled the fitness journey as a search problem. An A* planner estimates the fastest timeline to reach a goal (like going from 22% to 15% body fat). The heuristic is admissible, so A* is guaranteed to find the optimal plan. Then a Simulated Annealing optimizer builds a balanced weekly schedule. I defined a cost function that penalizes training the same muscle group two days in a row, muscle imbalance, and not enough rest. I ran 30 trials and used a Wilcoxon signed‑rank test to prove that Simulated Annealing is statistically better than Tabu Search and Hill Climbing.

- **Learning (Advanced Machine Learning):** I wanted the coach to adapt in real time, so I built a custom Gym environment that simulates a user's state (fatigue, success rate, errors). I trained two reinforcement learning agents: DQN and REINFORCE. The DQN agent failed. I kept it in the project because that failure taught me a lot about sparse rewards and the difficulty of value‑based RL in custom environments. The REINFORCE agent worked much better, which makes sense because policy gradient methods handle sparse rewards more gracefully. I also used SHAP and LIME to explain the posture classifier's decisions. I found that SHAP Gradient is fast enough for real‑time feedback while still agreeing with the slower but more accurate Kernel SHAP.

- **Communication (Speech & NLP):** The voice coach uses Whisper to transcribe speech, DistilBERT to analyze sentiment, and pyttsx3 to speak. I added keyword matching in English and Spanish so the user can say things like "start" or "I'm tired". If the user sounds tired or negative, the coach automatically lowers the target repetitions. I also built a Retrieval‑Augmented Generation (RAG) system with 36 peer‑reviewed papers. When the user asks for a workout plan, the system retrieves the most relevant evidence and asks the LLM to base its answer only on that evidence. I tested three prompting strategies and found that "evidence‑first" produces the most reliable and citable answers.

## Decisions I made and why

I did not just copy code from tutorials. Every choice had a reason.

- **Why angles instead of raw keypoints?** Raw coordinates change if you stand closer or farther from the camera. Angles are invariant to distance and body size. They also make the posture feedback explainable: I can tell you *"your knee angle is 45°, try to keep it above 60°."*

- **Why a Butterworth filter for the HybridCounter?** MediaPipe keypoints have high‑frequency jitter. A Butterworth filter has a flat passband, so it removes noise without distorting the shape of the movement. It works better than a moving average or a Kalman filter for this specific task.

- **Why A\* and Simulated Annealing?** A* gives the optimal macro‑timeline because the heuristic is admissible. Simulated Annealing escapes local optima that trap simpler algorithms like Hill Climbing. The statistical test proved it.

- **Why RAG with evidence‑first prompting?** Standard LLMs hallucinate. For fitness advice, that is dangerous. By forcing the model to list the retrieved evidence before answering, I made the plans verifiable and safe.

- **Why did I keep the failed DQN agent?** Because it is an honest result. It shows I understand the limitations of value‑based RL in custom environments. In my paper, I discuss this openly as a limitation and a lesson learned.

## What I learned

This project taught me that integration is harder than building individual models. I had to solve real problems: GPU memory conflicts between TensorFlow and MediaPipe (I fixed it by forcing TensorFlow to the CPU), massive class imbalance in the dataset (class weights and data augmentation), and how to stop the LLM from making up unsafe exercises (RAG + evidence‑first prompts). I also learned that sometimes a simple signal processing trick beats a complex deep learning model when you do not have millions of training videos.

Most importantly, I learned to be honest about what did not work. The DQN failure is not hidden; it is part of the story. It shows that I can analyze why something fails and use that knowledge to choose a better approach.

## Final thoughts

FitCoach AI is not a commercial product, but it is a working proof of concept that combines perception, planning, learning, and communication. I am proud of the integration, the scientific grounding of the advice, and the honest discussion of both successes and failures. I hope this project shows that with careful design, different AI techniques can work together to build something useful and trustworthy.

– Alejandro Varela
