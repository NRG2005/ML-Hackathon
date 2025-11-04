🤖 Hybrid ML Hangman AI Solver
This project is a high-performance Hangman solver built for the (Hackathon Name). It uses a unique hybrid model that combines traditional N-gram and frequency-based models with a Deep Q-Network (DQN) for reinforcement learning.

The project includes an interactive Streamlit web app to visualize the AI's decision-making process in real-time, showing the letter probabilities calculated by each of its "brain" components.

🎥 Demo
(You should record a quick GIF of the Streamlit app in action and save it as demo.gif in this folder)

✨ Features
Hybrid Reasoning Engine: The AI doesn't rely on one strategy. It blends three heuristic models:

FastPatternMatcher: Filters a 50,000-word corpus against the current pattern (e.g., _ _ A _ _).

FastNGramModel: Uses unigram, bigram, and trigram statistics to predict the most likely letter in a blank.

FastFrequencyModel: Uses global and length-specific letter frequencies as a baseline.

Reinforcement Learning (DQN): A MicroDQN (a tiny deep Q-network) acts as a "meta-learner." It learns from its mistakes and successes to decide the optimal blend of the heuristic models at any given game state.

Interactive Frontend: A simple and "cool" Streamlit (app.py) app that lets you:

Enter any secret word.

Watch the AI play the game step-by-step.

Visualize the AI's "Brain": A live bar chart shows the probabilities each model assigns to available letters, so you can see why the AI chose 'E' over 'T'.

🚀 How to Run
1. Prerequisites
Python 3.8+

A data folder containing corpus.txt and test_corpus.txt.

2. Setup
Clone the repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Create and activate a virtual environment (Recommended):

Bash

# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
Install the required libraries:

Bash

pip install torch numpy streamlit pandas tqdm
3. Step 1: Train the AI Model
You must run the training script once to generate the N-gram models and the trained DQN agent file (final_agent.pkl).

Bash

python hangman_ai.py
This will:

Load the data/corpus.txt.

Train the Pattern, N-Gram, and Frequency models.

Run 10,000 episodes of RL training to create final_agent.pkl.

Run a final test using data/test_corpus.txt.

4. Step 2: Run the Streamlit Web App
Once final_agent.pkl exists, you can start the frontend:

Bash

streamlit run app.py
Your web browser will automatically open to http://localhost:8501.

📂 Project Structure
hangman-ai-solver/
│
├── data/
│   ├── corpus.txt         # 50k+ word training corpus
│   └── test_corpus.txt    # Test corpus for final evaluation
│
├── hangman_ai.py          # Core AI logic, all models, and the training script
├── app.py                 # The Streamlit frontend
├── final_agent.pkl        # (Generated after running hangman_ai.py)
└── README.md              # This file
🧠 Model Architecture
The core of this AI is in the FastAgent's get_action method. It dynamically blends the outputs of its models based on the game's progress.

Heuristic Models:

FastPatternMatcher: This is the most powerful model. It filters the entire word list (candidates) to find words that could match the current pattern (e.g., _ P P _ E) and have no "wrong" letters. It then counts the frequency of letters appearing in the blank spots of these matching words.

FastNGramModel: This model is excellent at the start of the game when the pattern model has too many candidates. It looks at the letters around a blank (e.g., ^__ or T _ E) and uses pre-computed N-gram (1, 2, and 3-letter) probabilities to guess what should go in the middle.

FastFrequencyModel: A simple baseline that uses global letter frequency ('E' is most common) and frequency based on word length.

Reinforcement Learning (DQN):

A MicroDQN (a tiny 1-hidden-layer neural net) is trained to predict the Q-value (expected future reward) of guessing any given letter.

The state given to the DQN is a 35-dimension vector describing the game (e.g., number of wrong guesses, ratio of blanks, vowel-to-consonant ratio, etc.).

The reward function heavily penalizes wrong guesses (-2.5) and repeated guesses (-8.0), and gives large bonuses for correct guesses (+4.0) and winning (+25.0).

The Hybrid Blend: The agent combines these models using dynamic weights.

When many blanks are left (blanks_ratio > 0.6), it trusts the N-Gram and Frequency models more.

When few blanks are left, it trusts the FastPatternMatcher almost exclusively, as the list of candidate words becomes very small and accurate.

The RL model's Q-values are normalized and added as a small, final weight, acting as a "policy polisher" to break ties and nudge the agent away from moves that seem good but have historically led to losses.
