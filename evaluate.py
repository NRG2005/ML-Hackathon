import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import random
from tqdm import tqdm 
# This line in agent.py works because Hackathon.py is in the same folder
from Hackathon import HMMOracle # A nice progress bar library (install with: pip install tqdm)

# --- Import our custom classes ---
from Hackathon import HMMOracle
from agent import DQNAgent, HangmanEnvironment, load_word_list
from agent import STATE_SIZE, ACTION_SIZE, ALPHABET, letter_to_index, index_to_letter

# --- Define File Paths ---
# Make sure these paths are correct for your system
DATA_FOLDER = "/Users/nihalraviganesh/Documents/ML_Hackathon/Data"
TEST_PATH = os.path.join(DATA_FOLDER, "test.txt") # <-- We evaluate on this file

# --- Model File Paths ---
HMM_MODEL_PATH = "hmm_model.json"
DQN_MODEL_PATH = "dqn_hangman_model" # <-- Path to the saved Keras model

# --- Evaluation Constants ---
NUM_GAMES_TO_PLAY = 2000
STARTING_LIVES = 6

class Evaluator:
    """
    This class loads the trained agent and HMM, plays the
    specified number of games, and calculates the final score.
    """
    def __init__(self, hmm_model_path, dqn_model_path, test_word_list):
        self.test_words = test_word_list
        if not self.test_words:
            raise ValueError("Test word list is empty. Check TEST_PATH.")

        print("Loading HMM Oracle (Part 1)...")
        self.hmm = HMMOracle()
        if not self.hmm.load_model(hmm_model_path):
            raise FileNotFoundError("Could not load HMM model. Run hmm_oracle.py.")

        print("Loading DQN Agent (Part 2)...")
        self.agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
        self.agent.load(dqn_model_path)
        
        # Set agent to pure exploitation (no random guesses)
        self.agent.epsilon = 0.0 
        print("Agent loaded successfully. Running in exploitation-only mode.")

    def run_evaluation(self, num_games):
        """
        Runs the full evaluation against the test dataset.
        """
        total_wins = 0
        total_wrong_guesses = 0
        total_repeated_guesses = 0 # This should be 0 if our agent is smart

        print(f"\n--- Starting Evaluation: Playing {num_games} games ---")
        
        # Use tqdm for a progress bar
        for game_num in tqdm(range(num_games)):
            # Per the problem statement, we evaluate on the test set.
            # We can sample with replacement to play 2000 games.
            secret_word = random.choice(self.test_words)
            
            # --- Initialize new game ---
            pattern = "_" * len(secret_word)
            lives_left = STARTING_LIVES
            guessed_letters = set()
            game_over = False
            word_solved = False
            
            game_wrong_guesses = 0
            game_repeated_guesses = 0

            while not game_over:
                # 1. Get the current state
                hmm_probs = self.hmm.get_letter_probabilities(pattern, guessed_letters)
                state = self.agent.format_state(pattern, guessed_letters, lives_left, hmm_probs)
                
                # 2. Get Q-values from the DQN
                q_values = self.agent.model.predict(state, verbose=0)[0]
                
                # 3. Mask already-guessed letters
                for letter in ALPHABET:
                    if letter in guessed_letters:
                        q_values[letter_to_index(letter)] = -np.inf
                        
                # 4. Choose the best possible action (no exploration)
                action_index = np.argmax(q_values)
                action_letter = index_to_letter(action_index)

                # --- Simulate the game step ---
                
                # This check should be redundant if the Q-value masking works,
                # but we'll count it just in case.
                if action_letter in guessed_letters:
                    game_repeated_guesses += 1
                    # A good agent shouldn't do this, but we don't end the game
                    continue 
                
                guessed_letters.add(action_letter)

                if action_letter in secret_word:
                    # Correct guess
                    new_pattern = list(pattern)
                    for i, char in enumerate(secret_word):
                        if char == action_letter:
                            new_pattern[i] = action_letter
                    pattern = "".join(new_pattern)
                    
                    if "_" not in pattern:
                        word_solved = True
                        game_over = True
                else:
                    # Wrong guess
                    lives_left -= 1
                    game_wrong_guesses += 1
                    
                    if lives_left <= 0:
                        game_over = True
            
            # --- End of game, record stats ---
            if word_solved:
                total_wins += 1
            total_wrong_guesses += game_wrong_guesses
            total_repeated_guesses += game_repeated_guesses

        # --- All games finished, calculate final score ---
        success_rate = total_wins / num_games
        
        # This is the formula from Problem_Statement.pdf 
        final_score = (success_rate * 2000) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)

        print("\n--- Evaluation Complete ---")
        print(f"Total Games Played: {num_games}")
        print("-" * 25)
        print(f"Total Wins:         {total_wins} ({success_rate * 100:.2f}%)")
        print(f"Total Wrong Guesses:  {total_wrong_guesses}")
        print(f"Total Repeated Guesses: {total_repeated_guesses}")
        print("-" * 25)
        print(f"FINAL SCORE:        {final_score:.2f}")


if __name__ == "__main__":
    # 1. Load the *test* words
    test_word_list = load_word_list(TEST_PATH)
    
    if test_word_list:
        # 2. Initialize the Evaluator
        evaluator = Evaluator(HMM_MODEL_PATH, DQN_MODEL_PATH, test_word_list)
        
        # 3. Run the evaluation and get the score
        evaluator.run_evaluation(NUM_GAMES_TO_PLAY)
    else:
        print(f"Could not start evaluation. Failed to load words from {TEST_PATH}")