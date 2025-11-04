# app.py
import streamlit as st
import pandas as pd
import torch
import os
from collections import defaultdict

# Import the classes from your AI file
from hangman_ai import HangmanGame, FastAgent

# --- Helper Function for ASCII Art ---
def get_hangman_art(wrong_guesses):
    """Returns ASCII art for the hangman stages."""
    stages = [
        # 0 wrong
        """
           -----
           |   |
           |
           |
           |
           |
        ---------
        """,
        # 1 wrong
        """
           -----
           |   |
           |   O
           |
           |
           |
        ---------
        """,
        # 2 wrong
        """
           -----
           |   |
           |   O
           |   |
           |
           |
        ---------
        """,
        # 3 wrong
        """
           -----
           |   |
           |   O
           |  /|
           |
           |
        ---------
        """,
        # 4 wrong
        """
           -----
           |   |
           |   O
           |  /|\\
           |
           |
        ---------
        """,
        # 5 wrong
        """
           -----
           |   |
           |   O
           |  /|\\
           |  /
           |
        ---------
        """,
        # 6 wrong (Game Over)
        """
           -----
           |   |
           |   O
           |  /|\\
           |  / \\
           |
        ---------
        """
    ]
    return stages[min(wrong_guesses, len(stages) - 1)]

# --- Caching the AI Model ---
@st.cache_resource
def load_agent():
    """Loads the pre-trained agent. Caches it for performance."""
    if not os.path.exists('final_agent.pkl'):
        return None
    
    print("Loading agent from final_agent.pkl...")
    agent = FastAgent()
    try:
        agent.load('final_agent.pkl')
        agent.epsilon = 0.0 # Ensure no random moves during demo
        print("Agent loaded successfully.")
    except Exception as e:
        print(f"Error loading agent: {e}")
        return None
    return agent

# --- Main App ---
st.set_page_config(layout="wide")
st.title("🤖 ML Hangman AI Showcase")
st.markdown("Watch the hybrid N-Gram/RL agent solve Hangman. Enter a word and press 'Start Game'!")

# Load the agent
agent = load_agent()

if agent is None:
    st.error(
        "**Error: `final_agent.pkl` not found.**\n\n"
        "Please run the training script first by typing this in your terminal:\n\n"
        "`python hangman_ai.py`"
    )
    st.stop()

# --- Sidebar for Game Setup ---
st.sidebar.header("Game Setup")
secret_word = st.sidebar.text_input("Enter a secret word for the AI:").strip().lower()

if st.sidebar.button("Start Game", disabled=(not secret_word)):
    if all('a' <= c <= 'z' for c in secret_word):
        # Initialize game state
        st.session_state.game = HangmanGame(secret_word)
        st.session_state.game_active = True
        st.session_state.ai_thoughts = []
        st.session_state.last_action = ""
    else:
        st.sidebar.error("Please enter only letters (a-z).")

# --- Main Game Area ---
if 'game_active' in st.session_state and st.session_state.game_active:
    game = st.session_state.game

    # Layout for game display
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Game Status")
        # Display Hangman ASCII art
        st.code(get_hangman_art(game.wrong_guesses), language=None)
        
        # Display game state
        st.markdown(f"**Wrong Guesses:** {game.wrong_guesses} / {game.max_wrong}")
        wrong_letters = sorted(list(game.guessed_letters - set(game.word)))
        st.markdown(f"**Wrong Letters:** `{', '.join(wrong_letters) or 'None'}`")

    with col2:
        st.subheader("Word to Guess")
        # Display the word pattern (e.g., _ _ L L _)
        display_word = " ".join(game.get_state())
        st.header(f"`{display_word}`")
        
        if st.session_state.last_action:
            st.info(f"AI just guessed: **{st.session_state.last_action.upper()}**")

        # --- AI's "Brain" Visualization ---
        st.subheader("🧠 AI's Brain: Letter Probabilities")
        
        if 'ai_thoughts' in st.session_state and st.session_state.ai_thoughts:
            # Get the last set of probabilities calculated
            probs_df = st.session_state.ai_thoughts[-1]
            st.write("The AI weighs 3 models to pick the best letter. Here's what it was thinking:")
            
            # Use a bar chart to visualize
            st.bar_chart(probs_df)
        else:
            st.write("Click 'Let AI Guess' to see what the AI is thinking...")

    st.divider()

    # --- Game Controls ---
    col_btn, col_msg = st.columns([1, 3])

    if game.is_won():
        st.success(f"**AI Won!** 🥳 The word was: **{game.word}**")
        st.session_state.game_active = False
    elif game.is_lost():
        st.error(f"**AI Lost!** 😞 The word was: **{game.word}**")
        st.session_state.game_active = False
    else:
        # The button that runs one step of the AI
        if col_btn.button("🤖 Let AI Guess Next Letter"):
            
            # 1. Get AI's action
            action = agent.get_action(game, training=False)
            
            if action:
                # 2. Get AI's "thoughts" *before* it makes the guess
                available = game.get_available_letters()
                pattern_str = game.get_state()
                guessed = game.guessed_letters
                
                # Get probs from each model for visualization
                pattern_probs = agent.pattern.get_letter_probs(pattern_str, guessed, available)
                ngram_probs = agent.ngram.get_letter_probs(pattern_str, available)
                freq_probs = agent.freq.get_letter_probs(pattern_str, game.word_length, available)

                # Combine into a DataFrame for the chart
                prob_data = defaultdict(dict)
                for letter in available:
                    prob_data[letter]["1. Pattern Model"] = pattern_probs.get(letter, 0)
                    prob_data[letter]["2. N-Gram Model"] = ngram_probs.get(letter, 0)
                    prob_data[letter]["3. Frequency Model"] = freq_probs.get(letter, 0)
                
                probs_df = pd.DataFrame(prob_data).T
                # Sort by a weighted sum to show best guesses on top
                probs_df['Combined'] = (probs_df["1. Pattern Model"] * 0.6 + 
                                        probs_df["2. N-Gram Model"] * 0.3 + 
                                        probs_df["3. Frequency Model"] * 0.1)
                probs_df = probs_df.sort_values('Combined', ascending=False)
                
                st.session_state.ai_thoughts.append(probs_df.drop('Combined', axis=1))

                # 3. Make the guess and update state
                game.guess(action)
                st.session_state.game = game
                st.session_state.last_action = action
                
                # 4. Rerun the script to update the UI
                st.rerun()
            else:
                col_msg.error("AI could not find an action.")
else:
    st.info("Enter a word in the sidebar and click 'Start Game' to begin.")