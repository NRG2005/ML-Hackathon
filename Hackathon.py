import collections
import re
import json
import os

class HMMOracle:
    """
    This class acts as the "HMM Oracle".
    
    It works by:
    1.  Training: Reading the entire corpus and grouping all words by length.
    2.  Querying: When asked for probabilities, it filters its word list to 
        find all words that *match* the current pattern.
    3.  Caching: Stores results of previous queries for speed.
    """
    
    def __init__(self):
        self.words_by_length = collections.defaultdict(list)
        self.alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self._regex_cache = {} 
        
        # --- NEW: Caching for probabilities ---
        self.probability_cache = {}

    def train(self, corpus_file_path):
        """
        Trains the model by reading the corpus file and organizing
        words by their length.
        """
        print(f"Starting training on {corpus_file_path}...")
        try:
            with open(corpus_file_path, 'r') as f:
                for line in f:
                    # Clean and standardize the word
                    word = line.strip().upper()
                    
                    # Ensure it's a valid, alphabetic word
                    if word and word.isalpha():
                        self.words_by_length[len(word)].append(word)
            
            print(f"Training complete. Found words for {len(self.words_by_length)} different lengths.")
            
        except FileNotFoundError:
            print(f"ERROR: Corpus file not found at {corpus_file_path}")
            print("Please make sure the absolute path is correct.")
        except Exception as e:
            print(f"An error occurred during training: {e}")

    def save_model(self, model_path="hmm_model.json"):
        """
        Saves the trained word lists to a JSON file for persistence.
        """
        print(f"Saving model to {model_path}...")
        try:
            with open(model_path, 'w') as f:
                json.dump(self.words_by_length, f, indent=2)
            print(f"Model saved successfully to {os.path.abspath(model_path)}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, model_path="hmm_model.json"):
        """
        Loads a previously saved model from a JSON file.
        """
        print(f"Loading model from {model_path}...")
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            print("Please run the training process first.")
            return False
            
        try:
            with open(model_path, 'r') as f:
                # Load the dictionary from JSON
                data = json.load(f)
                
                # JSON saves all keys as strings, so we must convert
                # word lengths (keys) back to integers.
                self.words_by_length = {int(k): v for k, v in data.items()}
            print(f"Model loaded successfully. {sum(len(v) for v in self.words_by_length.values())} words loaded.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def _get_pattern_regex(self, pattern, revealed_letters):
        """
        Creates a compiled regex to find matching words.
        """
        key = (pattern, tuple(sorted(list(revealed_letters))))
        if key in self._regex_cache:
            return self._regex_cache[key]
        
        avoid_str = "".join(revealed_letters)
        
        # Create the regex pattern
        regex_pattern = ""
        for char in pattern:
            if char == '_':
                # Match any character *except* the ones already revealed
                if avoid_str:
                    regex_pattern += f"[^{avoid_str}]"
                else:
                    regex_pattern += "." # No revealed letters yet
            else:
                # Match the exact revealed letter
                regex_pattern += re.escape(char)
        
        final_regex = re.compile(f"^{regex_pattern}$")
        self._regex_cache[key] = final_regex
        return final_regex

    def get_letter_probabilities(self, pattern_str, guessed_letters_set):
        """
        The core "oracle" function.
        """
        
        # --- NEW: Check cache first ---
        cache_key = (pattern_str, tuple(sorted(list(guessed_letters_set))))
        if cache_key in self.probability_cache:
            return self.probability_cache[cache_key]
        
        word_len = len(pattern_str)
        if word_len not in self.words_by_length:
            return {} 

        available_guesses = self.alphabet - guessed_letters_set
        revealed_letters = set(c for c in pattern_str if c != '_')
        wrong_guesses = guessed_letters_set - revealed_letters

        word_list = self.words_by_length[word_len]
        pattern_regex = self._get_pattern_regex(pattern_str, revealed_letters)

        matching_words = []
        for word in word_list:
            if pattern_regex.match(word):
                if not any(wrong_guess in word for wrong_guess in wrong_guesses):
                    matching_words.append(word)
        
        letter_counts = collections.Counter()
        if not matching_words:
            self.probability_cache[cache_key] = {} # Cache the empty result
            return {}

        for word in matching_words:
            for i, char in enumerate(word):
                if pattern_str[i] == '_' and char in available_guesses:
                    letter_counts[char] += 1

        total_valid_letters = sum(letter_counts.values())
        if total_valid_letters == 0:
            self.probability_cache[cache_key] = {} # Cache the empty result
            return {}
            
        probabilities = {
            char: count / total_valid_letters
            for char, count in letter_counts.items()
        }
        
        # --- NEW: Save to cache before returning ---
        self.probability_cache[cache_key] = probabilities
        return probabilities

# ======================================================================
# Main execution block 
# ======================================================================
if __name__ == "__main__":
    
    # --- Local Paths ---
    DATA_FOLDER = "/Users/nihalraviganesh/Documents/ML_Hackathon/Data"
    CORPUS_PATH = os.path.join(DATA_FOLDER, "corpus.txt")
    MODEL_JSON_PATH = "hmm_model.json" 
    
    print("-" * 30)
    print("PART 1: HMM ORACLE TRAINING")
    print("-" * 30)
    
    oracle_trainer = HMMOracle()
    oracle_trainer.train(CORPUS_PATH)
    oracle_trainer.save_model(MODEL_JSON_PATH)
    
    print("\n" * 2)
    print("HMM Model saved.")