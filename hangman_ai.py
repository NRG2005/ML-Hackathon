# hangman_ai.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HangmanGame:
    # --- FIXED: Was _init_ ---
    def __init__(self, word):
        self.word = word.lower()
        self.word_length = len(word)
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.max_wrong = 6
        self.repeated_guesses = 0
        
    def guess(self, letter):
        letter = letter.lower()
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            return False, False
        self.guessed_letters.add(letter)
        if letter in self.word:
            return True, self.is_won()
        else:
            self.wrong_guesses += 1
            return False, self.is_lost()
    
    def get_state(self):
        return ''.join([c if c in self.guessed_letters else '_' for c in self.word])
    
    def is_won(self):
        return all(c in self.guessed_letters for c in self.word)
    
    def is_lost(self):
        return self.wrong_guesses >= self.max_wrong
    
    def is_done(self):
        return self.is_won() or self.is_lost()
    
    def get_available_letters(self):
        return set('abcdefghijklmnopqrstuvwxyz') - self.guessed_letters

class FastPatternMatcher:
    # --- FIXED: Was _init_ ---
    def __init__(self):
        self.words_by_length = defaultdict(list)
        
    def train(self, words):
        print("Training Pattern Matcher...")
        for word in tqdm(words):
            word = word.lower()
            self.words_by_length[len(word)].append(word)
    
    def get_letter_probs(self, pattern, guessed_letters, available_letters):
        candidates = self.words_by_length.get(len(pattern), [])
        if not candidates or not available_letters:
            return {letter: 1.0/max(len(available_letters), 1) for letter in available_letters}
        
        wrong_letters = guessed_letters - set(pattern)
        matches = []
        
        for word in candidates:
            if any(letter in word for letter in wrong_letters):
                continue
            if all(p == '_' or p == w for p, w in zip(pattern, word)):
                matches.append(word)
                if len(matches) > 500:  # Speed limit
                    break
        
        if not matches:
            return {letter: 1.0/len(available_letters) for letter in available_letters}
        
        letter_counts = defaultdict(float)
        for word in matches:
            for i, char in enumerate(word):
                if pattern[i] == '_' and char in available_letters:
                    letter_counts[char] += 1
        
        total = sum(letter_counts.values())
        if total == 0:
            return {letter: 1.0/len(available_letters) for letter in available_letters}
        
        return {letter: letter_counts[letter] / total for letter in available_letters}

class FastNGramModel:
    # --- FIXED: Was _init_ ---
    def __init__(self):
        self.unigram = defaultdict(int)
        self.bigram = defaultdict(lambda: defaultdict(int))
        self.trigram = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.total = 0
        
    def train(self, words):
        print("Training N-Gram Model...")
        for word in tqdm(words):
            word = word.lower()
            padded = f"^{word}$"
            
            for char in word:
                self.unigram[char] += 1
                self.total += 1
            
            for i in range(len(padded) - 1):
                if padded[i+1].isalpha():
                    self.bigram[padded[i]][padded[i+1]] += 1
            
            for i in range(len(padded) - 2):
                if padded[i+2].isalpha():
                    self.trigram[padded[i]][padded[i+1]][padded[i+2]] += 1
    
    def get_letter_probs(self, pattern, available_letters):
        probs = defaultdict(float)
        state = f"^{pattern}$"
        
        if not available_letters:
            return {}

        for letter in available_letters:
            score = 0
            count = 0
            
            for i in range(len(state) - 2):
                if state[i+1] == '_':
                    left = state[i]
                    right = state[i+2] if i+2 < len(state) else '$'
                    tri_count = self.trigram[left][letter].get(right, 0)
                    tri_total = sum(self.trigram[left][letter].values()) or 1
                    score += (tri_count / tri_total) * 3.0
                    count += 1
            
            for i in range(len(state) - 1):
                if state[i+1] == '_':
                    bi_count = self.bigram[state[i]].get(letter, 0)
                    bi_total = sum(self.bigram[state[i]].values()) or 1
                    score += (bi_count / bi_total) * 2.0
                    count += 1
            
            score += (self.unigram[letter] / (self.total or 1)) * 1.0
            count += 1
            probs[letter] = score / count
        
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        elif available_letters:
            probs = {k: 1.0/len(available_letters) for k in available_letters}
            
        return probs

class FastFrequencyModel:
    # --- FIXED: Was _init_ ---
    def __init__(self):
        self.letter_freq = defaultdict(int)
        self.length_freq = defaultdict(lambda: defaultdict(int))
        self.total = 0
        
    def train(self, words):
        print("Training Frequency Model...")
        for word in tqdm(words):
            word = word.lower()
            for char in word:
                self.letter_freq[char] += 1
                self.length_freq[len(word)][char] += 1
                self.total += 1
    
    def get_letter_probs(self, pattern, word_length, available_letters):
        probs = {}
        vowels = set('aeiou')
        revealed = [c for c in pattern if c != '_']
        revealed_vowels = sum(1 for c in revealed if c in vowels)
        
        if not available_letters:
            return {}

        for letter in available_letters:
            global_freq = self.letter_freq.get(letter, 1) / (self.total or 1)
            length_freq = self.length_freq[word_length].get(letter, 1) / max(sum(self.length_freq[word_length].values()), 1)
            score = global_freq * 0.3 + length_freq * 0.7
            
            if letter in vowels and revealed_vowels < 2:
                score *= 1.5
            elif letter in 'tnrslhd' and len(revealed) < 3:
                score *= 1.3
            
            probs[letter] = score
        
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        elif available_letters:
            probs = {k: 1.0/len(available_letters) for k in available_letters}
            
        return probs

class MicroDQN(nn.Module):
    # --- FIXED: Was _init_ ---
    def __init__(self, state_dim, action_dim):
        super(MicroDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    # --- FIXED: Was _init_ ---
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    # --- FIXED: Was _len_ ---
    def __len__(self):
        return len(self.buffer)

class FastAgent:
    # --- FIXED: Was _init_ ---
    def __init__(self):
        self.pattern = FastPatternMatcher()
        self.ngram = FastNGramModel()
        self.freq = FastFrequencyModel()
        
        self.state_dim = 35
        self.action_dim = 26
        
        self.policy_net = MicroDQN(self.state_dim, self.action_dim).to(device)
        self.target_net = MicroDQN(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001, weight_decay=0.01)
        self.replay_buffer = ReplayBuffer(5000)
        
        self.gamma = 0.90
        self.epsilon = 1.0
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.9995
        self.batch_size = 64
        self.target_update_freq = 500
        self.steps = 0
        
    def train_models(self, corpus_file):
        with open(corpus_file, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(words)} training words")
        
        self.pattern.train(words)
        self.ngram.train(words)
        self.freq.train(words)
        
        return words
    
    def state_to_vector(self, game):
        pattern_str = game.get_state()
        
        features = [
            game.word_length / 20.0,
            game.wrong_guesses / 6.0,
            len(game.guessed_letters) / 26.0,
            pattern_str.count('_') / max(len(pattern_str), 1), # Added max to avoid / 0
            (6 - game.wrong_guesses) / 6.0,
        ]
        
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            features.append(1.0 if letter not in game.guessed_letters else 0.0)
        
        revealed = [c for c in pattern_str if c != '_']
        vowels = set('aeiou')
        features.extend([
            sum(1 for c in revealed if c in vowels) / max(len(revealed), 1),
            sum(1 for c in revealed if c not in vowels) / max(len(revealed), 1),
            len(set(revealed)) / 26.0,
            1.0 if pattern_str and pattern_str[0] != '_' else 0.0, # Added check for empty pattern
        ])
        
        return torch.FloatTensor(features).to(device)
    
    def get_action(self, game, training=True):
        available = game.get_available_letters()
        if not available:
            return None
        
        pattern_str = game.get_state()
        guessed = game.guessed_letters
        blanks_ratio = pattern_str.count('_') / len(pattern_str)
        
        pattern_probs = self.pattern.get_letter_probs(pattern_str, guessed, available)
        ngram_probs = self.ngram.get_letter_probs(pattern_str, available)
        freq_probs = self.freq.get_letter_probs(pattern_str, game.word_length, available)
        
        if training and random.random() < self.epsilon:
            scores = {}
            for letter in available:
                scores[letter] = (0.50 * pattern_probs.get(letter, 0) +
                                0.40 * ngram_probs.get(letter, 0) +
                                0.10 * freq_probs.get(letter, 0))
            
            # Handle case where all scores are 0
            if all(v == 0 for v in scores.values()):
                return random.choice(sorted(available))
                
            probs = np.array([scores[l] for l in sorted(available)])
            probs = probs / (probs.sum() + 1e-10)
            return np.random.choice(sorted(available), p=probs)
        
        if blanks_ratio > 0.6:
            weights = {'pattern': 0.45, 'ngram': 0.40, 'freq': 0.13, 'rl': 0.02}
        elif blanks_ratio > 0.3:
            weights = {'pattern': 0.55, 'ngram': 0.33, 'freq': 0.10, 'rl': 0.02}
        else:
            weights = {'pattern': 0.65, 'ngram': 0.25, 'freq': 0.08, 'rl': 0.02}
        
        state_vector = self.state_to_vector(game)
        with torch.no_grad():
            q_values = self.policy_net(state_vector.unsqueeze(0)).squeeze(0).cpu().numpy()
        
        scores = {}
        q_available_vals = [q_values[ord(l) - ord('a')] for l in available]
        q_min, q_max = min(q_available_vals), max(q_available_vals)
        
        for letter in available:
            idx = ord(letter) - ord('a')
            p_score = pattern_probs.get(letter, 0)
            n_score = ngram_probs.get(letter, 0)
            f_score = freq_probs.get(letter, 0)
            
            if q_max > q_min:
                rl_score = (q_values[idx] - q_min) / (q_max - q_min)
            else:
                rl_score = 0.5
            
            scores[letter] = (weights['pattern'] * p_score +
                            weights['ngram'] * n_score +
                            weights['freq'] * f_score +
                            weights['rl'] * rl_score)
        
        return max(scores, key=scores.get)
    
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < 128:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.LongTensor([ord(a) - ord('a') for a in actions]).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path='agent.pkl'):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path='agent.pkl'):
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        # Set to evaluation mode
        self.policy_net.eval()
        self.target_net.eval()


def train_agent(agent, words, num_episodes=10000):
    print(f"\nTraining RL for {num_episodes} episodes...")
    
    for episode in tqdm(range(num_episodes)):
        word = random.choice(words)
        game = HangmanGame(word)
        state = agent.state_to_vector(game)
        
        while not game.is_done():
            action = agent.get_action(game, training=True)
            if action is None:
                break
            
            correct, done = game.guess(action)
            
            if game.repeated_guesses > 0:
                reward = -8.0
            elif correct:
                reward = 4.0
            else:
                reward = -2.5
            
            if game.is_won():
                reward += 25.0
            elif game.is_lost():
                reward -= 18.0
            
            next_state = agent.state_to_vector(game)
            agent.update(state, action, reward, next_state, game.is_done())
            state = next_state
        
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        if (episode + 1) % 2000 == 0:
            win_rate = evaluate(agent, words, 300)
            print(f"\nEp {episode+1}: WinRate={win_rate:.3f} ε={agent.epsilon:.3f}")
            agent.save('best_agent.pkl')
    
    return agent

def evaluate(agent, words, num_games):
    wins = 0
    for _ in range(num_games):
        word = random.choice(words)
        game = HangmanGame(word)
        while not game.is_done():
            action = agent.get_action(game, training=False)
            if action is None:
                break
            game.guess(action)
        if game.is_won():
            wins += 1
    return wins / num_games

def test(agent, test_file, num_games=2000):
    with open(test_file, 'r') as f:
        words = [line.strip() for line in f if line.strip()]
    
    print(f"\nTesting on {num_games} games...")
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    length_stats = defaultdict(lambda: {'wins': 0, 'games': 0, 'wrong': 0})
    
    for i in tqdm(range(num_games)):
        word = words[i % len(words)]
        game = HangmanGame(word)
        
        while not game.is_done():
            action = agent.get_action(game, training=False)
            if action is None:
                break
            game.guess(action)
        
        length = len(word)
        length_stats[length]['games'] += 1
        length_stats[length]['wrong'] += game.wrong_guesses
        
        if game.is_won():
            wins += 1
            length_stats[length]['wins'] += 1
        
        total_wrong += game.wrong_guesses
        total_repeated += game.repeated_guesses
    
    success_rate = wins / num_games
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Wins: {wins}/{num_games} ({success_rate*100:.2f}%)")
    print(f"Avg Wrong: {total_wrong/num_games:.2f}")
    print(f"Avg Repeated: {total_repeated/num_games:.2f}")
    print("="*70)
    print("By Length:")
    for length in sorted(length_stats.keys()):
        s = length_stats[length]
        wr = s['wins']/s['games']*100 if s['games'] > 0 else 0
        aw = s['wrong']/s['games'] if s['games'] > 0 else 0
        print(f"  {length:2d}: {s['wins']:4d}/{s['games']:4d} ({wr:5.1f}%) - Wrong: {aw:.2f}")
    print("="*70)
    print(f"SCORE: {final_score:.2f}")
    print("="*70)
    
    return final_score

def main():
    print("="*70)
    print("FAST HYBRID AGENT - 50K DATASET")
    print("="*70)
    
    # --- UPDATED: Simplified paths ---
    corpus_path = 'Data/corpus.txt'
    test_corpus_path = 'Data/corpus.txt'
    
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        print("Please create a 'data' folder and place 'corpus.txt' inside it.")
        return

    agent = FastAgent()
    words = agent.train_models(corpus_path)
    
    print(f"\nTraining RL...")
    agent = train_agent(agent, words, num_episodes=10000)
    
    agent.save('final_agent.pkl')
    print("\nSaved final_agent.pkl")
    
    if os.path.exists(test_corpus_path):
        print("\nTesting...")
        test(agent, test_corpus_path, num_games=2000)
    else:
        print(f"\nSkipping test: 'data/test_corpus.txt' not found.")

# --- FIXED: Was _name_ == "_main_" ---
if __name__ == "__main__":
    main()