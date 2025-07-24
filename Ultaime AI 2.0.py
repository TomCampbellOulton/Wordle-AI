# Next-Guess Wordle AI Trainer (Heuristic-Augmented, Fast, Flexible)

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import os
from nltk.corpus import words
from collections import Counter

# Configuration
WORD_LENGTH = 5
MAX_GUESSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "wordle_next_guess_ai.pth"
SAVE_FIRST_GUESS = "first_guess.txt"

# Load valid word list
WORD_LIST = [w.lower() for w in words.words() if len(w) == WORD_LENGTH and w.isalpha()]
WORD_LIST = list(set(WORD_LIST))[:5000]  # Keep it manageable
LETTER_TO_IDX = {ch: idx for idx, ch in enumerate(sorted(set("".join(WORD_LIST))))}
VOCAB_SIZE = len(LETTER_TO_IDX)
FEEDBACK_ENCODING = {'R': 0, 'O': 1, 'G': 2}

# Get feedback like Wordle
def get_feedback(guess, target):
    feedback = ['R'] * WORD_LENGTH
    target_chars = list(target)
    for i in range(WORD_LENGTH):
        if guess[i] == target[i]:
            feedback[i] = 'G'
            target_chars[i] = None
    for i in range(WORD_LENGTH):
        if feedback[i] == 'R' and guess[i] in target_chars:
            feedback[i] = 'O'
            target_chars[target_chars.index(guess[i])] = None
    return "".join(feedback)

# Encode a guess + feedback into a fixed vector
def encode_history(history):
    vec = torch.zeros(MAX_GUESSES, WORD_LENGTH, VOCAB_SIZE + 1)
    for turn, (guess, fb) in enumerate(history):
        for i, (ch, f) in enumerate(zip(guess, fb)):
            vec[turn, i, LETTER_TO_IDX[ch]] = 1
            vec[turn, i, -1] = FEEDBACK_ENCODING[f] / 2.0
    return vec.view(1, -1).to(DEVICE)

# Simple frequency-based first guess
letter_freq = Counter("".join(WORD_LIST))
def best_first_guess():
    scored = [(word, sum(letter_freq[c] for c in set(word))) for word in WORD_LIST]
    return max(scored, key=lambda x: x[1])[0]

# Neural net
class WordleNextGuessNet(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, len(WORD_LIST))

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Training data
def generate_next_guess_data(n_samples=10000):
    X, y = [], []
    for _ in range(n_samples):
        target = random.choice(WORD_LIST)
        history = []
        possible_words = WORD_LIST.copy()
        for _ in range(MAX_GUESSES):
            guess = best_first_guess() if not history else random.choice(possible_words)
            fb = get_feedback(guess, target)
            x = encode_history(history)
            X.append(x.squeeze(0))
            y.append(torch.tensor(WORD_LIST.index(guess)))
            history.append((guess, fb))
            if guess == target:
                break
            possible_words = [w for w in possible_words if get_feedback(guess, w) == fb]
    return torch.stack(X), torch.stack(y)

# Training function
def train_model(epochs=10):
    input_dim = MAX_GUESSES * WORD_LENGTH * (VOCAB_SIZE + 1)
    model = WordleNextGuessNet(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Generating training data...")
    X, y = generate_next_guess_data()

    print("Training model...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X.to(DEVICE))
        loss = criterion(out, y.to(DEVICE))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    with open(SAVE_FIRST_GUESS, 'w') as f:
        f.write(best_first_guess())
    return model

# Load model + first guess
def load_model():
    input_dim = MAX_GUESSES * WORD_LENGTH * (VOCAB_SIZE + 1)
    model = WordleNextGuessNet(input_dim).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    return model

def load_first_guess():
    if os.path.exists(SAVE_FIRST_GUESS):
        with open(SAVE_FIRST_GUESS) as f:
            return f.read().strip()
    return best_first_guess()

# Guess function
def ai_guess(model, history):
    model.eval()
    with torch.no_grad():
        x = encode_history(history)
        scores = model(x).cpu().squeeze()
        for guess_idx in torch.argsort(scores, descending=True):
            guess = WORD_LIST[guess_idx]
            if guess not in [h[0] for h in history]:
                return guess

# Evaluate AI performance
def evaluate(model, num_games=100):
    first_guess = load_first_guess()
    success = 0
    for target in random.sample(WORD_LIST, num_games):
        history = []
        possible_words = WORD_LIST.copy()
        for _ in range(MAX_GUESSES):
            guess = first_guess if not history else ai_guess(model, history)
            fb = get_feedback(guess, target)
            history.append((guess, fb))
            if guess == target:
                success += 1
                break
            possible_words = [w for w in possible_words if get_feedback(guess, w) == fb]
    print(f"✅ Solved {success}/{num_games} games")

# Entry point
if __name__ == '__main__':
    if os.path.exists(MODEL_PATH):
        model = load_model()
    else:
        model = train_model(epochs=10000)
    evaluate(model)
