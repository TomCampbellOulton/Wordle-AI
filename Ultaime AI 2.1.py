import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import os
import time
from functools import wraps
from nltk.corpus import words

# --- Timing utility ---
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱️ Function '{func.__name__}' took {time.time() - start:.4f} seconds")
        return result
    return wrapper

# --- Configuration ---
WORD_LENGTH = 5
MAX_GUESSES = 6
TRAINING_TIMES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "wordle_ultimate_ultimate_ai_21.pth"
FIRST_GUESS_PATH = "first_guess_ultimate_21.txt"

# --- Wordlist ---
WORD_LIST = [w.lower() for w in words.words() if len(w) == WORD_LENGTH and w.isalpha()]
WORD_LIST = list(set(WORD_LIST))

ALL_LETTERS = sorted(set("".join(WORD_LIST)))
LETTER_TO_IDX = {ch: idx for idx, ch in enumerate(ALL_LETTERS)}
VOCAB_SIZE = len(ALL_LETTERS)

# --- Feedback: one-hot [R, O, G] = [1, 0, 0], [0, 1, 0], [0, 0, 1] ---
FEEDBACK_DIM = 3  # Red, Orange, Green

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

# --- Encoding ---
def encode_history(history):
    vec = torch.zeros(MAX_GUESSES, WORD_LENGTH, VOCAB_SIZE + FEEDBACK_DIM)
    for turn, (guess, fb) in enumerate(history):
        for i, (ch, f) in enumerate(zip(guess, fb)):
            ch_idx = LETTER_TO_IDX[ch]
            vec[turn, i, ch_idx] = 1
            if f == 'R':
                vec[turn, i, VOCAB_SIZE + 0] = 1
            elif f == 'O':
                vec[turn, i, VOCAB_SIZE + 1] = 5
            elif f == 'G':
                vec[turn, i, VOCAB_SIZE + 2] = 25
    return vec.view(1, -1).to(DEVICE)

# --- Model ---
class WordleNet(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, len(WORD_LIST))

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# --- Entropy ---
def compute_entropy(word, possible_targets):
    outcomes = {}
    for target in possible_targets:
        fb = get_feedback(word, target)
        outcomes[fb] = outcomes.get(fb, 0) + 1
    total = len(possible_targets)
    return -sum((count/total) * math.log2(count/total) for count in outcomes.values())

# --- Data ---
def generate_data(n_samples=5000):
    X, y = [], []
    for _ in range(n_samples):
        target = random.choice(WORD_LIST)
        history = []
        for _ in range(MAX_GUESSES):
            guess = random.choice(WORD_LIST)
            fb = get_feedback(guess, target)
            history.append((guess, fb))
            if guess == target:
                break
        x = encode_history(history)
        X.append(x.squeeze(0))
        y.append(torch.tensor(WORD_LIST.index(target)))
    return torch.stack(X), torch.stack(y)

# --- Training ---
def train_model(training_times=10):
    input_dim = MAX_GUESSES * WORD_LENGTH * (VOCAB_SIZE + FEEDBACK_DIM)
    model = WordleNet(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Generating training data...")
    X, y = generate_data(10000)

    print("Training model...")
    model.train()
    for epoch in range(training_times):
        optimizer.zero_grad()
        outputs = model(X.to(DEVICE))
        loss = criterion(outputs, y.to(DEVICE))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    # Save best first guess
    first_guess = precompute_first_guess(model)
    with open(FIRST_GUESS_PATH, "w") as f:
        f.write(first_guess)
    print(f"📌 First guess '{first_guess}' saved.")
    return model

# --- Load ---
def load_model():
    input_dim = MAX_GUESSES * WORD_LENGTH * (VOCAB_SIZE + FEEDBACK_DIM)
    model = WordleNet(input_dim).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"📦 Model loaded from {MODEL_PATH}")
        return model
    else:
        print("⚠️ No saved model found. Please train the model first.")
        return None

# --- First guess caching ---
def precompute_first_guess(model):
    dummy_history = []
    return ai_guess(model, dummy_history, WORD_LIST)

def load_first_guess():
    if os.path.exists(FIRST_GUESS_PATH):
        with open(FIRST_GUESS_PATH, "r") as f:
            return f.read().strip()
    return None

# --- AI guess ---
@timeit
def ai_guess(model, history, possible_words):
    model.eval()
    with torch.no_grad():
        if len(possible_words) == 1:
            return possible_words[0]

        x = encode_history(history)
        probs = torch.softmax(model(x), dim=1).cpu().squeeze()
        scores = {}

        for idx, word in enumerate(WORD_LIST):
            if word in [h[0] for h in history]:
                continue
            p = probs[idx].item()
            h = compute_entropy(word, possible_words)
            denom = math.log2(len(possible_words)) if len(possible_words) > 1 else 1e-9
            turn = len(history)
            penalty = (turn+2) ** 2
            scores[word] = 0.5 * p + 0.5 * (h / denom) / penalty

        return max(scores, key=scores.get)

# --- Evaluation ---
def evaluate(model):
    first_guess = load_first_guess()
    success_count = 0

    for target in random.sample(WORD_LIST, 100):
        history = []
        possible_words = WORD_LIST.copy()

        for turn in range(MAX_GUESSES):
            if turn == 0 and first_guess:
                guess = first_guess
            else:
                guess = ai_guess(model, history, possible_words)

            fb = get_feedback(guess, target)
            history.append((guess, fb))
            if guess == target:
                success_count += 1
                break
            possible_words = [w for w in possible_words if get_feedback(guess, w) == fb]

    print(f"🎯 AI solved {success_count}/100 words within {MAX_GUESSES} guesses.")

# --- Entry point ---
if __name__ == '__main__':
    if os.path.exists(MODEL_PATH):
        model = load_model()
    else:
        model = train_model(TRAINING_TIMES)
    evaluate(model)
