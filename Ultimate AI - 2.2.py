# 89.96% of games lead to success, no hard coding.

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import os
from nltk.corpus import words

# Timing helper
import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱️ {func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

# Parameters
TRAINING_GUESSES = 35  # More room to learn
MAX_GUESSES = 6        # Evaluation guesses
TRAINING_TIMES = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feedback encoding with stronger weight on green
FEEDBACK_ENCODING = {'R': 0, 'O': 5, 'G': 25}

# Get valid words
def get_words(word_length):
    raw = [w.lower() for w in words.words() if len(w) == word_length and w.isalpha()]
    return list(set(raw))

# Feedback generator
def get_feedback(guess, target):
    feedback = ['R'] * len(guess)
    tgt_chars = list(target)
    for i in range(len(guess)):
        if guess[i] == target[i]:
            feedback[i] = 'G'
            tgt_chars[i] = None
    for i in range(len(guess)):
        if feedback[i] == 'R' and guess[i] in tgt_chars:
            feedback[i] = 'O'
            tgt_chars[tgt_chars.index(guess[i])] = None
    return ''.join(feedback)

# Encode history with padded size
def encode_history(history, word_len, vocab, max_guesses):
    vec = torch.zeros(max_guesses, word_len, len(vocab)+1)
    letter_to_idx = {ch: i for i, ch in enumerate(vocab)}
    for turn, (guess, fb) in enumerate(history):
        for i, (ch, f) in enumerate(zip(guess, fb)):
            vec[turn, i, letter_to_idx[ch]] = 1
            vec[turn, i, -1] = FEEDBACK_ENCODING[f] / 25.0
    return vec.view(1, -1).to(DEVICE)

# Network
class WordleNet(nn.Module):
    def __init__(self, input_dim, hidden=256, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Data generation
def generate_data(word_list, word_len, vocab, max_guesses, n_samples=10000):
    X, y = [], []
    for _ in range(n_samples):
        target = random.choice(word_list)
        history = []
        for _ in range(max_guesses):
            guess = random.choice(word_list)
            fb = get_feedback(guess, target)
            history.append((guess, fb))
            if guess == target:
                break
        X.append(encode_history(history, word_len, vocab, max_guesses).squeeze(0))
        y.append(torch.tensor(word_list.index(target)))
    return torch.stack(X), torch.tensor(y)

# Train
@timeit
def train_model(word_len, previous_model=None):
    word_list = get_words(word_len)
    vocab = sorted(set(''.join(word_list)))
    input_dim = TRAINING_GUESSES * word_len * (len(vocab)+1)
    model = WordleNet(input_dim=input_dim, output_dim=len(word_list)).to(DEVICE)
    model_path = f"model_{word_len}.pth"

    # ✅ Try loading existing model
    if os.path.exists(model_path):
        print(f"📦 Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model, word_list, vocab

    # Optional: transfer weights from previous model
    if previous_model:
        print("Transferring partial weights from earlier model")
        prev_w = previous_model.fc1.weight.data
        prev_b = previous_model.fc1.bias.data
        new_w = model.fc1.weight.data
        new_b = model.fc1.bias.data

        min_r = min(prev_w.shape[0], new_w.shape[0])
        min_c = min(prev_w.shape[1], new_w.shape[1])

        new_w[:min_r, :min_c] = prev_w[:min_r, :min_c]
        new_b[:min_r] = prev_b[:min_r]

    print(f"Training {word_len}-letter model with {len(word_list)} words")
    X, y = generate_data(word_list, word_len, vocab, TRAINING_GUESSES, 10000)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(TRAINING_TIMES):
        optimizer.zero_grad()
        out = model(X.to(DEVICE))
        loss = loss_fn(out, y.to(DEVICE))
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved {model_path}")
    return model, word_list, vocab


# Evaluation
@timeit
def evaluate(model, word_list, vocab, word_len):
    success = 0
    for target in random.sample(word_list, 9972):
        history = []
        possible_words = word_list.copy()
        for _ in range(MAX_GUESSES):
            x = encode_history(history, word_len, vocab, TRAINING_GUESSES)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1).cpu().squeeze()
            scores = {w: probs[word_list.index(w)].item() for w in possible_words}
            guess = max(scores, key=scores.get)
            fb = get_feedback(guess, target)
            history.append((guess, fb))
            if guess == target:
                success += 1
                break
            possible_words = [w for w in possible_words if get_feedback(guess, w) == fb]
    print(f"✅ Solved {success}/100 {word_len}-letter words with {MAX_GUESSES} guesses")

# Run curriculum training
if __name__ == '__main__':
    model3, wl3, v3 = train_model(3)
    model4, wl4, v4 = train_model(4, model3)
    model5, wl5, v5 = train_model(5, model4)
    evaluate(model5, wl5, v5, 5)
