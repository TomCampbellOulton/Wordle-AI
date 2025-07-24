import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import os
import time
from nltk.corpus import words
import matplotlib.pyplot as plt

# Config
WORD_LEN = 5
TRAINING_GUESSES = 35
MAX_GUESSES = 6
TRAINING_TIMES = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = f"model_{WORD_LEN}.pth"
DISTRIBUTION_LOG = "guess_distribution.txt"

FEEDBACK_ENCODING = {'R': 0, 'O': 5, 'G': 25}

# Helper: Timer decorator
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱️ {func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

# Get word list
def get_words(length):
    return list(set(w.lower() for w in words.words() if len(w) == length and w.isalpha()))

# Feedback logic
def get_feedback(guess, target):
    fb = ['R'] * len(guess)
    tgt_chars = list(target)
    for i in range(len(guess)):
        if guess[i] == target[i]:
            fb[i] = 'G'
            tgt_chars[i] = None
    for i in range(len(guess)):
        if fb[i] == 'R' and guess[i] in tgt_chars:
            fb[i] = 'O'
            tgt_chars[tgt_chars.index(guess[i])] = None
    return ''.join(fb)

# Encode history
def encode_history(history, word_len, vocab, max_guesses):
    vec = torch.zeros(max_guesses, word_len, len(vocab)+1)
    letter_to_idx = {ch: i for i, ch in enumerate(vocab)}
    for turn, (guess, fb) in enumerate(history):
        for i, (ch, f) in enumerate(zip(guess, fb)):
            vec[turn, i, letter_to_idx[ch]] = 1
            vec[turn, i, -1] = FEEDBACK_ENCODING[f] / 25.0
    return vec.view(1, -1).to(DEVICE)

# Model
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

# Load or train
@timeit
def load_or_train_model(word_len):
    word_list = get_words(word_len)
    vocab = sorted(set(''.join(word_list)))
    input_dim = TRAINING_GUESSES * word_len * (len(vocab)+1)

    model = WordleNet(input_dim=input_dim, output_dim=len(word_list)).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"📦 Loaded existing model from {MODEL_PATH}")
    else:
        print(f"📚 Training new model ({len(word_list)} words)...")
        X, y = generate_data(word_list, word_len, vocab, TRAINING_GUESSES, 10000)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(TRAINING_TIMES):
            optimizer.zero_grad()
            out = model(X.to(DEVICE))
            loss = loss_fn(out, y.to(DEVICE))
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ Saved model to {MODEL_PATH}")
    return model, word_list, vocab

# Evaluation
@timeit
def evaluate(model, word_list, vocab, word_len):
    guess_stats = {i: [] for i in range(1, MAX_GUESSES+1)}
    avg_guesses_per_100 = []
    time_per_100 = []
    total_guesses = 0
    correct = 0
    last_100_guesses = []
    with open(DISTRIBUTION_LOG, 'w') as f:
        for batch in range(100):
            start = time.time()
            batch_guesses = 0
            for i in range(100):
                target = random.choice(word_list)
                history = []
                banned_letters = set()
                possible_words = word_list.copy()

                for turn in range(MAX_GUESSES):
                    x = encode_history(history, word_len, vocab, TRAINING_GUESSES)
                    with torch.no_grad():
                        probs = torch.softmax(model(x), dim=1).cpu().squeeze()
                    valid_words = [w for w in possible_words if not any(b in w for b in banned_letters)]
                    if not valid_words:
                        break
                    scores = {w: probs[word_list.index(w)].item() for w in valid_words}
                    guess = max(scores, key=scores.get)
                    fb = get_feedback(guess, target)
                    history.append((guess, fb))
                    banned_letters.update({c for c, f in zip(guess, fb) if f == 'R'})
                    print(guess, target)
                    if guess == target:
                        guess_stats[turn+1].append(target)
                        batch_guesses += (turn + 1)
                        total_guesses += (turn + 1)
                        correct += 1
                        last_100_guesses.append(turn + 1)
                        if len(last_100_guesses) > 100:
                            last_100_guesses.pop(0)
                        break
            duration = time.time() - start
            avg = batch_guesses / 100 if batch_guesses else 0
            last_100_avg = sum(last_100_guesses) / len(last_100_guesses) if last_100_guesses else 0
            avg_guesses_per_100.append(avg)
            time_per_100.append(duration)
            print(f"[Batch {batch+1}] Avg guesses: {avg:.2f}, Time: {duration:.2f}s, Accuracy: {correct / ((batch+1)*100):.2%}, Last 100 Avg: {last_100_avg:.2f}")

        # Write to file
        for g in range(1, MAX_GUESSES+1):
            words_at_g = ", ".join(guess_stats[g])
            f.write(f"{g}: {words_at_g}\n")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(avg_guesses_per_100, label="Avg Guesses")
    plt.xlabel("Batch (100 words)")
    plt.ylabel("Average Guesses")
    plt.title("Avg Guesses Over Time")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(time_per_100, label="Time per 100", color="orange")
    plt.xlabel("Batch (100 words)")
    plt.ylabel("Time (s)")
    plt.title("Evaluation Time per 100")
    plt.legend()

    plt.tight_layout()
    plt.savefig("evaluation_stats.png")
    print("📊 Saved graph to evaluation_stats.png")

# Main
if __name__ == '__main__':
    model, word_list, vocab = load_or_train_model(WORD_LEN)
    evaluate(model, word_list, vocab, WORD_LEN)
