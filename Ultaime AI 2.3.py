# AI Trained with only non "r" lettered words, all words allowed in evaluation

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
TRAINING_GUESSES = 35
MAX_GUESSES = 6
TRAINING_TIMES = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feedback encoding
FEEDBACK_ENCODING = {'R': 0, 'O': 5, 'G': 25}

def get_words(word_length):
    raw = [w.lower() for w in words.words() if len(w) == word_length and w.isalpha()]
    return list(set(raw))

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

def encode_history(history, word_len, vocab, max_guesses):
    vec = torch.zeros(max_guesses, word_len, len(vocab)+1)
    letter_to_idx = {ch: i for i, ch in enumerate(vocab)}
    for turn, (guess, fb) in enumerate(history):
        for i, (ch, f) in enumerate(zip(guess, fb)):
            vec[turn, i, letter_to_idx[ch]] = 1
            vec[turn, i, -1] = FEEDBACK_ENCODING[f] / 25.0
    return vec.view(1, -1).to(DEVICE)

class WordleNet(nn.Module):
    def __init__(self, input_dim, hidden=256, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

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

    # Generate full training dataset once
    X, y = generate_data(word_list, word_len, vocab, TRAINING_GUESSES, 10000)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    total_start_time = time.time()

    for epoch in range(1, TRAINING_TIMES + 1):
        epoch_start = time.time()

        optimizer.zero_grad()
        out = model(X.to(DEVICE))
        loss = loss_fn(out, y.to(DEVICE))
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            duration = time.time() - epoch_start
            pred = torch.argmax(out, dim=1).cpu()
            correct = (pred == y).sum().item()
            accuracy = correct / len(y) * 100
            elapsed = time.time() - total_start_time
            eta = (elapsed / epoch) * (TRAINING_TIMES - epoch)

            print(f"Epoch {epoch:5d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}% "
                  f"| ⏱️ Batch Time: {duration:.2f}s | ETA: {eta/60:.1f} min")

    torch.save(model.state_dict(), f"model_{word_len}.pth")
    print(f"✅ Saved model_{word_len}.pth")
    return model, word_list, vocab


from collections import deque
import csv
import matplotlib.pyplot as plt

@timeit
def evaluate(model, word_list, vocab, word_len):
    success = 0
    guess_counts = {}  # {num_guesses: [word1, word2, ...]}
    total_guesses = 0
    recent_guesses = deque(maxlen=100)
    batch_start_time = time.time()
    failed_words = []

    # For CSV: (word, guesses)
    csv_data = []

    with open(f"guesses_{word_len}letter.txt", "w") as f:
        for idx, target in enumerate(random.sample(word_list, 9972), 1):
            history = []
            possible_words = word_list.copy()
            banned_letters = set()
            solved = False

            for attempt in range(1, MAX_GUESSES + 1):
                filtered_words = [w for w in possible_words if all(ch not in banned_letters for ch in w)]
                if not filtered_words:
                    break

                x = encode_history(history, word_len, vocab, TRAINING_GUESSES)
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1).cpu().squeeze()
                scores = {w: probs[word_list.index(w)].item() for w in filtered_words}
                guess = max(scores, key=scores.get)

                fb = get_feedback(guess, target)
                history.append((guess, fb))

                for i, fb_char in enumerate(fb):
                    if fb_char == 'R':
                        banned_letters.add(guess[i])

                if guess == target:
                    success += 1
                    total_guesses += attempt
                    recent_guesses.append(attempt)
                    guess_counts.setdefault(attempt, []).append(target)
                    csv_data.append((target, attempt))
                    solved = True
                    break

            if not solved:
                guess_counts.setdefault("fail", []).append(target)
                recent_guesses.append(MAX_GUESSES)
                csv_data.append((target, "fail"))

            if idx % 100 == 0:
                batch_time = time.time() - batch_start_time
                avg_overall = total_guesses / success if success > 0 else float('inf')
                avg_recent = sum(recent_guesses) / len(recent_guesses)
                acc = (success / idx) * 100
                print(f"[{idx}/10000] ✅ {success} correct, "
                      f"Accuracy: {acc:.2f}%, "
                      f"Avg guesses: {avg_overall:.2f}, "
                      f"Last 100 avg: {avg_recent:.2f}, "
                      f"Time: {batch_time:.2f}s")
                batch_start_time = time.time()

        # Write guesses text file
        for key in sorted(guess_counts, key=lambda x: (float('inf') if x == "fail" else x)):
            words = ', '.join(guess_counts[key])
            f.write(f"{key}: {words}\n")

    # Save CSV file
    with open(f'guesses_{word_len}letter.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Word', 'Guesses'])
        writer.writerows(csv_data)

    # Plot histogram of guesses (excluding fails)
    guess_numbers = [g for _, g in csv_data if isinstance(g, int)]
    plt.figure(figsize=(8, 5))
    plt.hist(guess_numbers, bins=range(1, MAX_GUESSES+2), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Guesses')
    plt.ylabel('Frequency')
    plt.title(f'Guess Distribution for {word_len}-Letter Words')
    plt.xticks(range(1, MAX_GUESSES+1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'guess_distribution_{word_len}letter.png')
    plt.show()

    # Final stats
    final_avg = total_guesses / success if success > 0 else float('inf')
    accuracy = (success / 9972) * 100
    print(f"\n🎯 Final Accuracy: {success}/10000 ({accuracy:.2f}%)")
    print(f"📊 Avg Guesses (correct only): {final_avg:.2f}")


# Run curriculum training
if __name__ == '__main__':
    model3, wl3, v3 = train_model(3)
    model4, wl4, v4 = train_model(4, model3)
    model5, wl5, v5 = train_model(5, model4)
    evaluate(model5, wl5, v5, 5)
