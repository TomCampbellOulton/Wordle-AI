# MemoryError - uses too much RAM to train


# Wordle AI with Neural Network, Evaluation, CSV Export, and Plotting
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import matplotlib.pyplot as plt
from nltk.corpus import words
from nltk import download
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Ensure NLTK words corpus is available
download('words')

# Game Settings
WORD_LENGTH = 5
MAX_GUESSES = 6
word_list = [w.lower() for w in words.words() if len(w) == WORD_LENGTH and w.isalpha()]
word_list = list(set(word_list))  # Ensure uniqueness

# Encoding
alphabet = list('abcdefghijklmnopqrstuvwxyz')
letter_to_idx = {l: i for i, l in enumerate(alphabet)}
result_to_idx = {'g': 0, 'o': 1, 'r': 2}  # green, orange, red

def encode_guess_result(guess, result):
    encoding = []
    for g_char, r_char in zip(guess, result):
        l_enc = [0] * 26
        r_enc = [0] * 3
        if g_char in letter_to_idx:
            l_enc[letter_to_idx[g_char]] = 1
        if r_char in result_to_idx:
            r_enc[result_to_idx[r_char]] = 1
        encoding.extend(l_enc + r_enc)
    return encoding


def get_feedback(secret, guess):
    feedback = ['r'] * WORD_LENGTH
    secret_used = [False] * WORD_LENGTH
    guess_used = [False] * WORD_LENGTH

    # Green pass
    for i in range(WORD_LENGTH):
        if guess[i] == secret[i]:
            feedback[i] = 'g'
            secret_used[i] = True
            guess_used[i] = True

    # Orange pass
    for i in range(WORD_LENGTH):
        if not guess_used[i]:
            for j in range(WORD_LENGTH):
                if not secret_used[j] and guess[i] == secret[j]:
                    feedback[i] = 'o'
                    secret_used[j] = True
                    break

    return ''.join(feedback)

def simulate_game(secret, max_guesses=MAX_GUESSES):
    history = []
    for _ in range(max_guesses):
        guess = random.choice(word_list)
        result = get_feedback(secret, guess)
        history.append((guess, result))
        if guess == secret:
            break
    return history

def encode_history(history):
    enc = []
    for guess, result in history:
        enc.extend(encode_guess_result(guess, result))
    while len(history) < MAX_GUESSES:
        enc.extend([0] * (WORD_LENGTH * (26 + 3)))
        
    return enc

# Neural Network Definition
class WordleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# Training Preparation
input_dim = WORD_LENGTH * MAX_GUESSES * (26 + 3)
vocab_size = len(word_list)

model = WordleNet(input_dim, vocab_size)
model_file = "wordle_model.pt"

if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file))
    model.eval()
    print("📦 Loaded pre-trained model.")
else:
    print("🧠 No pre-trained model found. Training from scratch.")
    X, y = [], []
    for _ in range(5000):
        secret = random.choice(word_list)
        hist = simulate_game(secret)
        X.append(encode_history(hist))
        y.append(word_list.index(secret))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_file)
    print("✅ Model saved as wordle_model.pt")

# Full Simulation
print("\n📊 Running full evaluation on all words...")

def simulate_model_on_all_words(model, word_list, max_guesses=MAX_GUESSES):
    model.eval()
    guess_counts = defaultdict(int)

    for i, secret in enumerate(word_list):
        history = []
        used_guesses = set()
        for attempt in range(1, max_guesses + 1):
            encoded_input = torch.tensor([encode_history(history)], dtype=torch.float32)
            with torch.no_grad():
                output = model(encoded_input)
                for idx, word in enumerate(word_list):
                    if word in used_guesses:
                        output[0][idx] = -float('inf')
                pred_idx = torch.argmax(output, dim=1).item()
                guess = word_list[pred_idx]
                used_guesses.add(guess)
                feedback = get_feedback(secret, guess)
                history.append((guess, feedback))
                if guess == secret:
                    guess_counts[attempt] += 1
                    break
        else:
            guess_counts[max_guesses + 1] += 1

        if i % 500 == 0:
            print(f"Progress: {i}/{len(word_list)}")

    return guess_counts

results = simulate_model_on_all_words(model, word_list)

# Save to CSV
with open("wordle_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Guesses", "Words"])
    for guess_count in sorted(results):
        writer.writerow([guess_count, results[guess_count]])
print("📄 Results saved to wordle_results.csv")

# Plotting
labels = sorted(results)
counts = [results[k] for k in labels]
plt.figure(figsize=(10, 5))
plt.bar([str(l) for l in labels], counts, color='skyblue')
plt.xlabel("Number of Guesses")
plt.ylabel("Number of Words Solved")
plt.title("Wordle AI Guess Distribution")
plt.tight_layout()
plt.savefig("wordle_distribution.png")
plt.show()
print("📊 Plot saved as wordle_distribution.png")
