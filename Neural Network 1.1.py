# Often couldn't get in 6 turns

# Wordle AI with Memory-Efficient Neural Network Training and Evaluation
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import matplotlib.pyplot as plt
from nltk.corpus import words
from nltk import download
from torch.utils.data import Dataset, DataLoader

# Ensure NLTK words corpus is available
download('words')

# Game Settings
WORD_LENGTH = 5
MAX_GUESSES = 6
TRAINING_SAMPLES = 5000
BATCH_SIZE = 64
EPOCHS = 10

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
    for i in range(WORD_LENGTH):
        if guess[i] == secret[i]:
            feedback[i] = 'g'
            secret_used[i] = True
            guess_used[i] = True
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

def is_word_valid(word, history):
    greens = [None] * WORD_LENGTH
    yellows = [set() for _ in range(WORD_LENGTH)]
    absent_letters = set()
    # Analyze feedback history to build constraints
    for guess, feedback in history:
        for i, (g_char, f_char) in enumerate(zip(guess, feedback)):
            if f_char == 'g':
                greens[i] = g_char
            elif f_char == 'o':
                yellows[i].add(g_char)
            elif f_char == 'r':
                # If letter is never green/orange elsewhere in guess, mark as absent
                # Check if letter appears as green/orange elsewhere in the guess
                if g_char not in [guess[j] for j, fb in enumerate(feedback) if fb in ['g', 'o']]:
                    absent_letters.add(g_char)
    # Validate word against constraints
    for i, letter in enumerate(word):
        if greens[i] and letter != greens[i]:
            return False
        if letter in yellows[i]:
            return False
    for letter in absent_letters:
        if letter in word:
            return False
    return True

def encode_history(history):
    enc = []
    for i in range(MAX_GUESSES):
        if i < len(history):
            guess, result = history[i]
            enc.extend(encode_guess_result(guess, result))
        else:
            enc.extend([0] * (WORD_LENGTH * (26 + 3)))
    return enc


# Dataset for training
class WordleDataset(Dataset):
    def __init__(self, word_list, num_samples):
        self.samples = []
        for _ in range(num_samples):
            secret = random.choice(word_list)
            hist = simulate_game(secret)
            x = encode_history(hist)
            y = word_list.index(secret)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# Neural Network
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

# Training Setup
input_dim = WORD_LENGTH * MAX_GUESSES * (26 + 3)
vocab_size = len(word_list)
model = WordleNet(input_dim, vocab_size)
model_file = "wordle_model.pt"

if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file))
    model.eval()
    print("📦 Loaded pre-trained model.")
else:
    print("🧠 No pre-trained model found. Training from scratch...")
    dataset = WordleDataset(word_list, TRAINING_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), model_file)
    print("✅ Model saved as wordle_model.pt")

# Evaluation on all words (streaming)
def simulate_single_word(secret):
    model.eval()
    history = []
    used_guesses = set()
    for attempt in range(1, MAX_GUESSES + 1):
        encoded_input = torch.tensor([encode_history(history)], dtype=torch.float32)
        with torch.no_grad():
            output = model(encoded_input)
            for idx, word in enumerate(word_list):
                if word in used_guesses or not is_word_valid(word, history):
                    output[0][idx] = -float('inf')
            pred_idx = torch.argmax(output, dim=1).item()
            guess = word_list[pred_idx]
            used_guesses.add(guess)
            feedback = get_feedback(secret, guess)
            history.append((guess, feedback))
            if guess == secret:
                return attempt
    return MAX_GUESSES + 1



# Streaming simulation + CSV
print("\n📊 Running full evaluation and writing to CSV...")
results = {}
with open("wordle_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Guesses", "Word"])
    for i, secret in enumerate(word_list):
        guess_count = simulate_single_word(secret)
        writer.writerow([guess_count, secret])
        results[guess_count] = results.get(guess_count, 0) + 1
        if i % 500 == 0:
            print(f"Progress: {i}/{len(word_list)}")

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
