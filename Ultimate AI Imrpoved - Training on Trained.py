import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import os
import time
from functools import wraps
from nltk.corpus import words
from collections import Counter
import matplotlib.pyplot as plt


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱️ {func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

TRAINING_GUESSES = 75
MAX_GUESSES = 6
TRAINING_TIMES = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def generate_data(word_list, word_len, vocab, max_guesses, n_samples=10):
    X, y, weights = [], [], []
    guess_counts = []

    for target in word_list:
        for i in range(int(max(n_samples / len(word_list), 1))):  # repeat k times
            history = []
            guess_count = 0

            for _ in range(max_guesses):
                guess = random.choice(word_list)
                fb = get_feedback(guess, target)
                history.append((guess, fb))
                guess_count += 1
                if guess == target:
                    break

            guess_counts.append(guess_count)
            if (i + 1) % 500 == 0:
                avg = sum(guess_counts[-500:]) / 500
                print(f"📊 Avg guesses (last 500 samples): {avg:.2f}")

            X.append(encode_history(history, word_len, vocab, max_guesses).squeeze(0))
            y.append(word_list.index(target))
            weights.append(1.0 / (guess_count ** 2))

    return torch.stack(X), torch.tensor(y), torch.tensor(weights)


@timeit
def train_model(word_len, previous_model=None):
    word_list = get_words(word_len)
    vocab = sorted(set(''.join(word_list)))
    input_dim = TRAINING_GUESSES * word_len * (len(vocab)+1)
    model = WordleNet(input_dim=input_dim, output_dim=len(word_list)).to(DEVICE)
    model_path = f"model_{word_len}.pth"

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
    X, y, weights = generate_data(word_list, word_len, vocab, TRAINING_GUESSES, 10)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(TRAINING_TIMES):
        optimizer.zero_grad()
        out = model(X.to(DEVICE))
        losses = loss_fn(out, y.to(DEVICE))
        loss = (losses * weights.to(DEVICE)).mean()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved {model_path}")
    return model, word_list, vocab
@timeit
def evaluate(model, word_list, vocab, word_len):
    import matplotlib.pyplot as plt
    from datetime import timedelta

    success = 0
    guess_counts = []
    chunk_accuracy = []
    chunk_guesses = []
    history_graph = []
    chunk_size = 100

    total_words = min(10000, len(word_list))
    targets = random.sample(word_list, total_words)

    start_time = time.time()

    for i, target in enumerate(targets):
        history = []
        possible_words = word_list.copy()

        for guess_num in range(1, MAX_GUESSES + 1):
            x = encode_history(history, word_len, vocab, TRAINING_GUESSES)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1).cpu().squeeze()

            scores = {w: probs[word_list.index(w)].item() for w in possible_words}
            if not scores:
                guess = random.choice(possible_words)
            else:
                guess = max(scores, key=scores.get)

            fb = get_feedback(guess, target)
            history.append((guess, fb))

            if guess == target:
                success += 1
                guess_counts.append(guess_num)
                break

            possible_words = [w for w in possible_words if get_feedback(guess, w) == fb]
        else:
            guess_counts.append(MAX_GUESSES + 1)  # failed to solve

        # Every chunk_size evaluations, compute stats and ETA
        if (i + 1) % chunk_size == 0:
            chunk = guess_counts[-chunk_size:]
            accuracy = sum(1 for g in chunk if g <= MAX_GUESSES) / chunk_size
            avg_guesses = sum(chunk) / chunk_size
            chunk_accuracy.append(accuracy * 100)
            chunk_guesses.append(avg_guesses)
            history_graph.append(i + 1)

            elapsed = time.time() - start_time
            chunks_done = (i + 1) // chunk_size
            chunks_total = total_words // chunk_size
            avg_time_per_chunk = elapsed / chunks_done
            eta_seconds = (chunks_total - chunks_done) * avg_time_per_chunk
            eta_str = str(timedelta(seconds=int(eta_seconds)))

            print(f"[{i+1}/{total_words}] ✅ Acc: {accuracy*100:.1f}% | Avg Guesses: {avg_guesses:.2f} | ETA: {eta_str}")

    # Final stats
    print(f"\n🎯 Final Accuracy: {100 * success / total_words:.2f}%")
    print(f"📊 Avg guesses overall: {sum(guess_counts)/len(guess_counts):.2f}")

    # Plot graphs
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_graph, chunk_accuracy, marker='o')
    plt.title("Accuracy (%) over Time")
    plt.xlabel("Words Evaluated")
    plt.ylabel("Accuracy (%)")

    plt.subplot(1, 2, 2)
    plt.plot(history_graph, chunk_guesses, marker='x', color='green')
    plt.title("Average Number of Guesses")
    plt.xlabel("Words Evaluated")
    plt.ylabel("Guesses")

    plt.tight_layout()
    plt.show()

@timeit
def test_ai(model, word_list, vocab, word_len, save_path="test_results.txt"):
    print(f"🧪 Testing AI on {len(word_list)} words (no guess limit)...")
    start_time = time.time()

    guesses_per_word = []
    success = 0
    total = 0
    log_lines = []

    for i, target in enumerate(word_list, 1):
        history = []
        possible_words = word_list.copy()
        guess_count = 0

        while True:
            x = encode_history(history, word_len, vocab, TRAINING_GUESSES)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1).cpu().squeeze()

            scores = {
                w: probs[word_list.index(w)].item()
                for w in possible_words
            }

            if not scores:
                log_lines.append(f"{target}: wrong (ran out of options)\n")
                break

            guess = max(scores, key=scores.get)
            fb = get_feedback(guess, target)
            history.append((guess, fb))
            guess_count += 1

            if guess == target:
                success += 1
                break

            possible_words = [w for w in possible_words if get_feedback(guess, w) == fb]

        guesses_per_word.append(guess_count)
        total += 1
        log_lines.append(f"{target}: right in {guess_count} guesses\n")

        # ETA display
        if i % 50 == 0:
            print(log_lines)
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (len(word_list) - i)
            avg_guesses = sum(guesses_per_word) / i
            print(f"[{i}/{len(word_list)}] Avg guesses: {avg_guesses:.2f} | ETA: {remaining:.1f}s")

    accuracy = (success / total) * 100
    avg_guesses = sum(guesses_per_word) / total

    print(f"\n✅ Test complete!")
    print(f"🎯 Final Accuracy: {accuracy:.2f}%")
    print(f"📊 Avg guesses overall: {avg_guesses:.2f}")
    
    # Save to file
    with open(save_path, "w") as f:
        f.write(f"Final Accuracy: {accuracy:.2f}%\n")
        f.write(f"Avg guesses overall: {avg_guesses:.2f}\n\n")
        f.writelines(log_lines)

    print(f"📄 Results saved to '{save_path}'")

    # Graph
    counts = Counter(guesses_per_word)
    x = sorted(counts.keys())
    y = [counts[k] for k in x]

    plt.figure(figsize=(10, 5))
    plt.bar(x, y, color="mediumseagreen", edgecolor="black")
    plt.title("Guess Distribution per Word")
    plt.xlabel("Number of Guesses")
    plt.ylabel("Number of Words")
    plt.xticks(x)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



# Run curriculum training
if __name__ == '__main__':
    model3, wl3, v3 = train_model(3)
    model4, wl4, v4 = train_model(4, model3)
    model5, wl5, v5 = train_model(5, model4)
    #evaluate(model5, wl5, v5, 5)
    test_ai(model5, wl5, v5, 5)
