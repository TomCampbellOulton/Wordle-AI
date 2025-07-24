from random import randint
import time
from functools import wraps
from datetime import datetime

try:
    import nltk
    nltk.download('words')
except:
    pass

from nltk.corpus import words

Version = 1.0
WORD_LENGTH = 5
NUMBER_OF_GUESSES = 26

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"\u23f1\ufe0f Function '{func.__name__}' took {end - start:.4f} seconds")
        return result
    return wrapper

words_l = [x.lower() for x in words.words() if len(x) == WORD_LENGTH and x.isalpha()]

def check_guess(secret_word, guess):
    secret_word = secret_word.lower()
    guess = guess.lower()
    response = ['r'] * WORD_LENGTH
    secret_used = [False] * WORD_LENGTH
    guess_used = [False] * WORD_LENGTH

    for i in range(WORD_LENGTH):
        if guess[i] == secret_word[i]:
            response[i] = 'g'
            secret_used[i] = True
            guess_used[i] = True

    for i in range(WORD_LENGTH):
        if not guess_used[i]:
            for j in range(WORD_LENGTH):
                if not secret_used[j] and guess[i] == secret_word[j]:
                    response[i] = 'o'
                    secret_used[j] = True
                    break

    return ''.join(response)

def process_response(response, guess, unused_letters, used_letters, known):
    for i in range(len(response)):
        res_letter = response[i]
        guess_letter = guess[i]

        if res_letter == 'g':
            known = known[:i] + guess_letter + known[i+1:]
        elif res_letter == 'o':
            if guess_letter in used_letters:
                if i not in used_letters[guess_letter]:
                    used_letters[guess_letter].append(i)
            else:
                used_letters[guess_letter] = [i]
        else:
            if guess_letter not in unused_letters and guess_letter not in known:
                unused_letters.append(guess_letter)
    return unused_letters, used_letters, known

def freqs(words, already_used_letters=[]):
    alphabet_d = {chr(i+97): 0 for i in range(26)}
    for word in words:
        temp = ''.join('_' if l in already_used_letters else l for l in word)
        for letter in temp:
            if letter != '_':
                alphabet_d[letter] += 1
    return sorted(alphabet_d.items(), key=lambda x: x[1], reverse=True)

def score_word(word, freqs):
    return sum(freqs.get(l, 0) for l in set(word))

def find_word_with_most_high_freq_letters(words, required_letters=[], forbidden_letters=[], known_letters="", previous_guesses=[]):
    letter_freq = dict(freqs(words))
    filtered_words = []

    for word in words:
        if word in previous_guesses:
            continue
        valid = True

        for letter in forbidden_letters:
            if letter in word and letter not in known_letters and letter not in [x[0] for x in required_letters]:
                valid = False
                break

        for x in required_letters:
            if isinstance(x, tuple):
                if word[x[1]] != x[0]:
                    valid = False
                    break
            elif isinstance(x, list):
                letter = x[0]
                if letter not in word:
                    valid = False
                    break
                for pos in x[1:]:
                    if word[pos] == letter:
                        valid = False
                        break
        if valid:
            filtered_words.append(word)

    scored_words = sorted(filtered_words, key=lambda w: score_word(w, letter_freq), reverse=True)
    return scored_words

def make_guess(words_l, num_guesses_used, unused_letters, used_letters, known, previous_guesses):
    if num_guesses_used == 0:
        options = find_word_with_most_high_freq_letters(words_l)
        return options[0] if options else "apple"

    required = [(l, i) for i, l in enumerate(known) if l != '_']
    for letter, positions in used_letters.items():
        required.append([letter] + positions)

    options = find_word_with_most_high_freq_letters(words_l, required, unused_letters, known, previous_guesses)
    return options[0] if options else "apple"

def run_game(SECRET_WORD, total, record, better_record, bug_fixing=False):
    num_guesses_used = 0
    SECRET_WORD = SECRET_WORD.lower()

    previous_guesses = []
    unused_letters = []
    used_letters = {}
    known = '_' * len(SECRET_WORD)
    guess = ""

    if bug_fixing:
        print("Secret Word:", SECRET_WORD)

    while num_guesses_used < NUMBER_OF_GUESSES and guess != SECRET_WORD:
        guess = make_guess(words_l, num_guesses_used, unused_letters, used_letters, known, previous_guesses)
        num_guesses_used += 1
        response = check_guess(SECRET_WORD, guess)
        previous_guesses.append(guess)
        unused_letters, used_letters, known = process_response(response, guess, unused_letters, used_letters, known)

        if bug_fixing:
            print(f"Guess {num_guesses_used}: {guess} => {response}")
            print(f"Known: {known}, Used: {used_letters}, Unused: {unused_letters}")

    total += num_guesses_used
    record.append((SECRET_WORD, num_guesses_used))
    better_record.setdefault(num_guesses_used, []).append(SECRET_WORD)
    return total, record, better_record

record = []
better_record = {}
total = 0
counter = 0
bug_fixing = False
possible_words = words_l

for SECRET_WORD in possible_words:
    total, record, better_record = run_game(SECRET_WORD, total, record, better_record, bug_fixing=bug_fixing)
    counter += 1
    if counter % 100 == 0:
        print(f"Progress: {100 * counter / len(words_l):.2f}%")

print(f"Average: {total / counter:.2f}")

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"WordleAI_Attempt_V{Version}_{timestamp}.txt"

with open(filename, "w") as file:
    for guesses, words_list in sorted(better_record.items()):
        file.write(f"\n{guesses} - {words_list}")
    file.write(f"\nAverage - {total / counter:.2f}")

print(f"Saved as {filename}")
