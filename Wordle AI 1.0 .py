from random import randint

try:
    # Download all english words
    import nltk
    nltk.download('words')
except:
    pass

# Import the words downloaded
from nltk.corpus import words


# Time It
import time
from functools import wraps

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"⏱️ Function '{func.__name__}' took {end - start:.4f} seconds")
        return result
    return wrapper



Version = 1.0
WORD_LENGTH = 5
NUMBER_OF_GUESSES = 26
num_guesses_used = 0
# Filter to only allow words of length 5
# Golfed Version
words_l = [x for x in words.words() if len(x)==WORD_LENGTH]
# Ungolfed Version
words_l_2 = []
for x in words.words():
    if len(x) == WORD_LENGTH:
        words_l_2.append(x)


# Wordle Game
def check_guess(secret_word, guess):
    checks_complete = False

    while (not checks_complete):
        # Check if guess is correct
        if guess == secret_word:
            checks_complete = True
            return "Correct!"

        # Default response = _____, meaning nothing right in your guess
        # g____ meaning your guess for first letter is correct
        # o____ meaning your guess for first letter is right letter, wrong place
        response = "_" * WORD_LENGTH
        # Check how many letters are correct
        for x in range(WORD_LENGTH):
            if guess[x] == secret_word[x].lower():
                response = response[:x] + 'g' + response[x+1:]
                secret_word = secret_word[:x] + '_' + secret_word[x+1:]

        for x in range(WORD_LENGTH):
            char_checked = False
            # ie if not correct position
            if response[x] == '_':
                for y in range(WORD_LENGTH):
                    if guess[x] == secret_word[y] and not char_checked:
                        response = response[:x] + 'o' + response[x+1:]
                        secret_word = secret_word[:y] + '_' + secret_word[y+1:]
                        char_checked = True
        return response.replace('_', 'r')

# Sort out the new information
def process_response(response, guess, unused_letters, used_letters, known):
    for x in range(WORD_LENGTH):
        response_letter = response[x]
        guess_letter = guess[x]
        
        if response_letter == 'g':
            known = known[:x] + guess_letter + known[x+1:]
            
            
        elif response_letter == 'o':
            # If this letter has been guessed already
            if (guess_letter in used_letters.keys()):
                used_letters[guess_letter] += [0-x-1]
            else:
                used_letters[guess_letter] = [0-x-1]
                
        
        else: # x == 'r'
            # Check if already present
            if guess_letter not in unused_letters:
                unused_letters.append(guess_letter)
    return (unused_letters, used_letters, known)

# Find Frequency of Each Letter
def freqs(words, already_used_letters = []):
    # list of alphabet
    alphabet = [chr(i) for i in range(ord('a'),ord('z')+1)]
    # dictionary of alphabet, each letter as key, value as 0
    alphabet_d = {chr(i+96):0 for i in range(1,27)}

    # Check if there are any letters already used
    if len(already_used_letters) == 0:
        for word in words:
            for letter in word:
                alphabet_d[letter.lower()] += 1
        # Returns dictionary containing frequency of each letter
        return sorted(alphabet_d.items(), key=lambda x:x[1], reverse=True)
    else:
        for word in words:
            # Remove the already used letters
            for l in already_used_letters:
                for n in range(len(word)):
                    if l == word[n]:
                        word = word[:n] + '_' + word[n+1:]
                
            for letter in word:
                if letter != '_':
                    alphabet_d[letter.lower()] += 1
        # Returns dictionary containing frequency of each letter

            
        # Order the letters from highgest to lowest frequency
        ordered_frequencies = sorted(alphabet_d.items(), key=lambda x:x[1], reverse=True) # frequencies.items returns [(a,17), (b,7), (c, 3), etc]
        # lambda x:x[1] being a lambda function to take input x, then return value x[1]
        # so key=lambda x:x[1] returns index 1 of the input, being 17, then 7, then 3, etc
        # reverse=True to make it sort in descending order, with highest frequency first
        
        return ordered_frequencies

# NOT PERMANENT!!! A solution to make it 'predict' good starters
def find_word_with_most_high_freq_letters(words, WORD_LENGTH, required_letters=[], forbidden_letters=[], known_letters = "", previous_guesses=[]):
    # Try for max number of high frequency letters, step down until one is found

    # First, get frequencies
    frequencies = freqs(words)


    # We know the word must contain at least 1 letter, so only look at what has most frequently used letter
    filtered_words = []
    for word in words:
        if word not in previous_guesses:
            word = word.lower()
            # Check it meets the requirements
            valid_word = True
            # Check for forbidden letters first
            if len(forbidden_letters) > 0: # Little complex as can have multiple of a letter
                for letter in forbidden_letters:
                    if letter in word:
                        # Check there is not ANY of this letter allowed
                        num_allowed = 0
                        for l in required_letters:
                            if l[0] == letter:
                                num_allowed += 1
                        for l in known_letters:
                            if l[0] == letter:
                                num_allowed += 1
                        if word.count(letter) > num_allowed:                    
                            valid_word = False
                            
            # Check if it has the required letters, if there are any
            if len(required_letters) > 0 and valid_word:
                # e.g. required_letters = [('a', 0), ('b', -1)]
                # a index 0, b not index 2

                # First check if any letters are already correct placement
                for x in required_letters:
                    # If it's green - correct letter, correct index
                    if x[1] >= 0:
                        # Check if that index in the word is the same letter
                        if word[x[1]] != x[0]:
                            # If it's not, the word can't be used
                            valid_word = False
                    else: # We know the letter is present, and not in particular indexes (orange)
                        letter_found = False
                        for n in range(len(word)):
                            # If the letter is found at all
                            if word[n] == x[0]:
                                letter_found = True
                                
                            # If it's in a forbidden index, invalid word
                            if (0-n-1) in x and word[n] == x[0]:
                                valid_word = False
                        # If the letter is not ever found it's invalid
                        if (not letter_found):
                            valid_word = False

            if valid_word:
                filtered_words.append(word)
    
    used_letters = [frequencies[0][0]]
    counter = 0 # Ensures that there isn't an infinite loop, ie if the remaining words are all consisting of same letters
    
    more_filtered_words = filtered_words
    while len(more_filtered_words) > 0 and counter < WORD_LENGTH:
        counter += 1
        
        filtered_words = more_filtered_words
        
        # Now filter again
        new_freq = freqs(filtered_words, used_letters)
        used_letters.append(new_freq[0][0])
        
        more_filtered_words = []
        for word in filtered_words:
            if new_freq[0][0] in word: # if the most frequent letter in words
                more_filtered_words.append(word)
    # Best words
    return filtered_words

# TWO Choices. 1. Require correctly guessed letters to be used. 2. Don't
# Assume 1 for now

# The AI part
def make_guess(words_l, num_guesses_used, unused_letters, used_letters, known, previous_guesses):
    if num_guesses_used == 0:
        first_guess_options = find_word_with_most_high_freq_letters(words_l, WORD_LENGTH)
        my_guess = first_guess_options[0]
        return my_guess
    
    # Second guess, use none of the letters from the first word to help eliminate options
    elif num_guesses_used == 1:
        banned_letters = find_word_with_most_high_freq_letters(words_l, WORD_LENGTH)[0]
        second_guess_options = find_word_with_most_high_freq_letters(words_l, WORD_LENGTH, forbidden_letters=banned_letters, known_letters = known, previous_guesses=previous_guesses)
        my_guess = second_guess_options[0]
        return my_guess
        
    # All subsequent guesses, use the knowledge we now have
    else:
        req_lets = []
        counter = 0
        # Record the indexes of the known letters
        for letter in known:
            if letter != "_":
                req_lets.append((letter, counter))
            counter += 1
        # Record forbidden indexes of known letters
        forbidden_letters = []
        for letter in list(used_letters.keys()):
            # Iterate through the indexes
            forbidden_indexes = [letter]
            for x in used_letters[letter]:
                forbidden_indexes.append(x)
            req_lets.append(forbidden_indexes)
            
        
        guess_options = find_word_with_most_high_freq_letters(words_l, WORD_LENGTH, req_lets, unused_letters, known_letters = known, previous_guesses=previous_guesses)
        my_guess = guess_options[0]
        return my_guess

def run_game(SECRET_WORD, total, record, better_record, bug_fixing=False):
    # Game Setup
    num_guesses_used = 0
    SECRET_WORD = SECRET_WORD.lower()

    previous_guesses = []
    unused_letters = []
    used_letters = {} # key = letter, value = [indexes we know are NOT correct]
    known = "_" * WORD_LENGTH
    guess = ""

    if bug_fixing:
        print(SECRET_WORD)
    
    while (num_guesses_used < NUMBER_OF_GUESSES and guess != SECRET_WORD):
        
        # Play with info
        guess = make_guess(words_l, num_guesses_used, unused_letters, used_letters, known, previous_guesses)
        num_guesses_used += 1
        response = check_guess(SECRET_WORD, guess)
        previous_guesses.append(guess)

        unused_letters, used_letters, known = process_response(response, guess, unused_letters, used_letters, known)

        if bug_fixing:
            print(guess)
            print(unused_letters, used_letters, known)
        # AI part probssss being in making good first guess and good subsequent ones

    total += num_guesses_used
    record.append((SECRET_WORD, num_guesses_used))

    # Check if that number of guesses has been used before
    if num_guesses_used in better_record:
        better_record[num_guesses_used].append(SECRET_WORD)
    else:
        better_record[num_guesses_used] = [SECRET_WORD]
    return total, record, better_record


record = []
better_record = {}
total = 0
print(len(words_l))
counter = 0
possible_words = words_l
bug_fixing = False


# For Bug Fixing
#bug_fixing = True
#possible_words = ["asana", "arrie"]


# For Regular Play
for SECRET_WORD in possible_words:
    total, record, better_record = run_game(SECRET_WORD, total, record, better_record, bug_fixing=bug_fixing)
    
    counter += 1
    if counter % 100 == 0:
        print(f"Progress = {100 * counter / len(words_l)}")
        
print(record)
print(f"Average = {total/counter}")



# Save the record - ChatGPT wrote the following save code
from datetime import datetime
now = datetime.now()
# Format the datetime string (e.g., 2025-07-19_15-30-45)
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
# Create filename
filename = f"WordleAI_Attempt_V{Version}_{timestamp}.txt"

# Save file
with open(filename, "w") as file:
    for num_guesses in better_record:
        file.write(f"\n{num_guesses} - {better_record[num_guesses]}")
    file.write(f"\nAverage - {total/counter}")

print(f"Saved as {filename}")

# ChatGPT ends

# Error Crashes
#Mosur
#Chara
# Just Doesn't Guess
# Skyre
# swith
# 





