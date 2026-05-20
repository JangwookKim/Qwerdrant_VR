from collections import defaultdict
import random
import string

codebook = {
    "Q": "qwert",
    "A": "asdfg",
    "Z": "zxcvb",
    "W": "rtyu",
    "X": "vbnm",
    "E": "yuiop",
    "D": "hjkl",
    "C": "nm"
}

ALPHABET = string.ascii_lowercase

gesture_choices = defaultdict(list)

for g, chars in codebook.items():
    for c in chars:
        gesture_choices[c].append(g)


def encode_text(text):

    encoded = []

    for c in text:

        if c in gesture_choices:
            encoded.append(random.choice(gesture_choices[c]))
        else:
            encoded.append(c)

    return "".join(encoded)
    
def build_codebook_vectors(codebook):
    vectors = {}

    for gesture, chars in codebook.items():
        vec = [0.0] * 26
        for c in chars:
            if c in ALPHABET:
                vec[ALPHABET.index(c)] = 1.0
        vectors[gesture] = vec

    return vectors


def get_codebook_vector(token, codebook_vectors):
    if token in codebook_vectors:
        return codebook_vectors[token]

    # space, punctuation, digit, pad 등은 일단 0-vector
    return [0.0] * 26