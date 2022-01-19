import csv
from typing import List

sequence_length = 5

with open("sample.txt", "r", encoding="utf-8") as infile:
    infile.readline()  # Skip header
    text = infile.read()

    isInSequence = True
    sequence_counter = 0
    sequences: List[str] = [""]
    for i, char in enumerate(text):
        if sequence_counter >= sequence_length:
            isInSequence = False
            sequence_counter = 1
        # Search new upper char
        if not isInSequence:
            if char.isupper():
                last_two_characters = text[i-2:i]
                if last_two_characters == ". ":
                    isInSequence = True
                    sequences.append(char)
        # Is is sequence
        else:
            sequences[-1] += char
            sequence_counter += 1

    label_vecs = []
    for i, sequence in enumerate(sequences):
        label_vecs.append([])
        for char in sequence:
            label_vecs[i].append(1 if char.isupper() else 0)
        sequences[i] = sequence.lower()
    
    print(sequences)
    print(label_vecs)
