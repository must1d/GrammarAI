from pathlib import Path
from typing import List
from unidecode import unidecode
import string
import pandas


def read_training_data(path: Path):
    data = pandas.read_csv(path / "dataset.csv")
    inputs = data["input"].to_list()
    targets = data["target"].to_list()
    return inputs, targets


def one_hot(sequence: str) -> List[List[int]]:
    alphabet = string.printable
    sequence_one_hot = []
    # TODO New way to encode sequences
    # Can't assure that a character will decode into exactly one character
    decoded = unidecode(sequence)
    #if len(decoded) >= 51:
    #    print(sequence)
    #    print(decoded)
    for char in decoded:
        arr = [0] * len(alphabet)
        arr[alphabet.index(char)] = 1
        sequence_one_hot.append(arr)
    return sequence_one_hot
