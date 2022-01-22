from pathlib import Path
from typing import List
import string
import pandas

alphabet = string.printable + "ÄÖÜäöüß"


def read_training_data(path: Path):
    data = pandas.read_csv(path / "dataset.csv")
    inputs = data["input"].to_list()
    targets = data["target"].to_list()
    return inputs, targets


def one_hot(sequence: str) -> List[List[int]]:
    # character index 98 will be used for characters that are not part of alphabet
    sequence_one_hot = []

    for char in sequence:
        arr = [0] * len(alphabet)
        if char not in alphabet:
            arr[98] = 1
        else:
            arr[alphabet.index(char)] = 1
        sequence_one_hot.append(arr)
    return sequence_one_hot
