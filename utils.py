from pathlib import Path
from typing import List
from unidecode import unidecode
import string
import pandas


def read_training_data(path: Path):
    data = pandas.read_csv(path / "dataset.csv")
    return data["sequence"], data["labels"]


def one_hot(sequence: str) -> List[List[int]]:
    alphabet = string.printable
    sequence_one_hot = []
    decoded = unidecode(sequence)
    for char in decoded:
        arr = [0] * len(alphabet)
        arr[alphabet.index(char)] = 1
        sequence_one_hot.append(arr)
    return sequence_one_hot
