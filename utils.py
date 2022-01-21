from pathlib import Path
from typing import List
from unidecode import unidecode
import string
import pandas
import numpy as np


def read_training_data(path: Path):
    data = pandas.read_csv(path / "dataset.csv")

    sequences = data["sequence"].to_list()
    labels_str = data["labels"]
    labels = []

    # convert array string to array
    for label in labels_str:
        label = np.fromstring(label[1:-1], sep=',')
        labels.append(list(label))

    return sequences, labels


def one_hot(sequence: str) -> List[List[int]]:
    alphabet = string.printable
    sequence_one_hot = []
    decoded = unidecode(sequence)
    for char in decoded:
        arr = [0] * len(alphabet)
        arr[alphabet.index(char)] = 1
        sequence_one_hot.append(arr)
    return sequence_one_hot
