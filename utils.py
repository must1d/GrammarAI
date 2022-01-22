from pathlib import Path
from typing import List
import string
import pandas

alphabet = string.printable + "ÄÖÜäöü"


def read_training_data(path: Path):
    data = pandas.read_csv(path / "dataset.csv")
    inputs = data["input"].to_list()
    targets = data["target"].to_list()
    return inputs, targets


def one_hot(sequence: str) -> List[List[int]]:
    # character index 98 will be used for characters that are not part of alphabet
    sequence_one_hot = []
    if type(sequence) is float:
        print(sequence)
    for char in sequence:
        arr = [0] * len(alphabet)
        if char not in alphabet:
            arr[98] = 1
        else:
            arr[alphabet.index(char)] = 1
        sequence_one_hot.append(arr)
    return sequence_one_hot


def print_progress_bar(current, maximum, description: str = '', suffix: str = '', num_bar_chars=30, fill_char='█'):
    progress = current / maximum
    num_blocks = int(progress * num_bar_chars)
    bar = '|{}|'.format(num_blocks * fill_char + (num_bar_chars - num_blocks) * ' ')
    percentage = int(progress * 100)
    print('\r{} {} - {}% {}'.format(description, bar, percentage, suffix), end='')
    if progress >= 1:
        print()
