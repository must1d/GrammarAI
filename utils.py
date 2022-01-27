from pathlib import Path
import string
import pandas
import numpy as np

alphabet = string.printable + "ÄÖÜäöü"


def read_training_data(path: Path):
    data = pandas.read_csv(path / "dataset.csv")
    inputs = data["input"].to_list()
    targets = data["target"].to_list()
    return inputs, targets


def one_hot(sequence: str):
    # character index 98 will be used for characters that are not part of alphabet
    sequence_one_hot = np.zeros((len(sequence), len(alphabet)))
    for i, char in enumerate(sequence):
        if char not in alphabet:
            sequence_one_hot[i][98] = 1
        else:
            sequence_one_hot[i][alphabet.index(char)] = 1
    return sequence_one_hot


def print_progress_bar(current, maximum, description: str = '', suffix: str = '', num_bar_chars=30, fill_char='█'):
    progress = current / maximum
    num_blocks = int(progress * num_bar_chars)
    bar = '|{}|'.format(num_blocks * fill_char + (num_bar_chars - num_blocks) * ' ')
    percentage = int(progress * 100)
    print('\r{} {} - {}% {}'.format(description, bar, percentage, suffix), end='')
    if progress >= 1:
        print()
