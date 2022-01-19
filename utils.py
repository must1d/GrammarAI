from typing import List
from unidecode import unidecode
import string


def one_hot(sequence: str) -> List[List[int]]:
    # TODO Fix this structure
    alphabet = string.printable

    sequence_one_hot = []
    decoded = unidecode(sequence)
    for char in decoded:
        arr = [0] * len(alphabet)
        arr[alphabet.index(char)] = 1
        sequence_one_hot.append(arr)
    return sequence_one_hot