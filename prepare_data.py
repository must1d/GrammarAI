import csv
import argparse
from pathlib import Path
from typing import List


def main(args):
    sequence_length = args.sequence_length
    sequences: List[str] = [""]
    label_vecs = []

    with open(args.input_path / "sample.txt", "r", encoding="utf-8") as infile:
        infile.readline()  # Skip header
        text = infile.read()

        isInSequence = True
        sequence_counter = 0
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

        for i, sequence in enumerate(sequences):
            label_vecs.append([])
            for char in sequence:
                label_vecs[i].append(1 if char.isupper() else 0)
            sequences[i] = sequence.lower()

    # crate csv output file if it does not exist
    csv_path = args.output_path / "dataset.csv"
    if not csv_path.exists():
        with open(csv_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["sequence", "labels"])

    # append data to end of csv file
    with open(csv_path, "a") as csvfile:
        writer = csv.writer(csvfile)
        for sequence, label in zip(sequences, label_vecs):
            writer.writerow([sequence, label])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=Path)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("sequence_length", type=int, default=100, nargs="?")
    args = parser.parse_args()
    if args.sequence_length <= 0:
        raise argparse.ArgumentTypeError(
            "Sequence length has to be greater than 0!")
    main(args)
