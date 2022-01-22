import csv
import argparse
from pathlib import Path
from random import choice
from typing import List
import json


def main(args):
    sequence_length = args.sequence_length
    sequences: List[str] = []

    # TODO Change sample to some argument
    with open(args.input_path, "r", encoding="utf-8") as infile:
        json_strings = infile.readlines()

        for json_string in json_strings:
            json_obj = json.loads(json_string)
            text = json_obj["text"]

            isInSequence = True
            sequence_counter = 0
            sequences.append("")
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
                # Is is input
                else:
                    sequences[-1] += char
                    sequence_counter += 1

            if sequences[-1] == "":
                sequences.pop(-1)
            if len(sequences[-1]) != args.sequence_length:
                sequences.pop(-1)

    # crate csv output file if it does not exist
    csv_path = args.output_path / "dataset.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["input", "target"])

    # append data to end of csv file
    with open(csv_path, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for sequence in sequences:
            randomized_sequence = ""
            for c in sequence:
                if c == "ß":
                    randomized_sequence += "ß"
                else:
                    randomized_sequence += choice((str.upper, str.lower))(c)

            if len(randomized_sequence) > 50:
                print(ra)
            writer.writerow([randomized_sequence, sequence])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=Path)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--sequence_length", type=int, default=50)
    args = parser.parse_args()
    if args.sequence_length <= 0:
        raise argparse.ArgumentTypeError(
            "Sequence length has to be greater than 0!")
    main(args)
