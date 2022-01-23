import argparse
import csv
import json
import random
from pathlib import Path
from typing import List


def main(args):
    sequences: List[str] = []

    with open(args.input_path, "r", encoding="utf-8") as infile:
        json_strings = infile.readlines()
        # Read file as lines, one line is one json string
        for json_string in json_strings:
            # Create json object
            json_obj = json.loads(json_string)
            # Read plain text from json object
            text = json_obj["text"]

            inSequence = False
            sequence_counter = 0
            for i, char in enumerate(text):
                # First, check if the sequence is done
                if sequence_counter >= args.sequence_length - 1:
                    # Stop sequence, this char still has the chance to go into a new one
                    inSequence = False
                    sequence_counter = 0
                # Not in a sequence, look for a new start
                if not inSequence and char.isupper():
                    last_two_characters = text[i-2:i]
                    if last_two_characters == ". ":
                        # Start a sequence
                        inSequence = True
                        sequences.append(char)
                # In a sequence, get the next character
                elif inSequence:
                    sequences[-1] += char   # Append character to latest sequence
                    sequence_counter += 1
            # Remove last sequence if it's not long enough
            if len(sequences) > 0 and len(sequences[-1]) < 50:
                sequences.pop(-1)

    # Create csv output file if it does not exist
    csv_path = args.output_path / "dataset.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["input", "target"])

    # Append data to end of csv file
    with open(csv_path, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Prepare actual input for network
        # Problem: lower/upper string may be longer than original character
        for sequence in sequences:
            prepared_sequence = ""
            # Lower all characters
            for char in sequence:
                lowered = char.lower()
                # uppered = char.upper()
                if len(char) < len(lowered):
                    prepared_sequence += char
                else:
                    prepared_sequence += lowered
            # Random capitalization for all characters
            # for char in sequence:
            #     lowered = char.lower()
            #     uppered = char.upper()
            #     if len(char) < len(lowered) or len(char) < len(uppered):
            #         prepared_sequence += char
            #     else:
            #         prepared_sequence += random.choice((lowered, uppered))
            # writer.writerow([prepared_sequence, sequence])


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
