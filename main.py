import argparse


def main(args):
    print(args.sentence)
    print("Test für GitHub!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, default="No sentence!")
    args = parser.parse_args()
    main(args)