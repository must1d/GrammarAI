import argparse


def main(args):
    print(args.sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, default="blocks")
    args = parser.parse_args()
    main(args)
