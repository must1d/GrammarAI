import argparse
from pathlib import Path

import torch

from utils import one_hot_encoding_size
from networks import LSTM


def main(args):
    # check if Cuda is available for GPU accelerated training
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using Cuda for training!")
    else:
        print("Using CPU for training!")
    device = torch.device("cuda" if use_cuda else "cpu")


def create_train_val_loaders():
    print("Cool")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hidden_size", type=int, default=700)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--val_split", type=float, default=0.1)
    main(args)
