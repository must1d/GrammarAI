import argparse
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import string
import random

from utils import read_training_data, one_hot
from networks import LSTM


def main(args):
    # first check if cuda is available, otherwise use CPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using cuda!")
    else:
        print("Using CPU!")
    device = torch.device("cuda" if use_cuda else "cpu")

    # create new model and push it to device
    lstm = LSTM(len(string.printable), args.hidden_size, args.num_layers)
    lstm = lstm.to(device)

    # Adam optimizer with specified learning rate
    optimizer = torch.optim.Adam(lstm.parameters(), lr=args.lr)

    # TODO: Improve this
    # Load input and targets
    training_in, training_targets = read_training_data(args.dataset)
    
    # split into training and validation sets
    val_size = int(args.val_split*len(training_in))
    validation_in = []
    validation_targets = []
    for i in range(val_size):
        index = random.randint(0, len(training_in) - 1)
        validation_in.append(training_in.pop(index))
        validation_targets.append(training_targets.pop(index))

    # list of all batches as tensors
    batches: List[torch.Tensor] = []
    batches_targets: List[torch.Tensor] = []
    
    # prepare batches and encode inputs
    sequence_size = len(training_in[0])
    # [seq_size, batch_size, one_hot_size]
    for i in range(0, len(training_in), args.batch_size):
        batch_tensor = []
        sequences_in_batch_size = training_in[i:i+args.batch_size]
        # Look app sequences in batch
        for sequence in sequences_in_batch_size:
            one_hotted_sequence = one_hot(sequence)
            batch_tensor.append(one_hotted_sequence)

        batch_tensor = torch.as_tensor(batch_tensor, dtype=torch.float32)
        batches.append(batch_tensor)

        targets = training_targets[i:i+args.batch_size]
        targets = torch.as_tensor(targets, dtype=torch.float32)
        batches_targets.append(targets)

    # metrics for plots
    epoch_train_losses = []
    epoch_val_losses = []

    # Training
    for i in range(args.epochs):
        epoch_train_loss = []
        epoch_val_loss = []
        
        # learn from all batches
        for batch, target in zip(batches, batches_targets):
            hidden_state, cell_state = lstm.init_lstm_state(len(batch))
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)
            optimizer.zero_grad()

            loss = 0

            # for every character in sequence (each sequence in parallel)
            for i in range(sequence_size):
                lstm_input = batch[:, i, :]
                lstm_input = lstm_input[None, :, :]
                lstm_input = lstm_input.to(device)
                # print(lstm_input[0][:])

                lstm_state = (hidden_state, cell_state)
                predicted, lstm_state = lstm.forward(
                    lstm_input, lstm_state)

                # calculate loss
                predicted = predicted[0, :, 0].to(device)

                # get i-th target of each sequence in batch
                target_1D = target[:, i].to(device)
                loss += loss_fun(predicted, target_1D)

            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_train_losses.append(sum(epoch_train_loss)/len(epoch_train_loss))
        
        # validation
        # val_loss = 0
        # for all validation sequences:
            #lstm_state = LSTM.init_lstm_state(args.batch_size)
            #lstm_state = lstm_state.to(device)
            # for all characters in sequence / sentence:
            #    predicted, lstm_state = LSTM.forward(batch_inputs, lstm_state)
            #    loss += loss_fun(predicted, batch_targets)
            #val_loss += loss_fun
    print(epoch_train_losses)

def loss_fun(pred, target):
    # binary cross entropy for binary classification [upper-/ lowercase]
    return F.binary_cross_entropy(pred, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--hidden_size", type=int, default=700)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
