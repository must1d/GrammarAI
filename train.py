import argparse
from pathlib import Path
from datetime import datetime
import time
from typing import List
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import read_training_data, one_hot, alphabet, print_progress_bar
from networks import LSTM


def main(args):
    start_time = datetime.now().strftime("%y-%m-%d-%H.%M.%S")

    # first check if cuda is available, otherwise use CPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using cuda!")
    else:
        print("Using CPU!")
    device = torch.device("cuda" if use_cuda else "cpu")

    # create new model and push it to device
    lstm = LSTM(len(alphabet), args.hidden_size, args.num_layers)
    lstm = lstm.to(device)

    # Adam optimizer with specified learning rate
    optimizer = torch.optim.Adam(lstm.parameters(), lr=args.lr)

    # TODO: Improve this
    # Load input and targets
    training_inputs, training_targets = read_training_data(args.dataset)

    # split into training and validation sets
    val_size = int(args.val_split*len(training_inputs))
    validation_inputs = []
    validation_targets = []
    # TODO Replace this with random.shuffle() and pick elements
    for i in range(val_size):
        index = random.randint(0, len(training_inputs) - 1)
        validation_inputs.append(training_inputs.pop(index))
        validation_targets.append(training_targets.pop(index))

    # list of all batches as tensors
    batches_input_tensors: List[torch.Tensor] = []
    batches_target_tensors: List[torch.Tensor] = []

    # prepare batches and encode inputs and targets
    sequence_size = len(training_inputs[0])
    for i in range(0, len(training_inputs), args.batch_size):
        batch_inputs = training_inputs[i:i+args.batch_size]
        batch_targets = training_targets[i:i+args.batch_size]

        training_input_tensor = []      # [seq_size, batch_size, one_hot_size]
        training_targets_tensor = []    # [seq_size, batch_size, one_hot_size]
        for (sequence_in, sequence_out) in zip(batch_inputs, batch_targets):
            training_input_tensor.append(one_hot(sequence_in))
            training_targets_tensor.append(one_hot(sequence_out))
        training_input_tensor = torch.as_tensor(training_input_tensor, dtype=torch.float32)
        training_targets_tensor = torch.as_tensor(training_targets_tensor, dtype=torch.float32)

        batches_input_tensors.append(training_input_tensor)
        batches_target_tensors.append(training_targets_tensor)

    # prepare validation inputs and targets as tensors
    validation_input_tensor = []    # [seq_size, len(validation_inputs), one_hot_size]
    validation_targets_tensor = []  # [seq_size, len(validation_inputs), one_hot_size]
    for (sequence_in, sequence_out) in zip(validation_inputs, validation_targets):
        validation_input_tensor.append(one_hot(sequence_in))
        validation_targets_tensor.append(one_hot(sequence_out))
    validation_input_tensor = torch.as_tensor(validation_input_tensor, dtype=torch.float32)
    validation_targets_tensor = torch.as_tensor(validation_targets_tensor, dtype=torch.float32)

    # TODO: add accuracy as metric
    # metrics for plots to track performance in training
    epoch_train_losses = []
    epoch_val_losses = []

    # Training
    for i in range(args.epochs):
        num_epoch = i + 1
        # print("Epoch " + str(i + 1) + " starting:")

        epoch_train_loss = []

        # go through all batches
        for j, (batch, target) in enumerate(zip(batches_input_tensors, batches_target_tensors)):
            time_batch_start = time.time()

            optimizer.zero_grad()

            hidden_state, cell_state = lstm.init_lstm_state(len(batch))
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)

            loss = 0

            # for every character in sequence (each sequence in parallel)
            for i in range(sequence_size):
                lstm_input = batch[:, i, :]
                lstm_input = lstm_input[None, :, :]
                lstm_input = lstm_input.to(device)

                lstm_state = (hidden_state, cell_state)
                predicted, lstm_state = lstm.forward(
                    lstm_input, lstm_state
                )

                target_2D = target[:, i].to(device)
                # TODO: Check if better
                # loss2 = loss_fun(predicted, target_2D)
                # loss2.backward()
                # optimizer.step()
                loss += loss_fun(predicted, target_2D)

            epoch_train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            time_batch = time.time() - time_batch_start
            time_remaining = round((len(batches_input_tensors) - j - 1) * time_batch, 1)
            time_pred_string = f"Time remaining: {time_remaining}"
            print_progress_bar(
                j,
                len(batches_input_tensors) - 1,
                f"Epoch {num_epoch}",
                suffix=", " + time_pred_string + "s",
                fill_char="#"
            )

        # Validation
        hidden_state, cell_state = lstm.init_lstm_state(len(validation_inputs))
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)

        val_loss = 0
        for i in range(sequence_size):
            lstm_input = validation_input_tensor[:, i, :]
            lstm_input = lstm_input[None, :, :]
            lstm_input = lstm_input.to(device)

            lstm_state = (hidden_state, cell_state)
            predicted, lstm_state = lstm.forward(
                lstm_input, lstm_state
            )

            target_2D = validation_targets_tensor[:, i].to(device)
            val_loss += loss_fun(predicted, target_2D)
        epoch_val_losses.append(val_loss.item())

        epoch_train_losses.append(sum(epoch_train_loss)/len(epoch_train_loss))

    # Save network
    path = f"models/lstm_e={args.epochs}_bs={args.batch_size}_t={start_time}"
    os.mkdir(path)
    torch.save(lstm.state_dict(), f"{path}/model")
    # Plot
    plot_training(epoch_train_losses, epoch_val_losses, path)


def plot_training(epoch_train_losses, epoch_val_losses, path):
    plt.title("Losses")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    x_etl = np.arange(1, len(epoch_train_losses) + 1)
    plt.plot(x_etl, epoch_train_losses, "-r", label="Training Loss")
    x_evl = np.arange(1, len(epoch_val_losses) + 1)
    plt.plot(x_evl, epoch_val_losses, "-b", label="Validation Loss")
    plt.legend(loc="upper right")
    plt.savefig(f"{path}/losses.png")
    # plt.show()


def loss_fun(pred, target):
    return F.cross_entropy(pred, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
