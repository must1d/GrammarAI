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

    # Load input and targets
    training_inputs, training_targets = read_training_data(args.dataset)
    sequence_size = len(training_inputs[0])

    # shuffle inputs and targets
    temp_list = list(zip(training_inputs, training_targets))
    random.shuffle(temp_list)
    training_inputs, training_targets = zip(*temp_list)

    # split into training and validation sets
    val_size = int(args.val_split * len(training_inputs))
    validation_inputs = training_inputs[0:val_size - 1]
    validation_targets = training_targets[0:val_size - 1]
    training_inputs = training_inputs[val_size:-1]
    training_targets = training_targets[val_size:-1]

    # prepare batches
    training_input_tensors, training_target_tensors = prepare_batches(
        training_inputs,
        training_targets,
        args.batch_size)
    validation_input_tensors, validation_target_tensors = prepare_batches(
        validation_inputs,
        validation_targets,
        args.batch_size)

    # metrics for plots to track performance in training
    epoch_train_losses = []
    epoch_val_losses = []
    if args.accuracy:
        print("Warning: Accuracy is enabled! This significantly slows down learning!")
        epoch_train_accuracy = []
        epoch_val_accuracy = []

    # Training
    for num_epoch in range(args.epochs):
        # metric variables
        epoch_train_loss = []
        if args.accuracy:
            hits = 0
            attempts = 0

        # go through all batches
        for j, (batch, target) in enumerate(zip(training_input_tensors, training_target_tensors)):
            time_batch_start = time.time()

            # initializes hidden state
            hidden_state, cell_state = lstm.init_lstm_state(len(batch))
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)
            lstm_state = (hidden_state, cell_state)

            optimizer.zero_grad()

            batch = batch.to(device)
            target_reshaped = target.reshape(len(batch) * sequence_size, len(alphabet)).to(device)

            predicted, __ = lstm(batch, lstm_state)

            loss = loss_fun(predicted, target_reshaped)

            loss.backward()
            optimizer.step()

            # TODO: accuracy

            epoch_train_loss.append(loss.item())

            # Progress Bar and Time remaining
            time_batch = time.time() - time_batch_start
            time_remaining = round((len(training_input_tensors) - j - 1) * time_batch, 1)
            time_pred_string = f"Time remaining: {time_remaining}"
            print_progress_bar(
                j,
                len(training_input_tensors)-1,
                f"Epoch {num_epoch+1}",
                suffix=", " + time_pred_string + "s",
                fill_char="#"
            )

        # Training Metrics
        epoch_train_losses.append(sum(epoch_train_loss)/len(epoch_train_loss))
        if args.accuracy:
            epoch_train_accuracy.append(hits / attempts)

        # Validation
        epoch_validation_loss = []
        for j, (batch, target) in enumerate(zip(validation_input_tensors, validation_target_tensors)):
            # initializes hidden state
            hidden_state, cell_state = lstm.init_lstm_state(len(batch))
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)
            lstm_state = (hidden_state, cell_state)

            optimizer.zero_grad()

            batch = batch.to(device)
            target_reshaped = target.reshape(len(batch) * sequence_size, len(alphabet)).to(device)

            predicted, __ = lstm(batch, lstm_state)

            loss = loss_fun(predicted, target_reshaped)

            epoch_validation_loss.append(loss.item())

        # Validation Metrics
        epoch_val_losses.append(sum(epoch_validation_loss)/len(epoch_validation_loss))
        if args.accuracy:
            epoch_val_accuracy.append(hits / attempts)

    print(epoch_train_losses)
    print(epoch_val_losses)

    # Save network
    path = f"models/lstm_e={args.epochs}_bs={args.batch_size}_t={start_time}"
    os.mkdir(path)
    torch.save(lstm.state_dict(), f"{path}/model")
    # Plot
    plot_metric("Loss", epoch_train_losses, epoch_val_losses, path)
    if args.accuracy:
        plot_metric("Accuracy", epoch_train_accuracy, epoch_val_accuracy, path)


def batch_to_tensor(inputs, targets):
    inputs_tensor = []
    targets_tensor = []

    for (sequence_in, sequence_out) in zip(inputs, targets):
        inputs_tensor.append(one_hot(sequence_in))
        targets_tensor.append(one_hot(sequence_out))
    inputs_tensor = torch.as_tensor(inputs_tensor, dtype=torch.float32)
    targets_tensor = torch.as_tensor(targets_tensor, dtype=torch.float32)

    return inputs_tensor, targets_tensor


def print_tensor_batch(tensor):
    # [batch_size, seq_size, one_hot_size]
    print(f"Batch: {tensor.size()}")
    for i, sequence in enumerate(tensor):
        sequence_str = ""
        for char in sequence:
            index = torch.argmax(char).item()
            sequence_str += alphabet[index]
        print(f"Sequence {i}: {sequence_str}")


def plot_metric(metric_name, epoch_train_metric, epoch_val_metric, path):
    plt.clf()
    plt.title(metric_name)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    x_etl = np.arange(1, len(epoch_train_metric) + 1)
    plt.plot(x_etl, epoch_train_metric, "-r", label="Training")
    x_evl = np.arange(1, len(epoch_val_metric) + 1)
    plt.plot(x_evl, epoch_val_metric, "-b", label="Validation")
    plt.legend(loc="upper right")
    plt.savefig(f"{path}/{metric_name}.png")


def loss_fun(pred, target):
    return F.cross_entropy(pred, target)


def prepare_batches(inputs, targets, batch_size):
    input_tensors: List[torch.Tensor] = []
    target_tensors: List[torch.Tensor] = []

    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]

        # Tensor sizes: [batch_size, seq_size, one_hot_size]
        training_input_tensor, training_targets_tensor = batch_to_tensor(batch_inputs, batch_targets)

        input_tensors.append(training_input_tensor)
        target_tensors.append(training_targets_tensor)

    return input_tensors, target_tensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--hidden_size", type=int, default=700)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--accuracy", action="store_true")
    args = parser.parse_args()
    main(args)
