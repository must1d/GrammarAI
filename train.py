import argparse
from pathlib import Path
from datetime import datetime
import time
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
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}!")

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
    training_input_batches, training_target_batches = prepare_batches(
        training_inputs,
        training_targets,
        args.batch_size)
    validation_input_batches, validation_target_batches = prepare_batches(
        validation_inputs,
        validation_targets,
        args.batch_size)

    # metrics for plots to track performance in training
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accuracies = []
    epoch_val_accuracies = []

    # Training
    for num_epoch in range(args.epochs):
        # metric variables
        epoch_train_loss = []       # loss of each batch -> will be averaged
        epoch_val_loss = []         # loss of each batch -> will be averaged
        epoch_train_accuracy = []   # accuracy of each batch -> will be averaged
        epoch_val_accuracy = []     # accuracy of each batch -> will be averaged

        # go through all batches
        for j, (batch, target) in enumerate(zip(training_input_batches, training_target_batches)):
            time_batch_start = time.time()

            # initializes hidden state
            hidden_state, cell_state = lstm.init_lstm_state(len(batch))
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)
            lstm_state = (hidden_state, cell_state)

            optimizer.zero_grad()

            batch_tensors, target_tensors = batch_to_tensor(batch, target)
            batch_tensors = batch_tensors.to(device)
            target_reshaped = target_tensors.reshape(len(batch) * sequence_size, len(alphabet)).to(device)

            predicted, __ = lstm(batch_tensors, lstm_state)

            loss = loss_fun(predicted, target_reshaped)

            loss.backward()
            optimizer.step()

            # metrics for batch
            epoch_train_loss.append(loss.item())
            acc = get_accuracy(predicted, target_reshaped)
            epoch_train_accuracy.append(acc)

            # Progress Bar and Time remaining
            time_batch = time.time() - time_batch_start
            time_remaining = round((len(training_input_batches) - j - 1) * time_batch, 1)
            time_pred_string = f"Time remaining: {time_remaining}"
            print_progress_bar(
                j,
                len(training_input_batches)-1,
                f"Epoch {num_epoch+1}",
                suffix=", " + time_pred_string + "s",
                fill_char="#"
            )

        # Validation
        for j, (batch, target) in enumerate(zip(validation_input_batches, validation_target_batches)):
            # initializes hidden state
            hidden_state, cell_state = lstm.init_lstm_state(len(batch))
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)
            lstm_state = (hidden_state, cell_state)

            optimizer.zero_grad()

            batch_tensors, target_tensors = batch_to_tensor(batch, target)
            batch_tensors = batch_tensors.to(device)
            target_reshaped = target_tensors.reshape(len(batch) * sequence_size, len(alphabet)).to(device)

            predicted, __ = lstm(batch_tensors, lstm_state)

            loss = loss_fun(predicted, target_reshaped)

            # metrics for batch
            epoch_val_loss.append(loss.item())
            acc = get_accuracy(predicted, target_reshaped)
            epoch_val_accuracy.append(acc)

        # Training Metrics
        epoch_train_losses.append(sum(epoch_train_loss)/len(epoch_train_loss))
        epoch_train_accuracies.append(sum(epoch_train_accuracy)/len(epoch_train_accuracy))
        # Validation Metrics
        epoch_val_losses.append(sum(epoch_val_loss)/len(epoch_val_loss))
        epoch_val_accuracies.append(sum(epoch_val_accuracy)/len(epoch_val_accuracy))

    # Save network
    path = f"models/lstm_e={args.epochs}_bs={args.batch_size}_t={start_time}"
    os.mkdir(path)
    checkpoint = {'num_layers': lstm.num_layers,
                  'hidden_size': lstm.hidden_size,
                  'state_dict': lstm.state_dict()}
    torch.save(checkpoint, f"{path}/model")

    # Plot
    plot_metric("Loss", epoch_train_losses, epoch_val_losses, path)
    plot_metric("Accuracy", epoch_train_accuracies, epoch_val_accuracies, path)


def batch_to_tensor(inputs, targets):
    inputs_tensor = np.zeros((len(inputs), len(inputs[0]), len(alphabet)), dtype=np.float32)
    targets_tensor = np.zeros((len(inputs), len(inputs[0]), len(alphabet)), dtype=np.float32)

    for i in range(len(inputs)):
        inputs_tensor[i][:][:] = one_hot(inputs[i])[:][:]
        targets_tensor[i][:][:] = one_hot(targets[i])[:][:]

    inputs_tensor = torch.from_numpy(inputs_tensor)
    targets_tensor = torch.from_numpy(targets_tensor)

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


def get_accuracy(outputs, targets):
    hits = 0

    outputs_indices = torch.argmax(outputs, dim=1).tolist()
    targets_indices = torch.argmax(targets, dim=1).tolist()

    for (index_out, index_target) in zip(outputs_indices, targets_indices):
        hits = hits + 1 if index_out == index_target else hits

    return hits / len(outputs)


def prepare_batches(inputs, targets, batch_size):
    input_batches = []
    target_batches = []

    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]

        input_batches.append(batch_inputs)
        target_batches.append(batch_targets)

    return np.array(input_batches, dtype=object), np.array(target_batches, dtype=object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--hidden_size", type=int, default=700)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
