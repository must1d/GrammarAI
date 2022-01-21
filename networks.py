import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # LSTM Module
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # NN outputs a character (in one hot encoding)
        self.output_layer = nn.Linear(hidden_size, input_size)
        # Softmax layer
        self.softmax = nn.Softmax()

    def forward(self, input, lstm_state):
        # input [1, batch_size, one_hot_size]

        # Forward propagate LSTM [lstm_state = (hidden, cell)]
        # output [1, batch_size, hidden_size]
        output, lstm_state = self.lstm(input, lstm_state)

        # output [1, batchsize, one_hot_size]
        output = self.output_layer(output)

        # use softmax function to output probabilities
        # output [batchsize, one_hot_size]
        output = self.softmax(output[0])

        return output, lstm_state

    def init_lstm_state(self, num_inputs):
        # num_inputs: Inputs to LSTM / batch size
        zeros = torch.zeros(
            self.num_layers,
            num_inputs,
            self.hidden_size,
            dtype=torch.float32
        )
        return (zeros, zeros)
