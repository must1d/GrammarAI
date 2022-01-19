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

        # NN outputs probability of letter being uppercase or lowercase
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, input, lstm_state):
        # input [1, batch_size, one_hot_size]

        # Forward propagate LSTM [lstm_state = (hidden, cell)]
        # output [1, batch_size, hidden_size]
        output, lstm_state = self.lstm(input, lstm_state)

        # output [1, batchsize, 1]
        output = self.output_layer(output)

        # probability between 0 and 1
        output = torch.sigmoid(output)

        return output

    def init_lstm_state(self, num_inputs):
        zeros = torch.zeros(self.num_layers, num_inputs, self.hidden_size)
        return (zeros, zeros)


batch_size = 30
vec_size = 15
hidden_size = 5
num_layers = 1

input = torch.zeros(1, batch_size, vec_size)

nn = LSTM(vec_size, hidden_size, num_layers)
nn(input, nn.init_lstm_state(batch_size))
