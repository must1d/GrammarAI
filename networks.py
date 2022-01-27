import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # LSTM Module
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # NN outputs a character (in one hot encoding)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, input, lstm_state):
        # input [batch_size, sequence_length, one_hot_size]

        # output [batch_size, sequence_length, hidden_size]
        output, lstm_state = self.lstm(input, lstm_state)

        # output [batch_size * sequence_length, one_hot_size]
        output = output.reshape(output.size()[0]*output.size()[1], self.hidden_size)

        # output [batch_size * sequence_length, one_hot_size]
        # output contains every single character of batch with corresponding one hot vector
        output = self.output_layer(output)

        return output, lstm_state

    def init_lstm_state(self, num_inputs):
        # num_inputs: Inputs to LSTM / batch size
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, num_inputs, self.hidden_size).zero_(),
                weight.new(self.num_layers, num_inputs, self.hidden_size).zero_())
