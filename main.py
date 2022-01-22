import argparse
from pathlib import Path
import torch
from networks import LSTM
from utils import one_hot, alphabet


def main(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using cuda!")
    else:
        print("Using CPU!")
    device = torch.device("cuda" if use_cuda else "cpu")

    # load the model
    # TODO: Read hidden size and number of layers from file
    lstm = LSTM(len(alphabet), 300, 2)
    lstm.load_state_dict(torch.load(args.model / "model"))
    lstm.eval()
    lstm = lstm.to(device)
    # model = torch.load(args.model / "model")

    # prepare the sentence as tensor in onehot encoding
    # [1, sentence_length, one_hot_size]
    sentence = args.sentence
    sentence_tensor = one_hot(sentence)
    sentence_tensor = torch.as_tensor(sentence_tensor, dtype=torch.float32)

    output_sentence = ""

    # do the forward computations
    hidden_state, cell_state = lstm.init_lstm_state(1)
    hidden_state = hidden_state.to(device)
    cell_state = cell_state.to(device)

    for i in range(len(sentence)):
        # input each character as one hot encoding into LSTM
        lstm_input = sentence_tensor[i, :]
        lstm_input = lstm_input[None, None, :]  # [1, 1, one_hot_size]
        lstm_input = lstm_input.to(device)

        lstm_state = (hidden_state, cell_state)
        predicted, lstm_state = lstm.forward(
            lstm_input, lstm_state
        )

        # decode result to character
        index_max = torch.argmax(predicted[0])
        char = sentence[i] if index_max == 98 else alphabet[index_max]
        output_sentence += char

    print(output_sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", type=str)
    parser.add_argument("--model", type=Path, default=Path("models/main_model"))
    args = parser.parse_args()
    main(args)
