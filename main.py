import argparse
from pathlib import Path
import torch
from networks import LSTM
from utils import one_hot, alphabet


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load the model
    checkpoint = torch.load(args.model/"model")
    lstm = LSTM(len(alphabet), checkpoint['hidden_size'], checkpoint['num_layers'])
    lstm.load_state_dict(checkpoint['state_dict'])
    lstm.eval()
    lstm = lstm.to(device)

    # prepare the sentence as tensor in onehot encoding
    # [1, sentence_length, one_hot_size]
    sentence = args.sentence
    sentence_tensor = [one_hot(sentence)]
    sentence_tensor = torch.as_tensor(sentence_tensor, dtype=torch.float32)

    output_sentence = ""

    # do the forward computations
    hidden_state, cell_state = lstm.init_lstm_state(1)
    hidden_state = hidden_state.to(device)
    cell_state = cell_state.to(device)
    lstm_state = (hidden_state, cell_state)

    lstm_input = sentence_tensor.to(device)

    predicted, __ = lstm.forward(lstm_input, lstm_state)

    for i, char_pred in enumerate(predicted):
        index = torch.argmax(char_pred)
        char = sentence[i] if index == 98 else alphabet[index]
        output_sentence += char

    print(output_sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", type=str)
    parser.add_argument("--model", type=Path, default=Path("models/main_model"))
    args = parser.parse_args()
    main(args)
