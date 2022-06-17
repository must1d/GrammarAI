import argparse
from pathlib import Path
import torch
import numpy as np
from networks import LSTM
from utils import one_hot, alphabet


def main(args):
    device = determine_device_to_use()
    lstm = load_model(path=args.model/"model", device=device)
    sentence_tensor = encode_sentence_in_one_hot(sentence=args.sentence)
    output_sentence = correct_sentence(device, lstm, args.sentence, sentence_tensor)
    print(output_sentence)

def correct_sentence(device, lstm, sentence, sentence_tensor):
    lstm_state = initialize_lstm_state(device, lstm)
    lstm_input = sentence_tensor.to(device)
    return do_forward_computation(lstm, sentence, lstm_state, lstm_input)
    
def do_forward_computation(lstm, sentence, lstm_state, lstm_input):
    output_sentence = ""
    predicted, __ = lstm.forward(lstm_input, lstm_state)
    for i, char_pred in enumerate(predicted):
        index = torch.argmax(char_pred)
        char = sentence[i] if index == 98 else alphabet[index]
        output_sentence += char
    return output_sentence

def initialize_lstm_state(device, lstm):
    hidden_state, cell_state = lstm.init_lstm_state(1)
    hidden_state = hidden_state.to(device)
    cell_state = cell_state.to(device)
    lstm_state = (hidden_state, cell_state)
    return lstm_state

def encode_sentence_in_one_hot(sentence):
    sentence_tensor = np.zeros((1, len(sentence), len(alphabet)))
    sentence_tensor[0][:][:] = one_hot(sentence)[:][:]
    sentence_tensor = torch.as_tensor(sentence_tensor, dtype=torch.float32)
    return sentence_tensor

def determine_device_to_use():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def load_model(path, device):
    checkpoint = torch.load(path, map_location=torch.device(device=device))
    lstm = LSTM(len(alphabet), checkpoint['hidden_size'], checkpoint['num_layers'])
    lstm.load_state_dict(checkpoint['state_dict'])
    lstm.eval()
    lstm = lstm.to(device)
    return lstm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", type=str)
    parser.add_argument("--model", type=Path, default=Path("models/main_model"))
    args = parser.parse_args()
    main(args)
