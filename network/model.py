import torch
import torch.nn as nn
from torch.autograd import Variable


class CharRNN(nn.Module):
    """
    Generative multi-layer GRU model for character-level text modelling.
    The most important variable to manually tune here is the hidden_size,
    which determines the number of hidden units in the middle layers
    """
    def __init__(
            self, input_size, hidden_size, output_size, n_layers=2, dropout=0.1
    ):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, inp, hidden):
        batch_size = inp.size(0)
        # encode into embedding because float inputs are required for nn.GRU
        encoded = self.encoder(inp)
        # Main GRU computation, producing a new prediction from the encoded input and the previous
        # hidden state
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        # Transform the GRU output into the desired output space of ASCII indices
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros for starting off predictions.
        return Variable(torch.zeros(
            self.n_layers, batch_size, self.hidden_size
        ))
