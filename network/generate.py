#!/usr/bin/env python3

import argparse

import torch
from torch.autograd import Variable

from helpers import char_tensor, all_characters


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:, p], hidden)

    inp = prime_input[:, -1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted


# Run as standalone script
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('modelfile', type=str)
    argparser.add_argument('-p', '--prime-str', type=str, default='A')
    argparser.add_argument('-l', '--predict-len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    decoder = torch.load(args.modelfile)
    generated_text = generate(
        decoder,
        args.prime_str,
        args.predict_len,
        args.temperature,
        args.cuda
    )
    print(generated_text)
