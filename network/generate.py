#!/usr/bin/env python3

import argparse
import os

import torch
from torch.autograd import Variable

from network.helpers import char_tensor, all_characters, transliterate


def generate(
        decoder, prime_str='A', predict_len=100, temperature=0.8,
        german=False, until_first=None, until_last=None, min_predict_len=None, cuda=False
):
    if until_first is not None and until_last is not None:
        raise ValueError(
            'You can\'t specify both `until_first` AND `until_last`.\n'
            'Please decide for one of them and keep the other `None`.'
        )

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
        if predicted_char == until_first:
            break
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    if until_last is not None:
        if min_predict_len is None:
            min_predict_len = predict_len // 3
        if until_last is not None:
            end_index = predicted.rfind(until_last)
        if end_index >= 0 and min_predict_len < end_index:
            predicted = predicted[:end_index+1]

    if german:
        predicted = transliterate(predicted, mode='ascii2german')
    return predicted


# Run as standalone script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modelfile', type=str)
    parser.add_argument('-p', '--prime-str', type=str, default='A')
    parser.add_argument('-l', '--predict-len', type=int, default=1000)
    parser.add_argument('-t', '--temperature', type=float, default=0.8)
    parser.add_argument('-f', '--until-first', default=None)
    parser.add_argument('-u', '--until-last', default='.')
    parser.add_argument('-m', '--min-predict-len', type=int, default=None)
    parser.add_argument('-g', '--german', action='store_true',
        help='Convert digraphs in the generated text to German umlauts.'
    )
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    try:
        decoder = torch.load(os.path.expanduser(args.modelfile))
    except AssertionError:  # Loading cuda model without --cuda
        decoder = torch.load(
            os.path.expanduser(args.modelfile),
            map_location=lambda storage, loc: storage
        )
    generated_text = generate(
        decoder=decoder,
        prime_str=args.prime_str,
        predict_len=args.predict_len,
        temperature=args.temperature,
        german=args.german,
        until_first=args.until_first,
        until_last=args.until_last,
        min_predict_len=args.min_predict_len,
        cuda=args.cuda
    )
    print(generated_text)
