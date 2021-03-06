import argparse
import os

import torch
from torch.autograd import Variable

from network.helpers import char_tensor, all_characters, transliterate


def generate(
        model, prime_str='A', predict_len=100, temperature=0.8,
        german=False, until_first=None, until_last=None, min_predict_len=None, cuda=False
):
    """
    Generate a character sequence (text) from a trained model.

    :param model: The trained neural network model that should be used for text generation.
    :param prime_str: String to prime the generator model.
    :param predict_len: Desired number of characters of the generated sequence
        (the actual output length can also be affected by until_first or
        until_last values, see below).
    :param temperature: Determines the randomness of generated texts.
        High values lead to more "creative" outputs.
    :param german: If enabled, postprocess strings to recover German special
        characters from ASCII text.
    :param until_first: Stop text generation early when encountering this character.
        (E.g. titles should only contain one line, so for title generators,
        set until_first='\n')
    :param until_last: After generating a sequence of predict_len, or at least
        min_predict_len characters, the text is cut off at the last occurence
        of this character.
        If min_predict_len is not reached, no cutoff is performed.
        (E.g. article contents should end with a full stop, so for content
        generators, set until_last='.').
    :param min_predict_len: Minimum length that the generated text should
        have. If not manually set, it is automatically set to predict_len // 3.
        This is only used if until_last (see above) is set.
    :param cuda: Use GPU for generating texts.
    :return:
    """
    if until_first is not None and until_last is not None:
        raise ValueError(
            'You can\'t specify both `until_first` AND `until_last`.\n'
            'Please decide for one of them and keep the other `None`.'
        )

    hidden = model.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Prime the hidden state so that the network is in the state of "just having read" prime_str.
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[:, p], hidden)

    inp = prime_input[:, -1]

    # Recursively predict each the next char until predict_len is reached.
    for p in range(predict_len):
        output, hidden = model(inp, hidden)

        # Sample from the network as a multinomial distribution
        # Divide by the temperature to give less probable character candidates a chance
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]  # look up in the ASCII table and convert back to a python string
        predicted += predicted_char  # Add to generated sequence
        if predicted_char == until_first:  # If the limiter is reached.
            break
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    if until_last is not None:  # Cut off superfluous text
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
    parser.add_argument('modelfile', type=str,
                        help='File name of the trained model (*.pt)')
    parser.add_argument('-p', '--prime-str', type=str, default='A',
                        help='String for priming the generator')
    parser.add_argument('-l', '--predict-len', type=int, default=1000,
                        help='How long the generated text should be.')
    parser.add_argument('-t', '--temperature', type=float, default=0.8,
                        help='Determines the randomness of generated texts. High values lead to more "creative" outputs.')
    parser.add_argument('-f', '--until-first', default=None)
    parser.add_argument('-u', '--until-last', default='.')
    parser.add_argument('-m', '--min-predict-len', type=int, default=None)
    parser.add_argument('-g', '--german', action='store_true',
        help='Convert digraphs in the generated text to German umlauts.'
    )
    parser.add_argument('--cuda', action='store_true', help='Use GPU for generating texts.')
    args = parser.parse_args()

    try:
        model = torch.load(os.path.expanduser(args.modelfile))
    except AssertionError:  # Loading cuda model without --cuda
        model = torch.load(
            os.path.expanduser(args.modelfile),
            map_location=lambda storage, loc: storage
        )
    generated_text = generate(
        model=model,
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
