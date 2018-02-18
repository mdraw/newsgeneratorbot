import random
import string
import warnings

import torch


all_characters = string.printable
n_all_characters = len(all_characters)


def char_tensor(s):
    """Convert a Python string to a tensor.

    The returned tensor is a 1-dimensional LongTensor that represents
    each character as its index in the ASCII table. Its length is the same
    as the string length. Characters that are not ASCII-encodable are ignored
    during conversion.

    Note:
        Unicode 10 (136.690 characters) can be fully represented in 18 bits,
        and LongTensors (64 bits) have more than enough capacity for that.
        The reason why Unicode can't be used directly here although the data
        type would allow it is that the neural network's input and output
        dimensions scale with the number of unique characters it needs to
        recognize. Therefore the network size would be far too large (e.g.
        the last Linear layer would have to at least have a
        133690-dimensional output).
        Maybe this problem can be solved with sparse embedding layers?
    """
    tensor = torch.zeros(len(s)).long()
    for c in range(len(s)):
        try:
            # (PyTorch's CrossEntropyLoss implementation only supports
            # LongTensors, so although the data could be encoded in 7 bits
            # (-> ByteTensor), we have to use 64 bits (-> LongTensor) here.
            tensor[c] = all_characters.index(s[c])
        except ValueError:
            print(f'Skipping "{s[c]}" because it can\'t be encoded.')
            continue
    return tensor


def random_letter(only_capital=True):
    """Randomly select a letter from the alphabet"""
    if only_capital:
        letters = string.ascii_uppercase
    else:
        letters = string.ascii_letters
    return random.choice(letters)


# Remember to keep this in sync with ../scripts/transliterate.py
german2ascii_dict = {
    'ä': '_a',
    'ö': '_o',
    'ü': '_u',
    'Ä': '_A',
    'Ö': '_O',
    'Ü': '_U',
    'ß': '_s',
}


def transliterate(text, mode='ascii2german', dictionary=german2ascii_dict):
    """Transliterate German -> ASCII with custom digraphs or vice versa."""
    trans = text
    for ger, asc in dictionary.items():
        if mode == 'german2ascii':
            trans = trans.replace(ger, asc)
        elif mode == 'ascii2german':
            trans = trans.replace(asc, ger)
    return trans
