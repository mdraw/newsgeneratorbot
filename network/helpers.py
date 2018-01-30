import random
import string

import torch


all_characters = string.printable
n_all_characters = len(all_characters)


def char_tensor(s):
    """Convert a Python string to a char tensor (as torch.LongTensor)"""
    tensor = torch.zeros(len(s)).long()
    for c in range(len(s)):
        try:
            tensor[c] = all_characters.index(s[c])
        except:
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
    """Transliterate German -> ascii with custom digraphs or vice versa."""
    trans = text
    for ger, asc in dictionary.items():
        if mode == 'german2ascii':
            trans = trans.replace(ger, asc)
        elif mode == 'ascii2german':
            trans = trans.replace(asc, ger)
    return trans
