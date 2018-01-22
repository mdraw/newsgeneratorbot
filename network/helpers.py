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


def ascii2german(text):
    """Convert digraphs in the text to German umlauts."""
    trans = text\
        .replace('ae', 'ä')\
        .replace('oe', 'ö')\
        .replace('ue', 'ü')\
        .replace('Ae', 'Ä')\
        .replace('Oe', 'Ö')\
        .replace('Ue', 'Ü')\
        .replace('Qü', 'Qu')\
        .replace('qü', 'qu')\
        # .replace('ss', 'ß')  # We can't decide if this was an actual ß
    # (The "qü" replacement is necessary because e.g. "quer" would otherwise
    # be transliterated to "qür")
    return trans
