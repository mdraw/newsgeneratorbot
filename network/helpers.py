import random
import string

import torch
import unidecode


all_characters = string.printable
n_all_characters = len(all_characters)


def read_ascii(filename):
    """Read file and return its ascii transliteration (lossy)"""
    with open(filename) as f:
        content_raw = f.read()
        content_ascii = unidecode.unidecode(content_raw)
    return content_ascii


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