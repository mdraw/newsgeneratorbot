#!/usr/bin/env python3

import argparse
import os

from unidecode import unidecode


parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args()


# Remember to keep this in sync with ../network/helpers.py
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


def main():
    inpath = os.path.abspath(os.path.expanduser(args.input))
    if args.output is None:
        outpath = os.path.splitext(inpath)[0] + '_ascii.txt'
    else:
        outpath = os.path.expanduser(args.output)

    with open(inpath) as infile:
        with open(outpath, 'w') as outfile:
            for line in infile.readlines():
                transtext = transliterate(line, mode='german2ascii')
                asciitext = unidecode(transtext)
                outfile.write(asciitext)


if __name__ == '__main__':
    main()
