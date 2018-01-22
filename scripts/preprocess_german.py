import argparse
import os

from unidecode import unidecode


parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args()

inpath = os.path.abspath(os.path.expanduser(args.input))
if args.output is None:
    outpath = os.path.splitext(inpath)[0] + '_ascii.txt'
else:
    outpath = os.path.expanduser(args.output)


# TODO: Change german2ascii and ascii2german to use less ambiguous encoding
#       (not the standard digraphs)
def german2ascii(text):
    """Transliterate German-specific characters to digraphs."""
    trans = text\
        .replace('ä', 'ae')\
        .replace('ö', 'oe')\
        .replace('ü', 'ue')\
        .replace('Ä', 'Ae')\
        .replace('Ö', 'Oe')\
        .replace('Ü', 'Ue')\
        .replace('ß', 'ss')
    return trans


def main():
    with open(inpath) as infile:
        with open(outpath, 'w') as outfile:
            for line in infile.readlines():
                transtext = german2ascii(line)
                asciitext = unidecode(transtext)
                outfile.write(asciitext)


if __name__ == '__main__':
    main()
