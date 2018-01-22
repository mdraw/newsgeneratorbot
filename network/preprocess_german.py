import argparse
import os

from unidecode import unidecode

from network.helpers import transliterate


parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args()


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
