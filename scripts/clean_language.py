#!/usr/bin/env python3

import argparse
import os

import langdetect
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    'lang', help='Language code of the only allowed language, e.g. en or de.'
)
parser.add_argument('input', help='File name of the text to clean.')
parser.add_argument('--output', help='Output file name', default=None)
parser.add_argument('--quiet', action='store_true')
args = parser.parse_args()

def main():
    lang = args.lang
    inpath = os.path.abspath(os.path.expanduser(args.input))
    if args.output is None:
        outpath = os.path.splitext(inpath)[0] + f'_{lang}_only.txt'
    else:
        outpath = os.path.expanduser(args.output)

    with open(inpath) as infile:
        lines = infile.readlines()

    foreign_lines = []
    with open(outpath, 'w') as outfile:
        for line in tqdm(lines):
            try:
                if langdetect.detect(line) == lang:
                    outfile.write(line)
                else:
                    foreign_lines.append(line)
            except langdetect.lang_detect_exception.LangDetectException:
                pass  # No features found -> ignore this line

    if not args.quiet:
        print('Lines that were detected as not {lang}:')
        for fl in foreign_lines:
            print(fl)
        print()
        print(f'{len(foreign_lines)} lines were removed.')
        print(f'Stored cleaned version in {outpath}.')


if __name__ == '__main__':
    main()





