#!/usr/bin/env python3

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('input', help='File name of the text to deduplicate.')
parser.add_argument('--output', help='Output file name', default=None)
parser.add_argument('--quiet', action='store_true')
args = parser.parse_args()


def main():
    inpath = os.path.abspath(os.path.expanduser(args.input))
    if args.output is None:
        outpath = os.path.splitext(inpath)[0] + '_dedup.txt'
    else:
        outpath = os.path.expanduser(args.output)

    with open(inpath) as infile:

        lines = infile.readlines()
        seen = set()

        duplicate_lines = {
            line for line in lines  # Every line in the file
            if line != '\n'  # ... that is not a newline
            and line in seen  # ... and has already been read
            or seen.add(line)  # (if not, it is added to seen)
        }
        if not args.quiet:
            print('Duplicate lines found:\n')
            for d in duplicate_lines:
                print(d)
            print()

    with open(outpath, 'w') as outfile:
        for line in lines:
            if line not in duplicate_lines:
                outfile.write(line)
    if not args.quiet:
        print(f'Stored deduplicated version in {outpath}.')


if __name__ == '__main__':
    main()





