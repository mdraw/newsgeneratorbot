#!/usr/bin/env python3

import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('input', help='File name of the text to split.')
parser.add_argument(
    '--ratio',
    help='Ratio of training/validation lines',
    type=float,
    default=0.9)
parser.add_argument('--quiet', action='store_true')
args = parser.parse_args()


def main():
    split_ratio = args.ratio
    inpath = os.path.abspath(os.path.expanduser(args.input))
    trainpath = os.path.splitext(inpath)[0] + '_train.txt'
    validpath = os.path.splitext(inpath)[0] + '_valid.txt'

    with open(inpath) as infile:
        lines = infile.readlines()

    train_lines = []
    valid_lines = []
    for line in lines:
        if random.random() < split_ratio:
            train_lines.append(line)
        else:
            valid_lines.append(line)

    with open(trainpath, 'w') as f:
        for line in train_lines:
            f.write(line)
    with open(validpath, 'w') as f:
        for line in valid_lines:
            f.write(line)
    if not args.quiet:
        effective_ratio = len(train_lines) / len(lines)
        print(f'Effective ratio: {effective_ratio:.3f}')
        print(f'Lines: train {len(train_lines)}, valid {len(valid_lines)}')
        print(f'Stored splits in\n {trainpath}\n {validpath}')


if __name__ == '__main__':
    main()
