#!/usr/bin/env python3


import argparse
import os
import random
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from network.generate import generate
from network.helpers import read_ascii, char_tensor, n_all_characters
from network.model import CharRNN


parser = argparse.ArgumentParser()
parser.add_argument('textfile', type=str)
parser.add_argument('--n-steps', type=int, default=20000)
parser.add_argument('--print-every', type=int, default=100)
parser.add_argument('--preview-primer', default='A')
parser.add_argument('--preview-length', type=int, default=100)
parser.add_argument('--hidden-size', type=int, default=100)
parser.add_argument('--n-layers', type=int, default=2)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--chunk-len', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--num-workers', type=int, default=0)
args = parser.parse_args()

if args.cuda:
    print("Using CUDA")


# TODO: Maybe support lazy loading for large corpora
class TextChunkDataset(Dataset):
    def __init__(self, filename, chunk_len, n_samples, batch_size=args.batch_size):
        self.text = read_ascii(filename)
        self.n_chars = len(self.text)
        self.n_samples = n_samples
        self.chunk_len = chunk_len
        self.batch_size = batch_size  # This is just used for progress tracking

    def __getitem__(self, index):
        # index is currently currently ignored.
        # TODO: Respect index and defer randomization to DataLoader
        start_index = random.randint(0, self.n_chars - self.chunk_len)
        end_index = start_index + self.chunk_len + 1
        chunk = self.text[start_index:end_index]
        inp = char_tensor(chunk[:-1])
        target = char_tensor(chunk[1:])
        return inp, target

    def __len__(self):
        # return self.n_chars // self.chunk_len
        # Currently abusing __len__ to be variable. I am not sure why we
        # should let this represent an actual epoch size.
        return self.n_samples * self.batch_size


def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:, c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:, c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / args.chunk_len


def save():
    save_filename = os.path.splitext(os.path.basename(args.textfile))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)


# Initialize model and start training
decoder = CharRNN(
    n_all_characters,
    args.hidden_size,
    n_all_characters,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

train_data = TextChunkDataset(args.textfile, args.chunk_len, args.n_steps)
train_loader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=False,  # shuffling is done by the DataSet itself currently
    num_workers=args.num_workers
)

start = time.time()
all_losses = []
loss_avg = 0

try:
    print("Training for %d steps..." % args.n_steps)
    for i, (inp, target) in enumerate(tqdm(train_loader)):
        inp, target = Variable(inp), Variable(target)
        if args.cuda:
            inp, target = inp.cuda(), target.cuda()
        loss = train(inp, target)
        loss_avg += loss

        if i % args.print_every == 0 and i > 0:
            print(f'\n\nloss = {loss:.4f}')
            preview_text = generate(
                decoder,
                prime_str=args.preview_primer,
                predict_len=args.preview_length,
                cuda=args.cuda
            )
            print('\n"""\n', preview_text, '\n"""\n')

    print("Saving...")
    save()
except KeyboardInterrupt:
    print("Saving before quit...")
    save()
