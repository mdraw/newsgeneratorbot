#!/usr/bin/env python3


import argparse
import os
import random
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from generate import generate
from helpers import read_ascii, char_tensor, time_since, n_characters
from model import CharRNN


parser = argparse.ArgumentParser()
parser.add_argument('textfile', type=str)
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--print-every', type=int, default=100)
parser.add_argument('--preview-primer', default='A')
parser.add_argument('--preview-length', type=int, default=100)
parser.add_argument('--hidden-size', type=int, default=100)
parser.add_argument('--n-layers', type=int, default=2)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--chunk-len', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

if args.cuda:
    print("Using CUDA")

text = read_ascii(args.textfile)
text_len = len(text)


def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, text_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = text[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


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
    n_characters,
    args.hidden_size,
    n_characters,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_training_set(args.chunk_len, args.batch_size))
        loss_avg += loss

        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (
                time_since(start), epoch, epoch / args.n_epochs * 100, loss
            ))
            preview_text = generate(
                decoder,
                prime_str=args.preview_primer,
                predict_len=args.preview_length,
                cuda=args.cuda
            )
            print(preview_text, '\n')

    print("Saving...")
    save()
except KeyboardInterrupt:
    print("Saving before quit...")
    save()
