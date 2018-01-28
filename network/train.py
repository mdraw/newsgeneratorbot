#!/usr/bin/env python3


import argparse
import datetime
import math
import os
import random
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from unidecode import unidecode
from tensorboardX import SummaryWriter

from network.generate import generate
from network.helpers import char_tensor, random_letter, n_all_characters
from network.model import CharRNN


parser = argparse.ArgumentParser()
parser.add_argument('textfile', type=str)
parser.add_argument('--n-steps', type=int, default=20000)
parser.add_argument('--checkpoint-every', type=int, default=100)
parser.add_argument('--preview-length', type=int, default=200)
parser.add_argument('--preview-german', action='store_true',
    help='Convert digraphs in the generated preview text to German umlauts.'
)
parser.add_argument('--hidden-size', type=int, default=800)
parser.add_argument('--n-layers', type=int, default=1)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--chunk-len', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--num-workers', type=int, default=1)
args = parser.parse_args()

if args.cuda:
    print("Using CUDA")


model_name = os.path.splitext(os.path.basename(args.textfile))[0]
timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')


# TODO: Maybe support lazy loading for large corpora
# TODO: Try to use one line each as training samples, don't cut fixed-size chunks.
class TextChunkDataset(Dataset):
    def __init__(self, filename, chunk_len, n_steps, batch_size=args.batch_size):
        with open(filename) as f:
            self.text = f.read()
        self.n_chars = len(self.text)
        self.n_steps = n_steps
        self.chunk_len = chunk_len
        self.batch_size = batch_size  # This is just used for progress tracking

    def __getitem__(self, index):
        # index is currently currently ignored.
        # TODO: Respect index and defer randomization to DataLoader
        start_index = random.randint(0, self.n_chars - self.chunk_len - 1)
        end_index = start_index + self.chunk_len + 1
        chunk = self.text[start_index:end_index]
        inp = char_tensor(chunk[:-1])
        target = char_tensor(chunk[1:])
        if not inp.shape[0] == self.chunk_len and inp.shape == target.shape:  # For debugging
            import IPython ; IPython.embed()
        return inp, target

    def __len__(self):
        # return self.n_chars // self.chunk_len
        # Currently abusing __len__ to be variable. I am not sure why we
        # should let this represent an actual epoch size.
        return self.n_steps * self.batch_size


# Not working.
class TextLineDatasetLazy(Dataset):
    def __init__(self, filename):
        raise NotImplementedError
        self.file = open(filename)
        self.line_offsets = []
        offset = 0
        for line in self.file:
            self.line_offsets.append(offset)
            offset += len(line.strip())
        self.file.seek(0)

    def __getitem__(self, index):
        # TODO: Is the selection too specific? Should we randomly slice lines here?
        self.file.seek(self.line_offsets[index])
        line = self.file.readline()
        # return line
        inp_text = unidecode(line[:-1])
        target_text = unidecode(line[1:])
        inp = char_tensor(inp_text)
        target = char_tensor(target_text)
        return inp, target

    def __len__(self):
        return len(self.line_offsets)


# Not working with DataLoader (requires dynamic/padded batching!)
class TextLineDataset(Dataset):
    def __init__(self, filename):
        raise NotImplementedError
        with open(filename) as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        # TODO: Is the selection too specific? Should we randomly slice lines here?
        line = self.lines[index]
        inp_text = unidecode(line[:-1])
        target_text = unidecode(line[1:])
        inp = char_tensor(inp_text)
        target = char_tensor(target_text)
        return inp, target

    def __len__(self):
        return len(self.lines)


def train(inp, target):
    hidden = model.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    model.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = model(inp[:, c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:, c])

    loss.backward()
    optimizer.step()

    return loss.data[0] / args.chunk_len


def save():
    save_filename = model_name + '.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)


model = CharRNN(
    input_size=n_all_characters,
    hidden_size=args.hidden_size,
    output_size=n_all_characters,
    n_layers=args.n_layers,
)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, verbose=True, factor=0.2, patience=2
)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    model.cuda()

train_data = TextChunkDataset(args.textfile, args.chunk_len, args.n_steps)
train_loader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=False,  # shuffling is done by the DataSet itself currently
    num_workers=args.num_workers
)

# TensorboardX setup
tb_name = model_name + '__' + timestamp
tb_path = os.path.expanduser(f'~/ngtraining/{tb_name}')
os.makedirs(tb_path)
writer = SummaryWriter(tb_path)

start = time.time()
all_losses = []
loss_avg = 0
min_loss = math.inf

print("Training for %d steps..." % args.n_steps)
# try:
for i, batch in enumerate(tqdm(train_loader)):
    inp, target = batch
    inp, target = Variable(inp), Variable(target)
    if args.cuda:
        inp, target = inp.cuda(), target.cuda()
    loss = train(inp, target)
    loss_avg += loss

    writer.add_scalar('tr_loss', loss, i)

    if i % args.checkpoint_every == 0 and i > 0:
        curr_loss = loss_avg / args.checkpoint_every
        curr_lr = optimizer.param_groups[0]['lr']  # Assumes no groups
        print(f'\n\nLoss: {curr_loss:.4f}. Best loss was {min_loss:.4f}.')
        if curr_loss < min_loss:
            min_loss = curr_loss
            print('Best loss so far. Saving model...')
            save()
        scheduler.step(curr_loss)  # TODO: Use validation loss instead
        writer.add_scalar('lr', curr_lr, i)
        loss_avg = 0
        preview_text = generate(
            model=model,
            prime_str=random_letter(),
            predict_len=args.preview_length,
            german=args.preview_german,
            cuda=args.cuda
        )
        print(f'\n"""\n{preview_text}\n"""\n')
        writer.add_text('Text', preview_text, i)
        writer.file_writer.flush()
        all_losses.append(curr_loss)
# except:
#     import traceback
#     traceback.print_exc()
#     cont = False
#     import IPython; IPython.embed()
#     if not cont:
#         raise SystemExit
