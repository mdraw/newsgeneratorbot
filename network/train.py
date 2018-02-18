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
from tensorboardX import SummaryWriter

from network.generate import generate
from network.helpers import char_tensor, random_letter, n_all_characters
from network.model import CharRNN


parser = argparse.ArgumentParser()
parser.add_argument('training', type=str)
parser.add_argument('--validation', default=None)
parser.add_argument('--name', default=None)
parser.add_argument('--n-steps', type=int, default=20000)
parser.add_argument('--checkpoint-every', type=int, default=100)
parser.add_argument('--preview-length', type=int, default=200)
parser.add_argument(
    '--preview-german',
    action='store_true',
    help='Convert digraphs in the generated preview text to German umlauts.'
)
parser.add_argument('--hidden-size', type=int, default=800)
parser.add_argument('--n-layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--learning-rate', type=float, default=5e-3)
parser.add_argument('--weight-decay', type=float, default=1e-6)
parser.add_argument('--chunk-len', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument(
    '--valid-count',
    type=int,
    help='How many batches to sample for each validation pass.',
    default=10
)
parser.add_argument(
    '--patience', type=int, help='Patience of the LR scheduler', default=5
)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--num-workers', type=int, default=1)
args = parser.parse_args()


# Try to automatically find a validation data set in the same directory
if args.validation is None:
    training_filename = os.path.basename(args.training)
    if 'train' in training_filename:
        candidate = os.path.join(
            os.path.dirname(args.training),
            training_filename.replace('train', 'valid')
        )
        if os.path.isfile(candidate):
            args.validation = candidate
            print(f'Found validation set {candidate}. Enabling validation.')
if args.validation is None:
    print('Validation set could not be found. Disabling validation.')


if args.cuda:
    print("Using CUDA")

if args.name is None:
    model_name = os.path.splitext(os.path.basename(args.training))[0]
else:
    model_name = args.name
timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')


class TextChunkDataset(Dataset):
    def __init__(self, filename, chunk_len, n_steps, batch_size):
        with open(filename) as f:
            self.text = f.read()
        self.n_chars = len(self.text)
        self.n_steps = n_steps
        self.chunk_len = chunk_len
        self.batch_size = batch_size  # This is just used for progress tracking

    def __getitem__(self, index):
        # index is currently currently ignored.
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


def train(inp, target):
    model.train()
    hidden = model.init_hidden(inp.shape[0])
    if args.cuda:
        hidden = hidden.cuda()
    model.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = model(inp[:, c], hidden)
        loss += criterion(output.view(output.shape[0], -1), target[:, c])

    loss.backward()
    optimizer.step()

    return loss.data[0] / args.chunk_len


# TODO: Proper arguments
def validate():
    model.eval()
    val_loss = 0
    for inp, target in val_loader:
        hidden = model.init_hidden(inp.shape[0])
        inp = Variable(inp, volatile=True)
        target = Variable(target, volatile=True)
        if args.cuda:
            inp = inp.cuda()
            target = target.cuda()
            hidden = hidden.cuda()
        for c in range(args.chunk_len):
            output, hidden = model(inp[:, c], hidden)
            val_loss += criterion(output.view(output.shape[0], -1), target[:, c])
    val_loss /= args.chunk_len
    val_loss /= len(val_loader)
    return val_loss.data[0]


def save():
    save_filename = model_name + '.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)


model = CharRNN(
    input_size=n_all_characters,
    hidden_size=args.hidden_size,
    output_size=n_all_characters,
    n_layers=args.n_layers,
    dropout=args.dropout
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    verbose=True,
    factor=0.2,
    patience=args.patience,
    cooldown=3,
    min_lr=1e-7
)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    model.cuda()
    criterion.cuda()

train_data = TextChunkDataset(
    filename=args.training,
    chunk_len=args.chunk_len,
    n_steps=args.n_steps,
    batch_size=args.batch_size
)
train_loader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=False,  # shuffling is done by the DataSet itself currently
    num_workers=args.num_workers
)

if args.validation:
    val_data = TextChunkDataset(
        filename=args.validation,
        chunk_len=args.chunk_len,
        n_steps=args.valid_count,
        batch_size=args.batch_size
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,  # shuffling is done by the DataSet itself currently
        num_workers=0
    )

# TensorboardX setup
tb_name = model_name + '__' + timestamp
tb_path = os.path.expanduser(f'~/ngtraining/{tb_name}')  # TODO: Make this configurable
os.makedirs(tb_path)
writer = SummaryWriter(tb_path)

start = time.time()
all_losses = []
loss_avg = 0
min_loss = math.inf
min_val_loss = math.inf

print("Training for %d steps..." % args.n_steps)
for i, batch in enumerate(tqdm(train_loader)):
    inp, target = batch
    inp, target = Variable(inp), Variable(target)
    if args.cuda:
        inp, target = inp.cuda(), target.cuda()
    loss = train(inp, target)
    loss_avg += loss

    perplexity = math.exp(loss)
    writer.add_scalar('tr_loss', loss, i)
    writer.add_scalar('tr_perplexity', perplexity, i)

    if i % args.checkpoint_every == 0 and i > 0:
        curr_loss = loss_avg / args.checkpoint_every
        curr_lr = optimizer.param_groups[0]['lr']  # Assumes no groups
        print(f'\n\nTraining loss: {curr_loss:.4f}. '
              f'Best training loss was {min_loss:.4f}.')

        if args.validation:  # Validation loss is available
            val_loss = validate()
            val_perplexity = math.exp(val_loss)
            writer.add_scalar('val_loss', val_loss, i)
            writer.add_scalar('val_perplexity', val_perplexity, i)
            print(f'Validation loss: {val_loss:.4f}. '
                  f'Best validation loss was {min_val_loss:.4f}')
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print('Best validation loss so far. Saving model...')
                save()
            scheduler.step(val_loss)
        else:  # Validation loss not available -> use training loss instead
            if curr_loss < min_loss:
                print('Best loss so far. Saving model...')
                save()
            scheduler.step(curr_loss)

        min_loss = min(curr_loss, min_loss)

        writer.add_scalar('lr', curr_lr, i)  # lr after applying schedule
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
