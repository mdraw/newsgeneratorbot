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
parser.add_argument('training', type=str, help='File name of the training data set (required)')
parser.add_argument('--validation', default=None, help='File name of the validation data set')
parser.add_argument('--name', default=None, help='Experiment name (determines the file name of the model and of tensorboard logs)')
parser.add_argument('--n-steps', type=int, default=20000, help='Total number of optimization steps before stopping the training.')
parser.add_argument('--checkpoint-every', type=int, default=100, help='Interval that determines often validation and preview text generation should occur')
parser.add_argument('--preview-length', type=int, default=200, help='How long the generated text previews should be')
parser.add_argument(
    '--preview-german',
    action='store_true',
    help='Convert digraphs in the generated preview text to German umlauts.'
)
parser.add_argument('--hidden-size', type=int, default=800)
parser.add_argument('--n-layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--learning-rate', type=float, default=5e-3)
parser.add_argument('--weight-decay', type=float, default=1e-6)
parser.add_argument('--chunk-len', type=int, default=200, help='Size of the text chunks (blocks) that are sliced from the training data')
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
parser.add_argument('--cuda', action='store_true', help='Use CUDA')
parser.add_argument('--num-workers', type=int, default=1, help='Number of background processes for data loading')
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

# Determine a name for the model based on the training data set file name.
# This name is used for saving the trained network model and for
# the tensorboard log directory name
if args.name is None:
    model_name = os.path.splitext(os.path.basename(args.training))[0]
else:
    model_name = args.name
timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')


class TextChunkDataset(Dataset):
    """
    A PyToch-DataLoader-compatible data set class that produces
    training samples by randomly selecting text chunks of
    size chunk_len from the given training data set text file.

    The parameters n_steps and batch_size should be the same
    as the CLI arguments with the same name. They are only
    used to set a length of the data set iterator so that
    the number of training batches that are taken from is the
    number of iterations for which the network should be trained.
    """
    def __init__(self, filename, chunk_len, n_steps, batch_size):
        with open(filename) as f:
            self.text = f.read()
        self.n_chars = len(self.text)
        self.n_steps = n_steps
        self.chunk_len = chunk_len
        self.batch_size = batch_size  # This is just used for progress tracking

    def __getitem__(self, index):
        # index is currently currently ignored.

        # Select a random start index inside of the text whole text file,
        # (minus the last few characters, so that the end index
        # won't be out of bounds).
        start_index = random.randint(0, self.n_chars - self.chunk_len - 1)
        end_index = start_index + self.chunk_len + 1
        # Slice a fixed-size text block
        chunk = self.text[start_index:end_index]
        inp = char_tensor(chunk[:-1])  # The last character won't be needed as ground truth
        target = char_tensor(chunk[1:])  # The first character won't be predicted, it has to be given.
        return inp, target

    def __len__(self):
        # return self.n_chars // self.chunk_len
        # Currently abusing __len__ to be variable. I am not sure why we
        # should let this represent an actual epoch size.
        return self.n_steps * self.batch_size


def train(inp, target):
    """
    Training on one training batch

    :param inp: ground truth input character sequence (as LongTensor)
    :param target: ground truth target character sequence (as LongTensor)
    :return: cross entropy loss between target and predicted output
    """
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


def validate():
    """
    Get "--valid-count=10" batches from the validation data set
    and calculate losses on the prediction without optimizing
    the network parameters with it.

    :return: cross entropy loss between targets and predicted outputs
    """
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
    """
    Save the complete model with its trained parameters to a file.

    The file name is determined automatically based on the training data set name,
    or it can manually set by specifying the CLI argument "--name".
    """
    save_filename = model_name + '.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)


# Initialize a network model
model = CharRNN(
    input_size=n_all_characters,
    hidden_size=args.hidden_size,
    output_size=n_all_characters,
    n_layers=args.n_layers,
    dropout=args.dropout
)

# Set up the optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)

# Set up the dynamic learnining rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    verbose=True,
    factor=0.2,
    patience=args.patience,
    cooldown=3,
    min_lr=1e-7
)

# Specify loss function
criterion = nn.CrossEntropyLoss()

if args.cuda:
    model.cuda()
    criterion.cuda()

# Load training data set
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

# If available, load validation data set
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

# Train until --n_steps are reached or until Ctrl-C
print("Training for %d steps..." % args.n_steps)
for i, batch in enumerate(tqdm(train_loader)):
    # Prepare batch
    inp, target = batch
    inp, target = Variable(inp), Variable(target)
    if args.cuda:
        inp, target = inp.cuda(), target.cuda()

    # Train with one batch (predict and optimize)
    loss = train(inp, target)
    loss_avg += loss

    # Log additional metrics to tensorboard
    perplexity = math.exp(loss)
    writer.add_scalar('tr_loss', loss, i)
    writer.add_scalar('tr_perplexity', perplexity, i)

    # On reaching a checkpoint (default: every 100 steps)...
    if i % args.checkpoint_every == 0 and i > 0:
        # Log current training loss
        curr_loss = loss_avg / args.checkpoint_every
        curr_lr = optimizer.param_groups[0]['lr']  # Assumes no groups
        print(f'\n\nTraining loss: {curr_loss:.4f}. '
              f'Best training loss was {min_loss:.4f}.')

        # Validate and log validation metrics
        if args.validation:  # Validation loss is available
            val_loss = validate()
            val_perplexity = math.exp(val_loss)
            writer.add_scalar('val_loss', val_loss, i)
            writer.add_scalar('val_perplexity', val_perplexity, i)
            print(f'Validation loss: {val_loss:.4f}. '
                  f'Best validation loss was {min_val_loss:.4f}')
            # If a new best validation result is reached, save this
            # best-performing model.
            # This mechanism has the effect that the model file (*.pt)
            # for this training always contains the currently best model.
            # If it gets worse (-> overfitting), the model is not overwritten.
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print('Best validation loss so far. Saving model...')
                save()
            scheduler.step(val_loss)  # Reduce learning rate if no validation loss improvement can be measured.
        else:  # Validation loss not available -> use training loss instead
            if curr_loss < min_loss:
                print('Best loss so far. Saving model...')
                save()
            scheduler.step(curr_loss)

        min_loss = min(curr_loss, min_loss)

        writer.add_scalar('lr', curr_lr, i)  # lr after applying schedule
        loss_avg = 0
        # Generate a preview of generated text and log it to tensorboard and stdout.
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
