#!/usr/bin/env python3

import argparse
import logging
import os

import torch
from telegram.ext import Updater, CommandHandler

from network.generate import generate
from network.helpers import random_letter


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model-file',
    type=str,
    default='~/.newsgeneratorbot/model.pt',
    help='Trained RNN model (.pt file)'
)
parser.add_argument('-p', '--default-prime-str', type=str, default='A')
parser.add_argument('-l', '--predict-len', type=int, default=100)
parser.add_argument('-t', '--temperature', type=float, default=0.8)
parser.add_argument('--cuda', action='store_true')
parser.add_argument(
    '--token-file', type=str, default='~/.newsgeneratorbot/token'
)
cli_args = parser.parse_args()

# Read Telegram Bot API token from file (reading the first line)
token_file = os.path.expanduser(cli_args.token_file)
with open(token_file) as f:
    token = f.readlines()[0].strip()

# Load a trained RNN model. To produce one of these, use network/train.py
    try:
        decoder = torch.load(os.path.expanduser(cli_args.model_file))
    except AssertionError:  # Loading cuda model without --cuda
        decoder = torch.load(
            os.path.expanduser(cli_args.model_file),
            map_location=lambda storage, loc: storage
        )
logger.info('Sucessfully loaded network model.')


def hello(bot, update):
    update.message.reply_text(
        'Hello {}'.format(update.message.from_user.first_name)
    )


def write(bot, update, args):
    logger.info(f'New write request by {update.message.from_user.first_name}.')
    if args:
        prime_str = ' '.join(args)
        logger.info(f'Received writing prompt "{prime_str}".')
    else:
        prime_str = random_letter()
    generated_text = generate(
        decoder,
        prime_str,
        cli_args.predict_len,
        cli_args.temperature,
        cli_args.cuda
    )
    logger.info(f'Replying with:\n"""\n{generated_text}\n"""\n')
    update.message.reply_text(generated_text)


def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    # Create the EventHandler and pass it your bot's token.
    updater = Updater(token)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler('hello', hello))
    dp.add_handler(CommandHandler('write', write, pass_args=True))

    dp.add_error_handler(error)

    updater.start_polling()
    logger.info('Setup complete. Listening for events...')
    updater.idle()


if __name__ == '__main__':
    main()
