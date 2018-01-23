#!/usr/bin/env python3

import argparse
import logging
import os

import torch
from telegram.parsemode import ParseMode
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
    '--content-model',
    default='~/.newsgeneratorbot/content_model.pt',
    help='Trained RNN model for text content (.pt file)'
)
parser.add_argument(
    '--title-model',
    default='~/.newsgeneratorbot/title_model.pt',
    help='Trained RNN model for titles (.pt file)'
)
parser.add_argument('-p', '--default-prime-str', type=str, default='A')
parser.add_argument('-l', '--predict-len', type=int, default=500)
parser.add_argument('-t', '--temperature', type=float, default=0.8)
parser.add_argument('-w', '--disable-titles', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument(
    '--token-file', type=str, default='~/.newsgeneratorbot/token'
)
cli_args = parser.parse_args()

# Read Telegram Bot API token from file (reading the first line)
token_file = os.path.expanduser(cli_args.token_file)
with open(token_file) as f:
    token = f.readlines()[0].strip()

# Load trained RNN models. To produce one of these, use network/train.py
content_rnn = torch.load(
    os.path.expanduser(cli_args.content_model),
    map_location=lambda storage, loc: storage
)
logger.info('Sucessfully loaded content generator model.')

if not cli_args.disable_titles:
    title_rnn = torch.load(
        os.path.expanduser(cli_args.title_model),
        map_location=lambda storage, loc: storage
    )
    logger.info('Sucessfully loaded title generator model.')


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
    if not cli_args.disable_titles:
        generated_title = generate(
            model=title_rnn,
            prime_str=prime_str,
            temperature=cli_args.temperature,
            cuda=cli_args.cuda,
            until_first='\n'
        )
        # Strip newline (text formatting is done later).
        generated_title = generated_title.strip('\n')
        # This is kind of a hack to prime the content generator with "memory"
        # of just having "read" the title. One issue with this is that
        # the last characters of the generated title have overproportionate
        # influence on the first characters of the generated content.
        # Is there a way to evenly distribute the title characters' influence?
        prime_str = generated_title
    generated_content = generate(
        model=content_rnn,
        prime_str=prime_str,
        predict_len=cli_args.predict_len,
        temperature=cli_args.temperature,
        cuda=cli_args.cuda,
        until_last='.'
    )
    # If a title is generated and used as the prime string for content
    # generation, we need to cut it from the beginning of the content,
    # so it isn't duplicated.
    if not cli_args.disable_titles:
        generated_content = generated_content[len(prime_str):]

    full_text = f'*{generated_title}*\n\n{generated_content}'
    logger.info(f'Replying with:\n"""\n{full_text}\n"""\n')
    update.message.reply_text(full_text, parse_mode=ParseMode.MARKDOWN)


def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    # Create the EventHandler and pass it your bot's token.
    updater = Updater(token)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler('hello', hello))
    dp.add_handler(CommandHandler('w', write, pass_args=True))

    dp.add_error_handler(error)

    updater.start_polling()
    logger.info('Setup complete. Listening for events...')
    updater.idle()


if __name__ == '__main__':
    main()
