#!/usr/bin/env python3

import argparse
import logging
import os
import traceback
from uuid import uuid4

import torch
from telegram.parsemode import ParseMode
from telegram.ext import Updater, CommandHandler, InlineQueryHandler
from telegram import InlineQueryResultArticle, InputTextMessageContent

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
parser.add_argument(
    '-g', '--disable-german', action='store_true',
    help='Convert digraphs in the generated text to German umlauts.'
)
parser.add_argument('--cuda', action='store_true')
parser.add_argument(
    '--token-file', type=str, default='~/.newsgeneratorbot/token'
)
cli_args = parser.parse_args()
cli_args.german = not cli_args.disable_german

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


STARTTEXT = """
Hi, I am a news generator bot. I will generate short news articles for you \
on demand, using a neural network.

I am open source: https://github.com/mdraw/newsgeneratorbot.
""".strip()

HELPTEXT = """
Instructions:

To request a random freshly generated news article, just send "/w" and \
you will receive one after about 2 seconds.

You can optionally specify a writing prompt text that is used to prime the \
title generator. The generated title is then used to prime the article \
content generator, so the content is related to the title.
To specify a writing prompt, just write it after the "/w" before sending \
the request.
E.g. if you want the title to start with "Donald T" and want me to complete \
it to a full article, send "/w Donald T".
""".strip()


def start(bot, update):
    update.message.reply_text(STARTTEXT + '\n\n' + HELPTEXT)


def help(bot, update):
    update.message.reply_text(HELPTEXT)


def sanitize_html(text):
    """Prevent parsing errors by converting <, >, & to their HTML codes."""
    return text\
        .replace('<', '&lt;')\
        .replace('>', '&gt;')\
        .replace('&', '&amp;')


def generate_reply(prime_str):
    if not cli_args.disable_titles:
        generated_title = generate(
            model=title_rnn,
            prime_str=prime_str,
            temperature=cli_args.temperature,
            german=cli_args.german,
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
        prime_str = sanitize_html(generated_title)
    generated_content = generate(
        model=content_rnn,
        prime_str=prime_str,
        predict_len=cli_args.predict_len,
        temperature=cli_args.temperature,
        german=cli_args.german,
        cuda=cli_args.cuda,
        until_last='.'
    )
    # If a title is generated and used as the prime string for content
    # generation, we need to cut it from the beginning of the content,
    # so it isn't duplicated.
    if not cli_args.disable_titles:
        generated_content = generated_content[len(prime_str):]
    generated_content = sanitize_html(generated_content)
    if cli_args.disable_titles:
        full_text = generated_content
    else:
        full_text = f'<b>{generated_title}</b>\n\n{generated_content}'
    # TODO: Sanitize/escape full_text. Some text sequences lead to HTML parser errors.
    return full_text


def write(bot, update, args):
    logger.info(f'New write request by {update.message.from_user.first_name}.')
    if args:
        prime_str = ' '.join(args)
        logger.info(f'Received writing prompt "{prime_str}".')
    else:
        prime_str = random_letter()

    try:
        full_text = generate_reply(prime_str)
        logger.info(f'Replying with:\n"""\n{full_text}\n"""\n')
        update.message.reply_text(full_text, parse_mode=ParseMode.HTML)
    except:
        traceback.print_exc()
        update.message.reply_text(
            'Sorry, I couldn\'t generate a text. Please retry with a different '
            'writing prompt.'
        )


def inlinequery(bot, update):
    logger.info(f'New inline request by {update.inline_query.from_user.first_name}.')
    query = update.inline_query.query
    if query:
        prime_str = query
    else:
        prime_str = random_letter()

    try:
        full_text = generate_reply(prime_str)
        logger.info(f'Replying with:\n"""\n{full_text}\n"""\n')
        answer = InlineQueryResultArticle(
            id=uuid4(),
            title=f'Generate article ("{prime_str}...")',
            input_message_content=InputTextMessageContent(
                full_text, parse_mode=ParseMode.HTML
            )
        )
        update.inline_query.answer([answer])
    except:
        traceback.print_exc()



def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    # Create the EventHandler and pass it your bot's token.
    updater = Updater(token)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('help', help))
    dp.add_handler(CommandHandler('w', write, pass_args=True))
    dp.add_handler(InlineQueryHandler(inlinequery))

    dp.add_error_handler(error)

    updater.start_polling()
    logger.info('Setup complete. Listening for events...')
    updater.idle()


if __name__ == '__main__':
    main()
