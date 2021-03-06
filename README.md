# News Generator Bot

A Telegram bot that generates custom news articles on demand, based on a GRU Neural Network.


## Overview


### Neural network (`network/`)

The underlying [GRU](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)-based neural network is implemented in [PyTorch](https://github.com/pytorch/pytorch). It can be trained to model natural language at character level.
For training, see `python3 -m network.train -h`, for text generation see `python3 -m network.generate -h`.
This module can also be used stand-alone without the bot.


### Scripts for data set preparation (`scripts/`)

The `scripts` subdirectory contains Python scripts that automate data set creation:

- `scrape_articles.py`: A tool for conveniently scraping news article contents and titles from any news web site. Built on top of the [newspaper](https://github.com/codelucas/newspaper) scraping library.
- `clean_language.py`: Removes texts of undesired languages from the data set.
- `transliterate.py`: Transliterates unicode text to ASCII. German-specific non-ASCII characters get a custom transliteration for reduced information loss.
- `deduplicate_lines.py`: Removes duplicate lines in data sets. Those occur often in scraped content (ad blocker warnings etc.) and will introduce a heavy bias towards them in the language model if they are not cleaned.
- `split.py`: Splits a data set into training and validation data, with a specified line count ratio between them.

For more usage information see the section [Building a data set for training and validation](#building-a-data-set-for-training-and-validation) below.


### Bot (`newsgeneratorbot.py`)

The Telegram bot is implemented with the [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) library.
Its "official" instance, trained with German news articles, is sometimes running as [@NewsGeneratorBot](https://t.me/newsgeneratorbot), but it will be offline most of the time, because I don't have permanent access to a server where I can host it. It is easy to host your own instance of it if you are interested, though, by following the instructions below.


## Installation, usage and hosting


### Setup

For installation, you will need the [conda](https://conda.io/docs/) package manager.
To install all dependencies for all parts of the project, just run the following commands in your source root:

    conda env create -f environment.yml
    conda activate newsgeneratorbot


### Training and testing the neural network

You will then be able to train a new model with

    python3 -m network.train <path/to/file-train.txt>

If there is a file in the same directory with the same name, but with "valid" instead of "train", this file will be automatically used as validation data.
(For obtaining data sets, you can follow the section [Building a data set for training and validation](#building-a-data-set-for-training-and-validation) below or use your own text files. Just make sure they are at least a few hundred kilobytes big for good results.)

You can monitor the training progress both on the CLI and in [TensorBoard](https://github.com/tensorflow/tensorboard). Loss and perplexity performance on training and validation sets are logged every 100 training steps by default, together with previews of generated text. You can monitor training progress and compare different configurations at https://localhost:6006 after starting a local TensorBoard server with

    tensorboard --logdir ~/ngtraining

The model with the best performing trained parameter state will be saved to a `.pt` file in the CWD, with a file name based on your text data set file.

You can manually generate text from a trained model file by running

    python3 -m network.generate <path-to-model-file.pt>


### Hosting the bot

If you are satisfied with the generated text and want to run the bot with it, copy the model to `~/.newsgeneratorbot/content_model.pt`. If you also have trained a model for generating titles, place it in `~/.newsgeneratorbot/title_model.pt`.
Generate your own Telegram bot API token by following the API [documentation](https://core.telegram.org/bots#6-botfather) and place the token into a file called `~/.newsgeneratorbot/token`.
Once you have done all this, you can run the bot with its default settings:

    python3 -m newsgeneratorbot

All of the mentioned commands can be customized in many ways. To get an overview over the available options, run them with the `--help` flag.


## Building a data set for training and validation

For copyright reasons, I can't share the data sets that I created and used, but you can build your own data set easily by scraping websites:

### Scraping news websites

- `cd scripts`
- For each news website `<url>` (e.g. https://rnz.de), run

      ./scrape_articles.py <url>

- For each site name `<site>`, two files `<site>_content.txt` and `<site>_titles.txt` are saved in the CWD, each containing raw articles text contents and titles respectively.


### Preprocessing the texts

- Concatenate all content files and title files to one big raw data set each

      cat *_content.txt > content.txt
      cat *_title.txt > titles.txt

- Clean and prepare files (In the following, you can replace `de` by another language code like `en` if you don't use German texts):

      cd scripts

      # Only keep lines that are in the expected language (here: German)
      # Note that there are always some false negatives that are removed,
      # because this just uses a "dumb" heuristic of language-typical
      # character sequences.
      ./clean_language.py de content.txt
      ./clean_language.py de titles.txt

      # Transliterate unicode characters
      # (with special almost-lossless handling of German umlauts)
      ./transliterate.py content_de_only.txt
      ./transliterate.py titles_de_only.txt

      # Remove all lines that appear multiple times in a file
      # (Those are mostly ads, ad blocker warnings etc.)
      ./deduplicate_lines.py content_de_only_ascii.txt
      ./deduplicate_lines.py titles_de_only_ascii.txt

  The preprocessed texts are stored as `content_de_only_ascii_dedup.txt` and `titles_de_only_ascii_dedup.txt`


### Splitting the data set

- Split data set into training and validation lines

      ./split.py --ratio 0.9 content_de_only_ascii_dedup.txt
      ./split.py --ratio 0.9 titles_de_only_ascii_dedup.txt

  This results in the final data sets for the content generator:

  - `content_de_only_ascii_dedup_train.txt`
  - `content_de_only_ascii_dedup_valid.txt`

  and for the title generator:
  - `titles_de_only_ascii_dedup_train.txt`
  - `titles_de_only_ascii_dedup_valid.txt`


## Credits

The article [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy was a major inspiration for this project.
Parts of the neural network implementation are based on the [Practical PyTorch](https://github.com/spro/practical-pytorch) tutorial "char-rnn-generation" by Sean Robertson.

This is a project for the [Artificial Intelligence lecture 2017/18 at Uni Heidelberg](https://hci.iwr.uni-heidelberg.de/compvis/teaching/ai).


## Outlook

The articles generated by this rather minimalistic neural network pipeline are not very convincing. They suffer from limited awareness of semantic long-term dependencies and generally fail to produce coherent news stories.
The paper [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833) (published after this project was finished) proposes a more sophisticated approach to "inventing" and telling new stories based on a user-defined writing prompt. Judging from the examples in the paper, it achieves an impressive level of coherence even in longer generated stories. I believe that the model proposed in **Hierarchical Neural Story Generation** could be adapted to be used as an alternative backbone of the News Generator Bot in order to greatly enhance the quality of generated stories. An open source implementation and pretrained network models are already available at https://github.com/pytorch/fairseq. I am not sure when I will find the time to pursue this idea, so anyone who is interested in this is welcome to try this out and open an issue about it.
