# News Generator Bot

A Telegram bot that generates custom news articles on demand, based on a GRU Neural Network.


## Overview


## Installation and hosting


## Building a data set for training and validation

### Scraping news websites

- For each news website `<url>` (e.g. https://rnz.de), run

      python3 scrape_articles.py <url>

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
