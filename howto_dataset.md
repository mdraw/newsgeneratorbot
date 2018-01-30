Building a data set for training and validation of the "news generator"
=======================================================================

Scrape news websites
--------------------

- For each news website `<url>` (e.g. https://rnz.de), run

      python3 scrape_articles.py <url>

- For each site name `<site>`, two files `<site_content.txt>` and `<site_titles.txt>` are saved in the CWD, each containing raw articles text contents and titles respectively.

- Concatenate all content files and title files to one big data set each

      cat *_content.txt > all_content.txt
      cat *_title.txt > all_titles.txt

- Preprocess files (In the following, you can replace `de` by another
  language code like `en` if you don't use German texts.)

      # Only keep lines that are in the expected language (here: German)
      python3 scripts/clean_language.py de all_content.txt
      python3 scripts/clean_language.py de all_titles.txt

      # Transliterate unicode characters
      # (with special almost-lossless handling of German umlauts)
      python3 scripts/transliterate.py all_content_de_only.txt
      python3 scripts/transliterate.py all_titles_de_only.txt

      # Remove all lines that appear multiple times in a file
      # (Those are mostly ads, ad blocker warnings etc.)
      python3 scripts/deduplicate_lines.py all_content_de_only_ascii.txt
      python3 scripts/deduplicate_lines.py all_titles_de_only_ascii.txt

  The preprocessed texts are stored as `all_content_de_only_ascii_dedup.txt`
  and `all_titles_de_only_ascii_dedup.txt`

- TODO Split train/valid
