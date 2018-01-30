#!/usr/bin/env python3

import argparse
import pickle
import traceback

import newspaper
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    'url',
    help='URL of a news website, e.g. https://www.cnn.com.'
)
parser.add_argument(
    '-n',
    type=int,
    help='Maximum number of articles to access.'
)
parser.add_argument(
    '--pickle',
    action='store_true',
    help='If true, store a dict mapping titles to texts in a .pkl file.'
)
args = parser.parse_args()
url = args.url
maxarticles = args.n

# Parse site
print('Building site index... (this can take a minute.)')

# newspaper requires explicit protocol, so make sure to at least prefix http
if not (url.startswith('https://') or url.startswith('http://')):
    url = f'http://{url}'  # https is still not supported everywhere...

site = newspaper.build(url, memoize_articles=False)
assert site.articles
print(f'Found {len(site.articles)} articles on {url} ({site.brand})')

articles = {}
failed_count = 0
empty_count = 0
# Download and parse articles, store in dict (title: content)
for ar in tqdm(site.articles[:maxarticles]):
    try:
        ar.download()
        ar.parse()
        content = ar.text
        if content:
            # Shorten long newline runs
            while '\n\n\n' in content:
                content = content.replace('\n\n\n', '\n')
            articles[ar.title] = content
        else:
            empty_count += 1
    except KeyboardInterrupt as e:
        raise e
    except:
        failed_count += 1
        traceback.print_exc()

# Print statistics
if failed_count > 0:
    print(f'{failed_count} articles could not be loaded.')
if empty_count > 0:
    print(f'{empty_count} articles were empty and were omitted.')
print(f'Successfully retrieved {len(articles)} articles.')


# Store raw articles dict in .pkl
if args.pickle:
    pklfilename = site.brand + '.pkl'
    print('Storing "articles" dict in  {pklfilename}')
    with open(pklfilename, 'wb') as f:
        pickle.dump(articles, f)


# Write article contents and titles to separate files
titlefilename = site.brand + '_titles' + '.txt'
contfilename = site.brand + '_content' '.txt'
titlefile = open(titlefilename, 'w')
contfile = open(contfilename, 'w')
for title, cont in articles.items():
    titlefile.write(title + '\n')
    contfile.write(cont + '\n')
titlefile.close()
contfile.close()
