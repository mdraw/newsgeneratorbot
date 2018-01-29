#!/usr/bin/env python3

import argparse
import pickle

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

print('Building site index... (this can take a minute.)')
site = newspaper.build(url, memoize_articles=False)

assert site.articles
print(f'Found {len(site.articles)} articles. on {url} ({site.brand})')

articles = {}

failed_count = 0
empty_count = 0
for ar in tqdm(site.articles[:maxarticles]):
    try:
        ar.download()
        ar.parse()
        if ar.text:
            articles[ar.title] = ar.text
        else:
            empty_count += 1
    except:
        failed_count += 1

if failed_count > 0:
    print(f'{failed_count} articles could not be loaded.')
if empty_count > 0:
    print(f'{empty_count} articles were empty and were omitted.')
print(f'Successfully retrieved {len(articles)} articles.')


if args.pickle:
    pklfilename = site.brand + '.pkl'
    print('Storing "articles" dict in  {pklfilename}')
    with open(pklfilename, 'wb') as f:
        pickle.dump(articles, f)


titlefilename = site.brand + '_titles' + '.txt'
textfilename = site.brand + '_texts' '.txt'
titlefile = open(titlefilename, 'w')
textfile = open(textfilename, 'w')
for title, text in articles.items():
    titlefile.write(title + '\n')
    textfile.write(text + '\n')
titlefile.close()
textfile.close()
