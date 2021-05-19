from nltk import word_tokenize
from nltk.corpus import reuters, stopwords

from sklearn.utils import shuffle
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from ml_datasets import imdb



import numpy as np
import pandas as pd

AVAILABLE_DATASET = {
    0: "Reuters-21578",
    1: "20 Newsgroups",
    2: "IMDB Movie Reviews"
}

REUTERS_CATEGORIES = ['earn', 'acq', 'money-fx', 'crude', 'trade']

class DataLoader:
    def __init__(self):
        pass

    def _load_dataset(self):
        pass




def clean_text(text):
    t = text.lower()
    t = word_tokenize(t)

    return " ".join([word for word in t if word.isalnum()])


def load_reuters():
    documents = reuters.fileids()

    train = shuffle([d for d in documents if d.startswith("tr")])
    test = shuffle([d for d in documents if d.startswith("te")])

    return {
        "x_train": [clean_text(reuters.raw(doc_id)) for doc_id in train],
        "x_test": [clean_text(reuters.raw(doc_id)) for doc_id in test],
        "y_train": [reuters.categories(doc_id) for doc_id in train],
        "y_test": [reuters.categories(doc_id) for doc_id in test],
        "labels": dict(zip(reuters.categories(), range(90))),
        "x_train_raw": [reuters.raw(doc_id) for doc_id in train],
        "x_test_raw": [reuters.raw(doc_id) for doc_id in test],
    }


def prepare_reuters(
        data,
        split_size=-1,
        contamination=0.1,
        is_train=True,
        raw=False):
    categories = [data['labels'][c] for c in REUTERS_CATEGORIES]

    xy = pd.DataFrame()

    if is_train:
        xy['X'] = data['x_train'] if not raw else data['x_train_raw']
        xy['raw'] = data['x_train_raw']
        xy['Y'] = data['y_train']

        if 'train_emotion' in data.keys():
            xy['emotion'] = data['train_emotion']
    else:
        xy['X'] = data['x_test'] if not raw else data['x_test_raw']
        xy['raw'] = data['x_test_raw']
        xy['Y'] = data['y_test']

        if 'test_emotion' in data.keys():
            xy['emotion'] = data['test_emotion']

    xy['unique'] = xy.Y.apply(lambda y: 1 if len(y) == 1 else 0)

    xy = xy[xy['unique'] == 1]

    xy['Y'] = xy.Y.apply(lambda y: data['labels'][y[0]])

    splits = {}

    for c in categories:
        if split_size == -1:
            tmp = xy[xy['Y'] == c]

            total_outliers = int(contamination * len(tmp))
            if total_outliers == 0:
                total_outliers = 1

            tmp = tmp.sample(len(tmp) - total_outliers)
        else:
            total_outliers = int(contamination * split_size)
            if total_outliers == 0:
                total_outliers = 1

            tmp = xy[xy['Y'] == c].sample(split_size - total_outliers)

        tmp = tmp.append(xy[xy['Y'] != c].sample(total_outliers)).sample(frac=1)

        splits[c] = tmp

    return splits


def load_newsgroups():
    rmv = ('headers', 'footers', 'quotes')

    train = fetch_20newsgroups(subset='train', remove=rmv)
    test = fetch_20newsgroups(subset='test', remove=rmv)

    return {
        "x_train": [clean_text(t) for t in train['data']],
        "x_test": [clean_text(t) for t in test['data']],
        "y_train": train['target'],
        "y_test": test['target'],
        "x_train_raw": [t for t in train['data']],
        "x_test_raw": [t for t in test['data']]
    }


def prepare_newsgroups(
        data,
        split_size=-1,
        contamination=0.1,
        is_train=True,
        raw=False):
    xy = pd.DataFrame()

    if is_train:
        xy['X'] = data['x_train'] if not raw else data['x_train_raw']
        xy['clean'] = data['x_train']
        xy['raw'] = data['x_train_raw']
        xy['Y'] = data['y_train']

        if 'train_emotion' in data.keys():
            xy['emotion'] = data['train_emotion']
    else:
        xy['X'] = data['x_test'] if not raw else data['x_test_raw']
        xy['clean'] = data['x_test']
        xy['raw'] = data['x_test_raw']
        xy['Y'] = data['y_test']

        if 'test_emotion' in data.keys():
            xy['emotion'] = data['test_emotion']

    xy['empty'] = xy['clean'].apply(lambda t: 1 if len(t) == 0 else 0)

    xy = xy[xy['empty'] == 0]

    splits = {}

    for c in xy['Y'].unique():
        if split_size == -1:
            tmp = xy[xy['Y'] == c]

            total_outliers = int(contamination * len(tmp))
            if total_outliers == 0:
                total_outliers = 1

            tmp = tmp.sample(len(tmp) - total_outliers)
        else:
            total_outliers = int(contamination * split_size)
            if total_outliers == 0:
                total_outliers = 1

            tmp = xy[xy['Y'] == c].sample(split_size - total_outliers)

        tmp = tmp.append(xy[xy['Y'] != c].sample(total_outliers)).sample(frac=1)

        splits[c] = tmp

    return splits
