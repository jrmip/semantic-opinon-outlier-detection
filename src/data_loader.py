from ml_datasets import imdb
from nltk import word_tokenize
from nltk.corpus import reuters
from sklearn.datasets import fetch_20newsgroups

import logging
import pandas as pd

AVAILABLE_DATASET = {
    0: "Reuters-21578",
    1: "20 Newsgroups",
    2: "IMDB Movie Reviews"
}

REUTERS_CATEGORIES = dict(zip(reuters.categories(), range(90)))

NEWSGROUPS_CATEGORIES = {
    'alt.atheism': 0,
    'comp.graphics': 1,
    'comp.os.ms-windows.misc': 2,
    'comp.sys.ibm.pc.hardware': 3,
    'comp.sys.mac.hardware': 4,
    'comp.windows.x': 5,
    'misc.forsale': 6,
    'rec.autos': 7,
    'rec.motorcycles': 8,
    'rec.sport.baseball': 9,
    'rec.sport.hockey': 10,
    'sci.crypt': 11,
    'sci.electronics': 12,
    'sci.med': 13,
    'sci.space': 14,
    'soc.religion.christian': 15,
    'talk.politics.guns': 16,
    'talk.politics.mideast': 17,
    'talk.politics.misc': 18,
    'talk.religion.misc': 19
}

IMDB_CATEGORIES = {"neg": 0, "pos": 1}


class DataLoader:
    def __init__(self):
        logger = logging.getLogger(__name__)

        logger.info(
            "Retrieving dataset and converting them into dataframes (can take "
            "few minutes)."
        )

        self._load_dataset()

        logger.info("Data have been correctly retrieved.")

    def _load_dataset(self):
        self.reuters = {}

        # load reuters with nltk
        documents = reuters.fileids()

        reuters_splits = {"train": "tr", "test": "te"}

        for k, v in reuters_splits.items():
            # init dataframe
            df = pd.DataFrame()
            df['ids'] = [d for d in documents if d.startswith(v)]
            df['raw'] = df['ids'].apply(lambda i: reuters.raw(i))
            df['clean'] = df['raw'].apply(lambda t: self._clean_text(t))
            df['label'] = df['ids'].apply(
                lambda i: [REUTERS_CATEGORIES[e] for e in reuters.categories(i)]
            )

            self.reuters[k] = df

        self.newsgroups = {}

        # we keep body of documents
        rmv = ('headers', 'footers', 'quotes')

        newsgroups_splits = ["train", "test"]

        for split in newsgroups_splits:
            # load newsgroups with sklearn
            fetch = fetch_20newsgroups(subset=split, remove=rmv)

            df = pd.DataFrame()
            df['raw'] = fetch['data']
            df['clean'] = df['raw'].apply(lambda t: self._clean_text(t))
            df['label'] = fetch['target']

            self.newsgroups[split] = df

        self.imdb = {}

        # load imdb with ml_datasets
        tr, te = imdb()

        imdb_splits = {'train': tr, 'test': te}

        for k, v in imdb_splits.items():
            df = pd.DataFrame()
            df['raw'] = [t for t, _ in v]
            df['clean'] = df['raw'].apply(lambda t: self._clean_text(t))
            df['label'] = [IMDB_CATEGORIES[l] for _, l in v]

            self.imdb[k] = df

    @staticmethod
    def _clean_text(text):
        t = word_tokenize(text.lower())

        return " ".join([word for word in t if word.isalnum()])
